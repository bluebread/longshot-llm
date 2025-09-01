"""
MAP-Elites algorithm service implementation for clusterbomb.
"""

import asyncio
import random
import time
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from multiprocessing import Pool
from functools import partial
import logging

from longshot.service import AsyncWarehouseClient
from longshot.utils import parse_gate_integer_representation

from .models import MAPElitesConfig, MAPElitesArchive, Elite, MAPElitesStatus
from .trajectory_generator import TrajectoryGenerator, run_mutations_sync
from .isodegrees import FormulaIsodegrees

logger = logging.getLogger(__name__)


class MAPElitesService:
    """MAP-Elites algorithm service for boolean formula optimization"""
    
    def __init__(self, config: MAPElitesConfig):
        """Initialize MAP-Elites service"""
        self.config = config
        self.archive = MAPElitesArchive(cell_density=config.cell_density)
        self.trajectories_lookup = {}
        
        # Initialize trajectory generator
        self.trajectory_generator = TrajectoryGenerator({
            "num_vars": config.num_vars,
            "width": config.width,
            "size": config.size
        })
        
        # Track state
        self.current_iteration = 0
        self.is_running = False
        self.start_time = None
        self.last_sync_time = None
        
    def log(self, message: str):
        """Log message if verbose mode is enabled"""
        if self.config.verbose:
            logger.info(f"[Iteration {self.current_iteration}] {message}")
    
    async def initialize_from_warehouse(self):
        """Initialize archive from existing warehouse trajectories"""
        self.log("Downloading trajectory dataset from warehouse...")
        
        try:
            async with AsyncWarehouseClient(
                self.config.warehouse_host,
                self.config.warehouse_port
            ) as warehouse:
                # Get trajectories filtered by configuration
                trajectories = await warehouse.get_trajectory_dataset(
                    num_vars=self.config.num_vars,
                    width=self.config.width
                )
                
                if not trajectories:
                    self.log("No existing trajectories found, generating initial population...")
                    await self.generate_initial_population(warehouse)
                else:
                    self.log(f"Processing {len(trajectories)} trajectories...")
                    for traj in trajectories:
                        self.process_trajectory_for_archive(traj, is_initialization=True)
                
                stats = self.archive.get_statistics()
                self.log(f"Initialization complete. Archive statistics:")
                self.log(f"  - Total cells: {stats['total_cells']}")
                self.log(f"  - Total elites: {stats['total_elites']}")
                if stats['total_elites'] > 0:
                    self.log(f"  - Avg avgQ: {stats['avg_avgQ']:.4f}")
                    self.log(f"  - Max avgQ: {stats['max_avgQ']:.4f}")
                
        except Exception as e:
            self.log(f"Error during initialization: {e}")
            self.log("Starting with empty archive...")
    
    async def generate_initial_population(self, warehouse: AsyncWarehouseClient):
        """Generate initial population when warehouse is empty"""
        initial_trajectories = self.trajectory_generator.generate_initial_trajectories(
            num_trajectories=self.config.batch_size * 2,
            steps_per_trajectory=self.config.num_steps
        )
        
        # Post to warehouse with error handling
        success_count = 0
        for traj in initial_trajectories:
            try:
                posted = await warehouse.post_trajectory(traj)
                if posted:
                    success_count += 1
                self.process_trajectory_for_archive(traj, is_initialization=True)
            except Exception as e:
                self.log(f"Warning: Failed to post initial trajectory: {e}")
        
        self.log(f"Generated {len(initial_trajectories)} initial trajectories, posted {success_count} successfully")
    
    def process_trajectory_for_archive(self, trajectory: dict, is_initialization: bool = False):
        """Process a trajectory and update archive with formulas along the path"""
        traj_id = trajectory.get("traj_id", trajectory.get("_id"))
        steps = trajectory.get("steps", [])
        
        # Store trajectory for later reference
        self.trajectories_lookup[traj_id] = trajectory
        
        # Track formula state incrementally
        fisod = FormulaIsodegrees(self.config.num_vars, [])
        used_variables = 0
        formula_gates = []
        current_formula_width = 0
        
        for i, step in enumerate(steps):
            # Parse step
            if isinstance(step, (list, tuple)) and len(step) == 3:
                token_type, litint, cur_avgQ = step
            else:
                continue
            
            # Parse literals
            lits = parse_gate_integer_representation(litint)
            
            # Track used variables
            used_variables |= (lits.pos | lits.neg)
            
            # Check constraints
            if used_variables.bit_count() > self.config.num_vars:
                break
            if token_type == 0 and lits.width > self.config.width:
                break
            if self.config.size and token_type == 0 and len(formula_gates) >= self.config.size:
                break
            
            # Update formula state
            if token_type == 0:  # ADD
                fisod.add_gate(litint)
                formula_gates.append(litint)
                current_formula_width = max(current_formula_width, lits.width)
            elif token_type == 1:  # DELETE
                fisod.remove_gate(litint)
                if litint in formula_gates:
                    formula_gates.remove(litint)
                    if formula_gates:
                        current_formula_width = max(
                            parse_gate_integer_representation(g).width
                            for g in formula_gates
                        )
                    else:
                        current_formula_width = 0
            else:
                continue
            
            # Only store as elite if formula is valid
            if len(formula_gates) > 0 and current_formula_width <= self.config.width:
                if not self.config.size or len(formula_gates) <= self.config.size:
                    # Get cell ID from feature
                    cell_id = fisod.feature
                    
                    # Create elite
                    elite = Elite(
                        traj_id=traj_id,
                        traj_slice=i,
                        avgQ=cur_avgQ,
                        discovery_iteration=0 if is_initialization else self.current_iteration
                    )
                    
                    # Update archive
                    if self.archive.update_cell(cell_id, elite):
                        if not is_initialization:
                            self.archive.iteration_discoveries[self.current_iteration] = \
                                self.archive.iteration_discoveries.get(self.current_iteration, 0) + 1
        
        self.archive.total_evaluations += len(steps)
    
    async def sync_archive_with_warehouse(self):
        """Sync archive with warehouse to get trajectories from other instances"""
        if not self.config.enable_sync:
            return
        
        self.log("Syncing with warehouse...")
        
        try:
            async with AsyncWarehouseClient(
                self.config.warehouse_host,
                self.config.warehouse_port
            ) as warehouse:
                # Get trajectories added since last sync
                new_trajectories = await warehouse.get_trajectory_dataset(
                    num_vars=self.config.num_vars,
                    width=self.config.width,
                    since=self.last_sync_time
                )
                
                # Process new trajectories
                new_count = 0
                for traj in new_trajectories:
                    traj_id = traj.get("traj_id", traj.get("_id"))
                    if traj_id not in self.trajectories_lookup:
                        self.process_trajectory_for_archive(traj)
                        new_count += 1
                
                self.log(f"Synced {new_count} new trajectories from warehouse")
                self.last_sync_time = datetime.now()
                
        except Exception as e:
            self.log(f"Warning: Sync failed: {e}")
    
    def select_elites(self, num_elites: int) -> List[Tuple[tuple, Elite]]:
        """Select elites for mutation based on strategy"""
        if not self.archive.cells:
            return []
        
        populated_cells = list(self.archive.cells.keys())
        strategy = self.config.elite_selection_strategy
        
        if strategy == "uniform":
            # Uniform random selection
            selected_cells = random.sample(
                populated_cells,
                min(num_elites, len(populated_cells))
            )
        elif strategy == "curiosity":
            # Prefer less-explored cells
            cell_sizes = [(cell, len(self.archive.cells[cell])) for cell in populated_cells]
            cell_sizes.sort(key=lambda x: x[1])
            selected_cells = [cell for cell, _ in cell_sizes[:num_elites]]
        elif strategy == "performance":
            # Bias toward high-performing cells
            cell_performance = [
                (cell, max(e.avgQ for e in self.archive.cells[cell]))
                for cell in populated_cells
            ]
            cell_performance.sort(key=lambda x: x[1], reverse=True)
            selected_cells = [cell for cell, _ in cell_performance[:num_elites]]
        else:
            # Default to uniform
            selected_cells = random.sample(
                populated_cells,
                min(num_elites, len(populated_cells))
            )
        
        # Return cell-elite pairs
        result = []
        for cell in selected_cells:
            # Select random elite from cell (could also select best)
            elite = random.choice(self.archive.cells[cell])
            result.append((cell, elite))
        
        return result
    
    @staticmethod
    def run_mutation_job(args: Tuple) -> List[Dict]:
        """Static worker function for parallel mutation execution"""
        prefix_traj, config_dict = args
        
        # Run mutations with the provided prefix
        return run_mutations_sync(
            num_vars=config_dict["num_vars"],
            width=config_dict["width"],
            num_trajectories=config_dict["num_trajectories"],
            steps_per_trajectory=config_dict["num_steps"],
            prefix_traj=prefix_traj,
            early_stop=True,
            size=config_dict.get("size")
        )
    
    async def mutate_elites(self, selected_elites: List[Tuple[tuple, Elite]]) -> List[Dict]:
        """Mutate selected elites using multiprocessing"""
        # Prepare mutation jobs with serializable data
        mutation_jobs = []
        config_dict = {
            "num_vars": self.config.num_vars,
            "width": self.config.width,
            "num_trajectories": self.config.num_trajectories,
            "num_steps": self.config.num_steps,
            "size": self.config.size
        }
        
        for cell_id, elite in selected_elites:
            # Get trajectory and extract prefix
            traj = self.trajectories_lookup.get(elite.traj_id)
            if not traj:
                continue
            
            steps = traj.get("steps", [])
            prefix_traj = steps[:elite.traj_slice + 1]
            
            # Validate prefix before adding to jobs
            if self.trajectory_generator.validate_trajectory_prefix(prefix_traj):
                mutation_jobs.append((prefix_traj, config_dict))
        
        if not mutation_jobs:
            return []
        
        # Use multiprocessing for parallel execution
        num_workers = self.config.num_workers
        with Pool(processes=num_workers) as pool:
            try:
                mutation_results = pool.map(MAPElitesService.run_mutation_job, mutation_jobs)
            finally:
                pool.close()
                pool.join()
        
        # Flatten results
        new_trajectories = []
        for trajectories in mutation_results:
            new_trajectories.extend(trajectories)
        
        return new_trajectories
    
    async def evolution_step(self):
        """Execute one iteration of MAP-Elites evolution"""
        self.log("Starting evolution step...")
        
        # Optional sync with warehouse
        if self.config.enable_sync and self.current_iteration % self.config.sync_interval == 0:
            await self.sync_archive_with_warehouse()
        
        # Select elites for mutation
        selected_elites = self.select_elites(self.config.batch_size)
        if not selected_elites:
            self.log("No elites available for mutation")
            return
        
        self.log(f"Selected {len(selected_elites)} elites for mutation")
        
        # Mutate elites
        new_trajectories = await self.mutate_elites(selected_elites)
        self.log(f"Generated {len(new_trajectories)} new trajectories")
        
        # Post to warehouse and update archive
        async with AsyncWarehouseClient(
            self.config.warehouse_host,
            self.config.warehouse_port
        ) as warehouse:
            # Post trajectories
            post_tasks = []
            for traj in new_trajectories:
                post_tasks.append(warehouse.post_trajectory(traj))
            
            # Execute posts concurrently
            results = await asyncio.gather(*post_tasks, return_exceptions=True)
            
            # Count successes
            successes = sum(1 for r in results if r is True)
            if successes < len(new_trajectories):
                self.log(f"Warning: Only {successes}/{len(new_trajectories)} trajectories posted successfully")
        
        # Process new trajectories in archive
        for traj in new_trajectories:
            self.process_trajectory_for_archive(traj)
    
    async def run(self):
        """Run the MAP-Elites algorithm"""
        self.is_running = True
        self.start_time = datetime.now()
        
        self.log("="*60)
        self.log("MAP-Elites Algorithm for Boolean Formula Optimization")
        self.log("="*60)
        self.log(f"Configuration:")
        self.log(f"  - Iterations: {self.config.num_iterations}")
        self.log(f"  - Cell density: {self.config.cell_density}")
        self.log(f"  - Formula space: {self.config.num_vars} vars, width {self.config.width}, size {self.config.size}")
        self.log(f"  - Mutation: {self.config.num_trajectories} trajectories of {self.config.num_steps} steps")
        self.log(f"  - Selection strategy: {self.config.elite_selection_strategy}")
        self.log(f"  - Sync enabled: {self.config.enable_sync}")
        self.log("="*60)
        
        # Initialize archive
        await self.initialize_from_warehouse()
        
        # Evolution loop
        for iteration in range(self.config.num_iterations):
            self.current_iteration = iteration + 1
            self.log(f"\n--- Iteration {self.current_iteration}/{self.config.num_iterations} ---")
            
            start_time = time.time()
            await self.evolution_step()
            elapsed = time.time() - start_time
            
            # Report statistics
            stats = self.archive.get_statistics()
            discoveries = self.archive.iteration_discoveries.get(self.current_iteration, 0)
            
            self.log(f"  Time: {elapsed:.2f}s")
            self.log(f"  New discoveries: {discoveries}")
            self.log(f"  Archive size: {stats['total_cells']} cells, {stats['total_elites']} elites")
            if stats['total_elites'] > 0:
                self.log(f"  Performance: avg={stats['avg_avgQ']:.4f}, max={stats['max_avgQ']:.4f}")
        
        # Final report
        self.log("\n" + "="*60)
        self.log("MAP-Elites Complete!")
        self.log("="*60)
        final_stats = self.archive.get_statistics()
        self.log(f"Final Archive Statistics:")
        self.log(f"  - Total cells discovered: {final_stats['total_cells']}")
        self.log(f"  - Total elites: {final_stats['total_elites']}")
        self.log(f"  - Total evaluations: {self.archive.total_evaluations}")
        if final_stats['total_elites'] > 0:
            self.log(f"  - Best avgQ: {final_stats['max_avgQ']:.4f}")
            self.log(f"  - Average avgQ: {final_stats['avg_avgQ']:.4f}")
        
        # Save archive if requested
        if self.config.save_archive:
            self.save_archive()
            self.log(f"\nArchive saved to: {self.config.archive_path}")
        
        self.is_running = False
    
    def save_archive(self):
        """Save the archive to a JSON file"""
        archive_data = self.archive.to_dict()
        archive_data["config"] = {
            "num_iterations": self.config.num_iterations,
            "cell_density": self.config.cell_density,
            "num_vars": self.config.num_vars,
            "width": self.config.width,
            "size": self.config.size,
            "num_steps": self.config.num_steps,
            "num_trajectories": self.config.num_trajectories,
            "batch_size": self.config.batch_size,
            "elite_selection_strategy": self.config.elite_selection_strategy,
            "enable_sync": self.config.enable_sync,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.config.archive_path, 'w') as f:
            json.dump(archive_data, f, indent=2)
    
    def get_status(self) -> MAPElitesStatus:
        """Get current status of MAP-Elites execution"""
        return MAPElitesStatus(
            is_running=self.is_running,
            current_iteration=self.current_iteration,
            total_iterations=self.config.num_iterations,
            archive_stats=self.archive.get_statistics(),
            last_sync_time=self.last_sync_time,
            start_time=self.start_time,
            config={
                "num_vars": self.config.num_vars,
                "width": self.config.width,
                "size": self.config.size,
                "enable_sync": self.config.enable_sync
            }
        )