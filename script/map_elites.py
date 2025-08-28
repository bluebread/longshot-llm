#!/usr/bin/env python3
"""
MAP-Elites Algorithm Implementation for Boolean Formula Optimization

This script implements the MAP-Elites (Multi-dimensional Archive of Phenotypic Elites)
algorithm adapted for optimizing boolean formulas in the gym-longshot framework.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import httpx

from longshot.agent import WarehouseAgent, ClusterbombAgent
from longshot.env import FormulaIsodegrees
from longshot.utils import parse_gate_integer_representation


@dataclass
class MAPElitesConfig:
    """Configuration for MAP-Elites algorithm"""
    # Core algorithm parameters
    num_iterations: int = 100
    cell_density: int = 1  # Maximum organisms per cell
    
    # Formula space parameters
    max_num_vars: int = 4
    max_width: int = 3
    max_size: int = 5
    
    # Mutation parameters
    mutate_length: int = 10
    num_mutate: int = 5
    
    # Optional parameters
    batch_size: int = 10
    elite_selection_strategy: str = "uniform"  # uniform, curiosity, performance
    initialization_strategy: str = "warehouse"  # warehouse, random
    
    # Service configuration
    warehouse_host: str = "localhost"
    warehouse_port: int = 8000
    clusterbomb_host: str = "localhost"
    clusterbomb_port: int = 8060
    
    # TrajectoryProcessor configuration
    trajproc_iterations: int = 5  # WL hash iterations
    trajproc_granularity: int = 20  # Q-value discretization granularity
    trajproc_num_summits: int = 5  # Number of summits to consider
    
    # Output
    verbose: bool = True
    save_archive: bool = True
    archive_path: str = "map_elites_archive.json"


@dataclass
class Elite:
    """Represents an elite solution in a MAP-Elites cell"""
    traj_id: str
    traj_slice: int
    avgQ: float
    discovery_iteration: int = 0
    
    def to_dict(self):
        return {
            "traj_id": self.traj_id,
            "traj_slice": self.traj_slice,
            "avgQ": self.avgQ,
            "discovery_iteration": self.discovery_iteration
        }


@dataclass
class MAPElitesArchive:
    """Archive structure for MAP-Elites algorithm"""
    cells: Dict[tuple, List[Elite]] = field(default_factory=dict)
    cell_density: int = 1
    total_evaluations: int = 0
    iteration_discoveries: Dict[int, int] = field(default_factory=dict)
    
    def update_cell(self, cell_id: tuple, elite: Elite) -> bool:
        """Update a cell with a new elite if it improves or adds diversity"""
        if cell_id not in self.cells:
            self.cells[cell_id] = []
        
        cell_elites = self.cells[cell_id]
        
        # Add if under capacity
        if len(cell_elites) < self.cell_density:
            cell_elites.append(elite)
            cell_elites.sort(key=lambda x: x.avgQ, reverse=True)
            return True
        
        # Replace worst if better
        if elite.avgQ > min(e.avgQ for e in cell_elites):
            # Find and replace worst
            min_idx = min(range(len(cell_elites)), key=lambda i: cell_elites[i].avgQ)
            cell_elites[min_idx] = elite
            cell_elites.sort(key=lambda x: x.avgQ, reverse=True)
            return True
        
        return False
    
    def get_statistics(self) -> dict:
        """Get archive statistics"""
        all_elites = [e for cell in self.cells.values() for e in cell]
        if not all_elites:
            return {
                "total_cells": 0,
                "total_elites": 0,
                "avg_avgQ": 0,
                "max_avgQ": 0,
                "min_avgQ": 0
            }
        
        avgQ_values = [e.avgQ for e in all_elites]
        return {
            "total_cells": len(self.cells),
            "total_elites": len(all_elites),
            "avg_avgQ": sum(avgQ_values) / len(avgQ_values),
            "max_avgQ": max(avgQ_values),
            "min_avgQ": min(avgQ_values)
        }
    
    def to_dict(self):
        """Convert archive to dictionary for serialization"""
        return {
            "cells": {str(k): [e.to_dict() for e in v] for k, v in self.cells.items()},
            "cell_density": self.cell_density,
            "total_evaluations": self.total_evaluations,
            "iteration_discoveries": self.iteration_discoveries,
            "statistics": self.get_statistics()
        }


class MAPElites:
    """MAP-Elites algorithm implementation for boolean formula optimization"""
    
    def __init__(self, config: MAPElitesConfig):
        self.config = config
        self.archive = MAPElitesArchive(cell_density=config.cell_density)
        self.trajectories_lookup = {}
        
        # Initialize service clients
        self.warehouse = WarehouseAgent(config.warehouse_host, config.warehouse_port)
        self.clusterbomb = ClusterbombAgent(config.clusterbomb_host, config.clusterbomb_port)
        
        # Track iteration progress
        self.current_iteration = 0
        
    def log(self, message: str):
        """Log message if verbose mode is enabled"""
        if self.config.verbose:
            print(f"[Iteration {self.current_iteration}] {message}")
    
    def initialize_from_warehouse(self):
        """Initialize archive from existing warehouse trajectories"""
        self.log("Downloading trajectory dataset from warehouse...")
        
        try:
            # Get all trajectories from warehouse
            response = self.warehouse._client.get("/trajectory/dataset")
            response.raise_for_status()
            dataset = response.json()
            trajectories = dataset.get("trajectories", [])
            
            self.log(f"Processing {len(trajectories)} trajectories...")
            
            for traj in trajectories:
                self.process_trajectory_for_archive(traj, is_initialization=True)
            
            stats = self.archive.get_statistics()
            self.log(f"Initialization complete. Archive statistics:")
            self.log(f"  - Total cells: {stats['total_cells']}")
            self.log(f"  - Total elites: {stats['total_elites']}")
            self.log(f"  - Avg avgQ: {stats['avg_avgQ']:.4f}")
            self.log(f"  - Max avgQ: {stats['max_avgQ']:.4f}")
            
        except httpx.HTTPStatusError as e:
            self.log(f"Warning: Could not download trajectories: {e}")
            self.log("Starting with empty archive...")
        except Exception as e:
            self.log(f"Error during initialization: {e}")
            self.log("Starting with empty archive...")
    
    def process_trajectory_for_archive(self, trajectory: dict, is_initialization: bool = False):
        """Process a trajectory and update archive with formulas along the path"""
        traj_id = trajectory.get("traj_id", trajectory.get("_id"))
        steps = trajectory.get("steps", [])
        
        # Store trajectory for later reference
        self.trajectories_lookup[traj_id] = trajectory
        
        # Get trajectory constraints if available
        max_num_vars = trajectory.get("max_num_vars", self.config.max_num_vars)
        max_width = trajectory.get("max_width", self.config.max_width)
        max_size = trajectory.get("max_size", self.config.max_size)
        
        # Track formula state incrementally
        fisod = FormulaIsodegrees(max_num_vars, [])
        used_variables = 0
        formula_gates = []
        current_formula_width = 0  # Track current max width in formula
        
        for i, step in enumerate(steps):
            # Parse step (handles both tuple and list formats)
            if isinstance(step, (list, tuple)) and len(step) == 3:
                token_type, litint, cur_avgQ = step
            else:
                # Skip malformed steps
                continue
            
            # Parse literals
            lits = parse_gate_integer_representation(litint)
            
            # Track used variables
            used_variables |= (lits.pos | lits.neg)
            
            # Check constraints
            if used_variables.bit_count() > max_num_vars:
                break
            if token_type == 0 and lits.width > max_width:
                break
            if token_type == 0 and len(formula_gates) >= max_size:
                break
            
            # Update formula state
            if token_type == 0:  # ADD
                fisod.add_gate(litint)
                formula_gates.append(litint)
                # Update max width in formula
                current_formula_width = max(current_formula_width, lits.width)
            elif token_type == 1:  # DELETE
                fisod.remove_gate(litint)
                if litint in formula_gates:
                    formula_gates.remove(litint)
                    # Recalculate max width if needed
                    if formula_gates:
                        current_formula_width = max(
                            parse_gate_integer_representation(g).width 
                            for g in formula_gates
                        )
                    else:
                        current_formula_width = 0
            else:
                continue  # Skip unknown token types
            
            # Only store as elite if the formula is valid for mutation
            # Check that reconstructed formula would be valid for clusterbomb
            if len(formula_gates) > 0 and current_formula_width <= max_width and len(formula_gates) <= max_size:
                # Get cell ID from feature
                cell_id = fisod.feature
                
                # Create elite with validation metadata
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
            # Prefer less-explored cells (cells with fewer elites)
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
            # Select best elite from cell
            elite = max(self.archive.cells[cell], key=lambda e: e.avgQ)
            result.append((cell, elite))
        
        return result
    
    def validate_prefix_trajectory(self, prefix_traj: list) -> bool:
        """Validate that a prefix trajectory produces a valid formula"""
        
        formula_gates = []
        current_width = 0
        used_variables = 0
        
        for step in prefix_traj:
            if isinstance(step, (list, tuple)) and len(step) == 3:
                token_type, litint, _ = step
            else:
                continue
                
            lits = parse_gate_integer_representation(litint)
            used_variables |= (lits.pos | lits.neg)
            
            if token_type == 0:  # ADD
                formula_gates.append(litint)
                current_width = max(current_width, lits.width)
            elif token_type == 1:  # DELETE
                if litint in formula_gates:
                    formula_gates.remove(litint)
                    if formula_gates:
                        current_width = max(
                            parse_gate_integer_representation(g).width 
                            for g in formula_gates
                        )
                    else:
                        current_width = 0
        
        # Check if resulting formula is valid
        return (
            len(formula_gates) > 0 and 
            len(formula_gates) <= self.config.max_size and
            current_width <= self.config.max_width and
            used_variables.bit_count() <= self.config.max_num_vars
        )
    
    def mutate_elite(self, elite: Elite) -> Optional[dict]:
        """Mutate an elite using clusterbomb rollout"""
        # Get the trajectory for this elite
        traj = self.trajectories_lookup.get(elite.traj_id)
        if not traj:
            self.log(f"Warning: Could not find trajectory {elite.traj_id}")
            return None
        
        # Extract prefix trajectory up to the elite's position
        steps = traj.get("steps", [])
        prefix_traj = steps[:elite.traj_slice + 1]
        
        # Validate prefix trajectory before sending
        if not self.validate_prefix_trajectory(prefix_traj):
            self.log(f"Warning: Invalid prefix trajectory for elite {elite.traj_id}, skipping mutation")
            return None
        
        try:
            # Build trajproc_config from configuration
            trajproc_config = {
                "iterations": self.config.trajproc_iterations,
                "granularity": self.config.trajproc_granularity,
                "num_summits": self.config.trajproc_num_summits
            }
            
            # Request mutation via clusterbomb
            response = self.clusterbomb.weapon_rollout(
                num_vars=self.config.max_num_vars,
                width=self.config.max_width,
                size=self.config.max_size,
                steps_per_trajectory=self.config.mutate_length,
                num_trajectories=self.config.num_mutate,
                prefix_traj=prefix_traj,
                early_stop=True,
                trajproc_config=trajproc_config
            )
            
            # The response contains trajectory IDs that were created
            # We need to fetch them from warehouse
            return {
                "num_trajectories": response.num_trajectories,
                "total_steps": response.total_steps
            }
            
        except Exception as e:
            self.log(f"Warning: Mutation failed for elite {elite.traj_id}: {e}")
            return None
    
    def evolution_step(self):
        """Execute one iteration of MAP-Elites evolution"""
        self.log(f"Starting evolution step...")
        
        # Select elites for mutation
        selected_elites = self.select_elites(self.config.batch_size)
        if not selected_elites:
            self.log("No elites available for mutation")
            return
        
        self.log(f"Selected {len(selected_elites)} elites for mutation")
        
        # Mutate each selected elite
        total_new_trajectories = 0
        total_new_steps = 0
        
        for cell_id, elite in selected_elites:
            self.log(f"  Mutating elite from cell {cell_id[:3]}... with avgQ={elite.avgQ:.4f}")
            
            result = self.mutate_elite(elite)
            if result:
                total_new_trajectories += result.get("num_trajectories", 0)
                total_new_steps += result.get("total_steps", 0)
        
        self.log(f"Generated {total_new_trajectories} trajectories with {total_new_steps} total steps")
        
        # Fetch and process new trajectories from warehouse
        # Note: In the real implementation, clusterbomb stores trajectories directly
        # We would need to track which trajectories were just created or poll for new ones
        # For now, we'll fetch recent trajectories
        
        try:
            # Get recent trajectories
            response = self.warehouse._client.get("/trajectory/dataset")
            response.raise_for_status()
            dataset = response.json()
            all_trajectories = dataset.get("trajectories", [])
            
            # Process only trajectories not yet in our lookup
            new_trajectories = [
                t for t in all_trajectories 
                if t.get("traj_id", t.get("_id")) not in self.trajectories_lookup
            ]
            
            self.log(f"Processing {len(new_trajectories)} new trajectories")
            for traj in new_trajectories:
                self.process_trajectory_for_archive(traj)
            
        except Exception as e:
            self.log(f"Warning: Could not fetch new trajectories: {e}")
    
    def run(self):
        """Run the MAP-Elites algorithm"""
        print("\n" + "="*60)
        print("MAP-Elites Algorithm for Boolean Formula Optimization")
        print("="*60)
        print(f"Configuration:")
        print(f"  - Iterations: {self.config.num_iterations}")
        print(f"  - Cell density: {self.config.cell_density}")
        print(f"  - Formula space: {self.config.max_num_vars} vars, width {self.config.max_width}, size {self.config.max_size}")
        print(f"  - Mutation: {self.config.num_mutate} trajectories of {self.config.mutate_length} steps")
        print(f"  - Selection strategy: {self.config.elite_selection_strategy}")
        print("="*60 + "\n")
        
        # Phase 1: Initialize from warehouse
        if self.config.initialization_strategy == "warehouse":
            self.initialize_from_warehouse()
        
        # Phase 2: Evolution loop
        for iteration in range(self.config.num_iterations):
            self.current_iteration = iteration + 1
            print(f"\n--- Iteration {self.current_iteration}/{self.config.num_iterations} ---")
            
            start_time = time.time()
            self.evolution_step()
            elapsed = time.time() - start_time
            
            # Report statistics
            stats = self.archive.get_statistics()
            discoveries = self.archive.iteration_discoveries.get(self.current_iteration, 0)
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  New discoveries: {discoveries}")
            print(f"  Archive size: {stats['total_cells']} cells, {stats['total_elites']} elites")
            print(f"  Performance: avg={stats['avg_avgQ']:.4f}, max={stats['max_avgQ']:.4f}")
        
        # Phase 3: Final report
        print("\n" + "="*60)
        print("MAP-Elites Complete!")
        print("="*60)
        final_stats = self.archive.get_statistics()
        print(f"Final Archive Statistics:")
        print(f"  - Total cells discovered: {final_stats['total_cells']}")
        print(f"  - Total elites: {final_stats['total_elites']}")
        print(f"  - Total evaluations: {self.archive.total_evaluations}")
        print(f"  - Best avgQ: {final_stats['max_avgQ']:.4f}")
        print(f"  - Average avgQ: {final_stats['avg_avgQ']:.4f}")
        
        # Save archive if requested
        if self.config.save_archive:
            self.save_archive()
            print(f"\nArchive saved to: {self.config.archive_path}")
    
    def save_archive(self):
        """Save the archive to a JSON file"""
        archive_data = self.archive.to_dict()
        archive_data["config"] = {
            "num_iterations": self.config.num_iterations,
            "cell_density": self.config.cell_density,
            "max_num_vars": self.config.max_num_vars,
            "max_width": self.config.max_width,
            "max_size": self.config.max_size,
            "mutate_length": self.config.mutate_length,
            "num_mutate": self.config.num_mutate,
            "batch_size": self.config.batch_size,
            "elite_selection_strategy": self.config.elite_selection_strategy,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.config.archive_path, 'w') as f:
            json.dump(archive_data, f, indent=2)
    
    def close(self):
        """Clean up resources"""
        self.warehouse._client.close()
        self.clusterbomb.close()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MAP-Elites for Boolean Formula Optimization")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument("--cell-density", type=int, default=1, help="Max elites per cell")
    parser.add_argument("--num-vars", type=int, default=4, help="Max number of variables")
    parser.add_argument("--width", type=int, default=3, help="Max formula width")
    parser.add_argument("--size", type=int, default=5, help="Max formula size")
    parser.add_argument("--mutate-length", type=int, default=10, help="Steps per mutation")
    parser.add_argument("--num-mutate", type=int, default=5, help="Trajectories per mutation")
    parser.add_argument("--batch-size", type=int, default=10, help="Mutations per iteration")
    parser.add_argument("--strategy", type=str, default="uniform", 
                       choices=["uniform", "curiosity", "performance"],
                       help="Elite selection strategy")
    parser.add_argument("--warehouse-host", type=str, default="localhost", help="Warehouse host")
    parser.add_argument("--warehouse-port", type=int, default=8000, help="Warehouse port")
    parser.add_argument("--clusterbomb-host", type=str, default="localhost", help="Clusterbomb host")
    parser.add_argument("--clusterbomb-port", type=int, default=8060, help="Clusterbomb port")
    parser.add_argument("--trajproc-iterations", type=int, default=5, help="WL hash iterations for TrajectoryProcessor")
    parser.add_argument("--trajproc-granularity", type=int, default=20, help="Q-value discretization granularity for TrajectoryProcessor")
    parser.add_argument("--trajproc-num-summits", type=int, default=5, help="Number of summits for TrajectoryProcessor")
    parser.add_argument("--output", type=str, default="map_elites_archive.json", help="Output file")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Create configuration
    config = MAPElitesConfig(
        num_iterations=args.iterations,
        cell_density=args.cell_density,
        max_num_vars=args.num_vars,
        max_width=args.width,
        max_size=args.size,
        mutate_length=args.mutate_length,
        num_mutate=args.num_mutate,
        batch_size=args.batch_size,
        elite_selection_strategy=args.strategy,
        warehouse_host=args.warehouse_host,
        warehouse_port=args.warehouse_port,
        clusterbomb_host=args.clusterbomb_host,
        clusterbomb_port=args.clusterbomb_port,
        trajproc_iterations=args.trajproc_iterations,
        trajproc_granularity=args.trajproc_granularity,
        trajproc_num_summits=args.trajproc_num_summits,
        archive_path=args.output,
        verbose=not args.quiet
    )
    
    # Run MAP-Elites
    map_elites = MAPElites(config)
    try:
        map_elites.run()
    finally:
        map_elites.close()


if __name__ == "__main__":
    main()