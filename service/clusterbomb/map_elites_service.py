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

from models import MAPElitesConfig, MAPElitesArchive, Elite, MAPElitesStatus
from trajectory_generator import TrajectoryGenerator, run_mutations_sync
from isodegrees import FormulaIsodegrees

logger = logging.getLogger(__name__)


class MAPElitesService:
    """MAP-Elites algorithm service for boolean formula optimization"""
    
    def __init__(self, config: MAPElitesConfig):
        """
        Initialize MAP-Elites service with configuration.
        
        Sets up the MAP-Elites algorithm service with the provided configuration,
        initializes the archive for storing elites, and creates a trajectory generator
        for internal mutations. The service tracks execution state and maintains a
        lookup table for efficient trajectory access.
        
        Args:
            config: MAPElitesConfig object containing all algorithm parameters including:
                - num_iterations: Number of evolution iterations to run
                - cell_density: Maximum elites per archive cell
                - num_vars: Number of boolean variables in formulas
                - width: Maximum formula width constraint
                - enable_sync: Whether to sync with other instances via warehouse
                - And other algorithm configuration parameters
        
        Attributes initialized:
            self.config: Stores the configuration
            self.archive: MAPElitesArchive for storing discovered elites
            self.trajectories_lookup: Dict mapping trajectory IDs to trajectory data
            self.trajectory_generator: Internal generator for creating mutations
            self.current_iteration: Current iteration counter (starts at 0)
            self.is_running: Flag indicating if algorithm is currently executing
            self.start_time: Timestamp when execution started
            self.last_sync_time: Last time archive was synced with warehouse
        """
        self.config = config
        self.archive = MAPElitesArchive(cell_density=config.cell_density)
        self.trajectories_lookup = {}
        
        # Initialize trajectory generator
        self.trajectory_generator = TrajectoryGenerator({
            "num_vars": config.num_vars,
            "width": config.width
        })
        
        # Track state
        self.current_iteration = 0
        self.is_running = False
        self.start_time = None
        self.last_sync_time = None
        
    def log(self, message: str):
        """
        Log message if verbose mode is enabled.
        
        Utility method for conditional logging based on the verbose configuration setting.
        Messages are prefixed with the current iteration number for context. This helps
        track algorithm progress and debug issues during execution.
        
        Args:
            message: The message string to log
        
        Note:
            Only logs when self.config.verbose is True
            Uses logger.info level for all messages
            Format: "[Iteration X] message"
        """
        if self.config.verbose:
            logger.info(f"[Iteration {self.current_iteration}] {message}")
    
    async def initialize_from_warehouse(self):
        """
        Initialize archive from existing warehouse trajectories.
        
        Asynchronously downloads existing trajectories from the warehouse service that match
        the current configuration (num_vars, width) and processes them to populate the initial
        archive. If no trajectories exist, generates an initial population using random
        mutations. This provides a warm start for the MAP-Elites algorithm by leveraging
        previously discovered solutions.
        
        The method handles three scenarios:
        1. Warehouse has existing trajectories: Downloads and processes them
        2. Warehouse is empty: Generates initial random trajectories
        3. Connection error: Starts with empty archive and logs warning
        
        Side effects:
            - Populates self.archive with discovered elites
            - Updates self.trajectories_lookup with trajectory data
            - Posts new trajectories to warehouse if generating initial population
            - Logs initialization statistics
        
        Raises:
            No exceptions raised - errors are caught and logged
        """
        self.log("Downloading trajectory dataset from warehouse...")
        
        try:
            async with AsyncWarehouseClient(
                self.config.warehouse_host,
                self.config.warehouse_port
            ) as warehouse:
                # Get trajectories filtered by configuration
                dataset_response = await warehouse.get_trajectory_dataset(
                    num_vars=self.config.num_vars,
                    width=self.config.width
                )
                
                # Extract trajectories from response
                trajectories = []
                if isinstance(dataset_response, dict):
                    trajectories = dataset_response.get('trajectories', [])
                elif isinstance(dataset_response, list):
                    trajectories = dataset_response
                
                # TODO: validate trajectories here
                
                if not trajectories:
                    self.log("No existing trajectories found, generating initial population...")
                    await self.generate_initial_population(warehouse)
                else:
                    self.log(f"Processing {len(trajectories)} trajectories...")
                    for traj in trajectories:
                        # Skip if trajectory is just an ID string
                        if isinstance(traj, str):
                            self.log(f"Warning: Skipping trajectory ID string: {traj}")
                            continue
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
        """
        Generate initial population when warehouse is empty.
        
        Creates an initial set of random trajectories to bootstrap the MAP-Elites algorithm
        when no existing trajectories are available in the warehouse. Generates twice the
        batch size to ensure sufficient initial diversity. Each generated trajectory is
        posted to the warehouse and processed into the archive.
        
        Args:
            warehouse: AsyncWarehouseClient instance for posting trajectories
        
        Process:
            1. Generate random trajectories starting from empty formulas
            2. Post each trajectory to warehouse (with error handling)
            3. Process trajectories into archive to extract elites
            4. Log generation statistics
        
        Side effects:
            - Posts new trajectories to warehouse
            - Updates self.archive with generated elites
            - Updates self.trajectories_lookup
        """
        initial_trajectories = self.trajectory_generator.generate_initial_trajectories(
            num_trajectories=self.config.batch_size * 2,
            steps_per_trajectory=self.config.num_steps
        )
        
        # Post to warehouse with error handling
        success_count = 0
        for traj in initial_trajectories:
            try:
                # Post trajectory using keyword arguments (returns trajectory ID on success)
                posted = await warehouse.post_trajectory(**traj)
                if isinstance(posted, str):  # Returns trajectory ID string on success
                    success_count += 1
                self.process_trajectory_for_archive(traj, is_initialization=True)
            except Exception as e:
                self.log(f"Warning: Failed to post initial trajectory: {e}")
        
        self.log(f"Generated {len(initial_trajectories)} initial trajectories, posted {success_count} successfully")
    
    def process_trajectory_for_archive(self, trajectory: dict, is_initialization: bool = False):
        """
        Process a trajectory and update archive with formulas along the path.
        
        Incrementally processes each step in a trajectory to extract elite solutions at
        different formula states. Uses FormulaIsodegrees to compute isomorphism-invariant
        features that serve as cell IDs in the archive. Only stores valid formulas that
        satisfy all constraints (num_vars, width, size).
        
        Args:
            trajectory: Dictionary containing:
                - traj_id: Unique identifier for the trajectory
                - steps: List of (token_type, litint, avgQ) tuples
                - Optional: num_vars, width, size constraints
            is_initialization: If True, marks elites as discovered during initialization
                              (iteration 0), otherwise uses current_iteration
        
        Process for each step:
            1. Parse token type and literals from step
            2. Update formula state (add/remove gates)
            3. Check constraints (variables used, width, size)
            4. Compute feature using FormulaIsodegrees
            5. Create elite and attempt archive update
            6. Track discovery statistics
        
        Side effects:
            - Updates self.archive with discovered elites
            - Updates self.trajectories_lookup with trajectory reference
            - Increments archive.total_evaluations
            - Updates archive.iteration_discoveries if not initialization
        
        Constraints checked:
            - Number of variables used <= config.num_vars
            - Formula width <= config.width
            - Formula must be non-empty (at least one gate)
        """
        # TODO: validate trajectories in other places instead of here
        traj_id = trajectory.get("traj_id", trajectory.get("_id"))
        steps = trajectory.get("steps", [])
        
        # Store trajectory for later reference
        self.trajectories_lookup[traj_id] = trajectory
        
        # Track formula state incrementally
        fisod = FormulaIsodegrees(self.config.num_vars, [])
        used_variables = 0
        
        for i, step in enumerate(steps):
            # Parse step
            token_type, litint, cur_avgQ = step
            
            # Parse literals
            lits = parse_gate_integer_representation(litint)
            
            # Track used variables
            used_variables |= (lits.pos | lits.neg)
            
            # Update formula state
            if token_type == 0:  # ADD
                fisod.add_gate(litint)
            elif token_type == 1:  # DELETE
                fisod.remove_gate(litint)
            
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
            if self.archive.update_cell(cell_id, elite) and not is_initialization:
                self.archive.iteration_discoveries[self.current_iteration] = \
                    self.archive.iteration_discoveries.get(self.current_iteration, 0) + 1
        
        self.archive.total_evaluations += len(steps)
    
    async def sync_archive_with_warehouse(self):
        """
        Sync archive with warehouse to get trajectories from other instances.
        
        Enables collaborative exploration by periodically downloading new trajectories
        from the warehouse that were created by other MAP-Elites instances. Only
        retrieves trajectories added since the last sync time to minimize data transfer.
        This allows multiple instances to share discoveries and accelerate convergence.
        
        Only executes if:
            - config.enable_sync is True
            - Called at sync_interval iterations
        
        Process:
            1. Connect to warehouse service
            2. Query trajectories added since last_sync_time
            3. Filter by num_vars and width configuration
            4. Process new trajectories into local archive
            5. Update last_sync_time
        
        Side effects:
            - Updates self.archive with trajectories from other instances
            - Updates self.trajectories_lookup with new trajectory data
            - Updates self.last_sync_time to current time
            - Logs sync statistics
        
        Error handling:
            - Connection failures are caught and logged
            - Sync failures don't interrupt algorithm execution
            - Returns silently if sync is disabled
        """
        if not self.config.enable_sync:
            return
        
        self.log("Syncing with warehouse...")
        host = self.config.warehouse_host
        port = self.config.warehouse_port
        
        try:
            async with AsyncWarehouseClient(host, port) as warehouse:
                # Get trajectories added since last sync
                dataset_response = await warehouse.get_trajectory_dataset(
                    num_vars=self.config.num_vars,
                    width=self.config.width,
                    since=self.last_sync_time
                )
                
                # Extract trajectories from response
                new_trajectories = []
                if isinstance(dataset_response, dict):
                    new_trajectories = dataset_response.get('trajectories', [])
                elif isinstance(dataset_response, list):
                    new_trajectories = dataset_response
                
                # TODO: validate trajectories here
                
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
        """
        Select elites for mutation based on configured strategy.
        
        Implements different selection strategies for choosing which elites to mutate
        in the next iteration. The strategy affects exploration vs exploitation balance
        and convergence behavior of the algorithm.
        
        Args:
            num_elites: Number of elites to select for mutation (typically batch_size)
        
        Returns:
            List of (cell_id, elite) tuples selected for mutation
            Returns empty list if archive is empty
        
        Selection strategies:
            - "uniform": Random uniform selection from all populated cells
                        Provides balanced exploration across the feature space
            - "curiosity": Prefers cells with fewer elites (less explored regions)
                          Encourages exploration of sparse areas in the archive  
            - "performance": Biases selection toward high-performing cells
                            Encourages exploitation of promising regions
        
        Implementation details:
            - Each cell contributes at most one elite (randomly selected from cell)
            - Number returned may be less than num_elites if archive has fewer cells
            - Falls back to uniform strategy if unknown strategy specified
        """
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
        """
        Static worker function for parallel mutation execution.
        
        Executes trajectory mutations in a separate process for parallelization.
        Must be static to be serializable for multiprocessing. Takes a prefix
        trajectory and generates new trajectories by continuing from that point.
        
        Args:
            args: Tuple containing:
                - prefix_traj: List of (token_type, litint, avgQ) steps to start from
                - config_dict: Dictionary with mutation parameters:
                    - num_vars: Number of variables
                    - width: Width constraint
                    - num_trajectories: Number of mutations to generate
                    - num_steps: Steps per mutation
        
        Returns:
            List of trajectory dictionaries with generated mutations
        
        Note:
            - Static method required for multiprocessing Pool.map()
            - Runs in separate process without access to instance state
            - Uses run_mutations_sync for thread-safe execution
        """
        prefix_traj, config_dict = args
        
        # Run mutations with the provided prefix
        return run_mutations_sync(
            num_vars=config_dict["num_vars"],
            width=config_dict["width"],
            num_trajectories=config_dict["num_trajectories"],
            steps_per_trajectory=config_dict["num_steps"],
            prefix_traj=prefix_traj,
            early_stop=True
        )
    
    async def mutate_elites(self, selected_elites: List[Tuple[tuple, Elite]]) -> List[Dict]:
        """
        Mutate selected elites using multiprocessing for parallelization.
        
        Takes selected elites and generates new trajectories by mutating from their
        positions. Uses multiprocessing Pool to parallelize mutations across CPU cores
        for improved performance. Each elite's trajectory is truncated at the elite's
        position and used as a prefix for generating new trajectories.
        
        Args:
            selected_elites: List of (cell_id, elite) tuples to mutate
        
        Returns:
            List of newly generated trajectory dictionaries
            Returns empty list if no valid mutations could be created
        
        Process:
            1. Extract prefix trajectory for each elite up to its position
            2. Validate prefix produces valid formula
            3. Prepare serializable job data for multiprocessing
            4. Distribute jobs across worker processes
            5. Collect and flatten results
        
        Performance notes:
            - Uses multiprocessing.Pool for true parallelism
            - Number of workers controlled by config.num_workers
            - Jobs distributed evenly across available cores
            - Pool properly closed and joined to prevent resource leaks
        
        Error handling:
            - Invalid prefixes are skipped
            - Missing trajectories are ignored
            - Pool cleanup happens even if errors occur
        """
        # Prepare mutation jobs with serializable data
        mutation_jobs = []
        config_dict = {
            "num_vars": self.config.num_vars,
            "width": self.config.width,
            "num_trajectories": self.config.num_trajectories,
            "num_steps": self.config.num_steps
        }
        
        for _, elite in selected_elites:
            # Get trajectory and extract prefix
            traj = self.trajectories_lookup.get(elite.traj_id)
            if not traj:
                continue
            
            steps = traj.get("steps", [])
            prefix_traj = steps[:elite.traj_slice + 1]
            mutation_jobs.append((prefix_traj, config_dict))
        
        if not mutation_jobs:
            return []
        
        # Use multiprocessing for parallel execution
        num_workers = self.config.num_workers
        with Pool(processes=num_workers) as pool:
            # Pool.map blocks until all results are ready
            # The context manager automatically handles close() and join()
            mutation_results = pool.map(MAPElitesService.run_mutation_job, mutation_jobs)
        
        # Flatten results
        new_trajectories = []
        for trajectories in mutation_results:
            new_trajectories.extend(trajectories)
        
        return new_trajectories
    
    async def evolution_step(self):
        """
        Execute one iteration of MAP-Elites evolution.
        
        Performs a single iteration of the MAP-Elites algorithm including elite selection,
        mutation, trajectory collection, and archive update. Optionally syncs with warehouse
        for multi-instance collaboration based on configuration.
        
        Process:
            1. Sync with warehouse (if enabled and at sync interval)
            2. Select elites for mutation based on strategy
            3. Generate mutations in parallel using multiprocessing
            4. Post new trajectories to warehouse
            5. Process trajectories into archive
            6. Update discovery statistics
        
        Side effects:
            - Updates self.archive with new elites
            - Posts trajectories to warehouse
            - May sync archive with other instances
            - Logs iteration progress
        
        Error handling:
            - Failed trajectory posts are logged but don't stop execution
            - Returns early if no elites available for mutation
            - Warehouse connection errors are caught and logged
        """
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
        host = self.config.warehouse_host
        port = self.config.warehouse_port
        
        async with AsyncWarehouseClient(host, port) as warehouse:
            # Post trajectories
            post_tasks = []
            for traj in new_trajectories:
                post_tasks.append(warehouse.post_trajectory(**traj))
            
            # Execute posts concurrently
            results = await asyncio.gather(*post_tasks, return_exceptions=True)
            
            # Count successes (post_trajectory returns trajectory ID string on success)
            successes = sum(1 for r in results if isinstance(r, str) and not isinstance(r, Exception))
            if successes < len(new_trajectories):
                self.log(f"Warning: Only {successes}/{len(new_trajectories)} trajectories posted successfully")
        
        # Process new trajectories in archive
        for traj in new_trajectories:
            self.process_trajectory_for_archive(traj)
    
    async def run(self):
        """
        Run the complete MAP-Elites algorithm.
        
        Main entry point for executing the MAP-Elites algorithm. Manages the full lifecycle
        from initialization through evolution iterations to final reporting. Handles both
        standalone and collaborative modes based on configuration.
        
        Execution phases:
            1. Initialization: Load existing trajectories or generate initial population
            2. Evolution: Run configured number of iterations
            3. Reporting: Generate final statistics and save archive
        
        Configuration parameters used:
            - num_iterations: Total iterations to run
            - enable_sync: Whether to sync with other instances
            - save_archive: Whether to save final archive to file
            - verbose: Controls logging verbosity
        
        Side effects:
            - Sets self.is_running flag during execution
            - Updates self.start_time at beginning
            - Modifies archive throughout execution
            - Posts trajectories to warehouse
            - Saves archive to file if configured
            - Extensive logging of progress
        
        Lifecycle management:
            - Sets is_running=True at start, False at end
            - Tracks execution time via start_time
            - Updates current_iteration throughout
        """
        self.is_running = True
        self.start_time = datetime.now()
        
        self.log("="*60)
        self.log("MAP-Elites Algorithm for Boolean Formula Optimization")
        self.log("="*60)
        self.log(f"Configuration:")
        self.log(f"  - Iterations: {self.config.num_iterations}")
        self.log(f"  - Cell density: {self.config.cell_density}")
        self.log(f"  - Formula space: {self.config.num_vars} vars, width {self.config.width}")
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
        """
        Save the archive to a JSON file.
        
        Serializes the current archive state to JSON format and saves to disk.
        Includes both the archive data (cells, elites, statistics) and the
        configuration used for this run. Useful for analysis, visualization,
        and resuming experiments.
        
        File structure:
            - cells: Dictionary of cell_id -> list of elites
            - cell_density: Maximum elites per cell
            - total_evaluations: Total trajectory steps processed
            - iteration_discoveries: Discoveries per iteration
            - statistics: Final archive statistics
            - config: Configuration parameters used
            - timestamp: When archive was saved
        
        Output location:
            Saves to self.config.archive_path (default: "map_elites_archive.json")
        
        Note:
            - Cell IDs are converted to strings for JSON compatibility
            - Elite objects are converted to dictionaries
            - Datetime objects are converted to ISO format strings
        """
        archive_data = self.archive.to_dict()
        archive_data["config"] = {
            "num_iterations": self.config.num_iterations,
            "cell_density": self.config.cell_density,
            "num_vars": self.config.num_vars,
            "width": self.config.width,
            "num_steps": self.config.num_steps,
            "num_trajectories": self.config.num_trajectories,
            "batch_size": self.config.batch_size,
            "elite_selection_strategy": self.config.elite_selection_strategy,
            "enable_sync": self.config.enable_sync,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(self.config.archive_path), exist_ok=True)
        
        with open(self.config.archive_path, 'w') as f:
            json.dump(archive_data, f, separators=(',', ':'))  # Compact mode
    
    def get_status(self) -> MAPElitesStatus:
        """
        Get current status of MAP-Elites execution.
        
        Returns a comprehensive status object containing current execution state,
        progress metrics, and archive statistics. Useful for monitoring long-running
        executions and debugging.
        
        Returns:
            MAPElitesStatus object containing:
                - is_running: Whether algorithm is currently executing
                - current_iteration: Current iteration number (0-based)
                - total_iterations: Total iterations configured
                - archive_stats: Statistics about discovered elites
                - last_sync_time: When archive was last synced (if enabled)
                - start_time: When execution began
                - config: Key configuration parameters
        
        Archive statistics include:
            - total_cells: Number of unique cells discovered
            - total_elites: Total elite solutions in archive
            - avg_avgQ: Average performance across all elites
            - max_avgQ: Best performance found
            - min_avgQ: Worst performance in archive
        
        Use cases:
            - Monitoring progress during execution
            - Checking if algorithm is still running
            - Analyzing archive growth over time
            - Debugging performance issues
        """
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
                "enable_sync": self.config.enable_sync
            }
        )