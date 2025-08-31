# MAP-Elites Algorithm

## Overview

This document outlines the implementation of the **MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) algorithm** integrated directly into the clusterbomb service for optimizing boolean formulas. MAP-Elites is a quality-diversity algorithm that maintains a collection of diverse, high-performing solutions organized in a multi-dimensional feature space.

## Background: MAP-Elites Algorithm

MAP-Elites, introduced by Mouret and Clune (2015), differs from traditional evolutionary algorithms by:
- **Quality-Diversity Optimization**: Instead of seeking a single optimal solution, it maintains an archive of diverse high-performing solutions
- **Feature-Based Organization**: Solutions are mapped to cells in a discretized behavioral/feature space
- **Elite Preservation**: Each cell retains only the highest-performing solution for that feature combination
- **Exploration vs Exploitation**: Naturally balances finding novel solutions with improving existing ones

### Core Algorithm Components

1. **Archive/Map**: Multi-dimensional grid where each cell contains at most one elite solution
2. **Feature Mapping**: Function that maps solutions to behavioral characteristics (cells)
3. **Performance Metric**: Fitness/quality measure for comparing solutions within the same cell
4. **Mutation Operators**: Mechanisms for generating new candidate solutions from existing elites

## Implementation Architecture

### Feature Space Design

In our implementation, each boolean formula is characterized by its `FormulaIsodegrees`:
- **Primary Mapping**: FormulaIsodegrees → Cell (isomorphism-invariant representation)
- **Cell Identity**: The sorted tuple of literal occurrence counts serves as the cell identifier
- **Uniqueness**: Isomorphic formulas map to the same cell, ensuring structural diversity

### Algorithm Parameters

```python
class MAPElitesConfig:
    # Core algorithm parameters
    num_iterations: int          # Total number of iterations to run
    cell_density: int            # Maximum organisms per cell (elites)
    
    # Formula space parameters
    num_vars: int                # Number of boolean variables
    width: int                   # Formula width constraint
    
    # Mutation parameters
    num_steps: int               # Steps to run in clusterbomb simulation
    num_trajectories: int        # Number of trajectories to collect per mutation
    
    # Machine-related parameters
    num_workers: int | None       # Number of parallel workers (None for using all cores)

    # Optional parameters
    batch_size: int = 10         # Number of parallel mutations
    elite_selection_strategy: str = "uniform"  # How to select elites
    initialization_strategy: str = "warehouse"  # Source of initial population
```

## Algorithm Process Flow

### Phase 1: Initialization

1. **Download Existing Data**
   ```python
   async with AsyncWarehouseClient() as warehouse:
       trajectories = await warehouse.get_trajectory_dataset(
           num_vars=config.num_vars,
           width=config.width
       )
   ```
   - Retrieve relevant trajectories filtered by `num_vars` and `width` from the warehouse service
   - Extract formulas and their performance metrics (avgQ values)
   - Filtering ensures only compatible trajectories are loaded for the current configuration

2. **Build Initial Archive**
   ```python
    fisod = FormulaIsodegrees(config.num_vars, [])
    archive = defaultdict(list)

    for trajectory in trajectories:
        tid = trajectory['traj_id']
        used_variables = 0

        for i, step in enumerate(trajectory['steps']):
            token_type, litint, cur_avgQ = step

            # Record used variables
            lits = parse_gate_integer_representation(litint)
            used_variables |= (lits.pos | lits.neg)

            # Break the loop if it violates the constraints
            if used_variables.bit_length() > config.num_vars:
                break
            if token_type == 0 and lits.width > config.width:
                break

            if token_type == 0:
                fisod.add_gate(litint)
            elif token_type == 1:
                fisod.remove_gate(litint)
            else:
                raise Exception()

            cell_id = fisod.feature  # Immutable tuple identifier
           
            # Initialize cell if needed
            if cell_id not in archive:
                archive[cell_id] = []
            
            cell_elites = archive[cell_id]
            
            # Add if under capacity
            if len(cell_elites) < config.cell_density:
                cell_elites.append(Elite(
                    traj_id=tid,
                    traj_slice=i,
                    avgQ=cur_avgQ
                ))
                cell_elites.sort(key=lambda x: x.avgQ, reverse=True)
            # Replace worst if better and at capacity
            elif cur_avgQ > min(e.avgQ for e in cell_elites):
                # Find and replace worst elite
                min_idx = min(range(len(cell_elites)), 
                             key=lambda i: cell_elites[i].avgQ)
                cell_elites[min_idx] = Elite(
                    traj_id=tid,
                    traj_slice=i,
                    avgQ=cur_avgQ
                )
                cell_elites.sort(key=lambda x: x.avgQ, reverse=True)
   ```
   - Process trajectories incrementally to build FormulaIsodegrees
   - Map formulas to cells based on their feature representation
   - Maintain up to `cell_density` elites per cell (sorted by performance)

### Phase 2: Main Evolution Loop

Repeat for `num_iterations`:

#### Step 1: Archive Synchronization
```python
async def sync_archive_with_warehouse(archive, last_sync_time):
    """
    Fetch new trajectories from warehouse to update local archive.
    This includes trajectories from other clusterbomb instances.
    """
    async with AsyncWarehouseClient() as warehouse:
        # Get trajectories added since last sync
        new_trajectories = await warehouse.get_trajectory_dataset(
            since=last_sync_time,
            num_vars=config.num_vars,
            width=config.width
        )
    
    # Update local archive with trajectories from all sources
    for trajectory in new_trajectories:
        update_archive_from_trajectory(archive, trajectory, config)
    
    return datetime.now()

def update_archive_from_trajectory(archive, trajectory, config):
    """Process a trajectory and update the archive with its steps."""
    fisod = FormulaIsodegrees(config.num_vars, [])
    
    for i, step in enumerate(trajectory.steps):
        token_type, litint, cur_avgQ = step
        
        # Incrementally update formula
        if token_type == 0:  # ADD
            fisod.add_gate(litint)
        elif token_type == 1:  # DELETE
            fisod.remove_gate(litint)
        elif token_type == 2:  # EOS
            break
        
        # Update archive for this step
        cell_id = fisod.feature
        if cell_id not in archive:
            archive[cell_id] = []
        
        cell_elites = archive[cell_id]
        
        # Check if we should add this elite
        if len(cell_elites) < config.cell_density:
            cell_elites.append(Elite(
                traj_id=trajectory.traj_id,
                traj_slice=i,
                avgQ=cur_avgQ
            ))
            cell_elites.sort(key=lambda x: x.avgQ, reverse=True)
        elif cur_avgQ > min(e.avgQ for e in cell_elites):
            # Replace worst elite
            min_idx = min(range(len(cell_elites)), 
                         key=lambda j: cell_elites[j].avgQ)
            cell_elites[min_idx] = Elite(
                traj_id=trajectory.traj_id,
                traj_slice=i,
                avgQ=cur_avgQ
            )
            cell_elites.sort(key=lambda x: x.avgQ, reverse=True)
```

#### Step 2: Elite Selection
```python
def select_elites(archive, strategy="uniform", batch_size=10):
    # Get all populated cells (cells with at least one elite)
    populated_cells = [cell for cell in archive if len(archive[cell]) > 0]
    
    if strategy == "uniform":
        # Uniform random selection from populated cells
        selected_cells = random.sample(
            populated_cells, 
            min(batch_size, len(populated_cells))
        )
    elif strategy == "curiosity":
        # Prefer less-explored or boundary cells
        selected_cells = select_curious_cells(populated_cells, batch_size)
    elif strategy == "performance":
        # Bias toward high-performing cells
        selected_cells = select_high_performing_cells(populated_cells, batch_size)
    
    # For each selected cell, pick a random elite from that cell
    selected_elites = []
    for cell in selected_cells:
        elite = random.choice(archive[cell])  # Pick random elite from cell
        selected_elites.append((cell, elite))
    
    return selected_elites
```

#### Step 3: Mutation and Trajectory Collection
```python
import asyncio
from typing import List
from multiprocessing import Pool
from functools import partial

def run_mutation_job(args):
    """Worker function for parallel mutation execution."""
    elite, traj_slice, config = args
    # This runs in a separate process
    trajectories = run_mutations_sync(
        num_vars=config.num_vars,
        width=config.width,
        num_trajectories=config.num_trajectories,
        steps_per_trajectory=config.num_steps,
        prefix_traj=elite[:traj_slice+1],
        early_stop=True
    )
    return trajectories

async def post_trajectory_with_retry(warehouse, trajectory, max_retries=3):
    """Post a trajectory with retry logic."""
    for attempt in range(max_retries):
        try:
            return await warehouse.post_trajectory(trajectory)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to post trajectory after {max_retries} attempts: {e}")
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

async def mutate_elites(warehouse, selected_elites, trajectories_lookup, config):
    new_trajectories = []
    post_tasks = []
    
    # Prepare mutation jobs for parallel execution
    mutation_jobs = []
    for cell_id, elite in selected_elites:
        traj = trajectories_lookup[elite.traj_id]
        mutation_jobs.append((traj, elite.traj_slice, config))
    
    # Use multiprocessing Pool to run mutations in parallel across CPU cores
    num_workers = config.num_workers if config.num_workers else None  # None uses all cores
    with Pool(processes=num_workers) as pool:
        # Dispatch mutation jobs evenly across worker processes
        mutation_results = pool.map(run_mutation_job, mutation_jobs)
    
    # Flatten results and prepare for posting to warehouse
    for mutated_trajectories in mutation_results:
        # Create retry-enabled post tasks
        post_tasks.extend([
            post_trajectory_with_retry(warehouse, trajectory) 
            for trajectory in mutated_trajectories
        ])
        
        new_trajectories.extend(mutated_trajectories)
    
    # Execute all POST requests with error handling
    results = await asyncio.gather(*post_tasks, return_exceptions=True)
    
    # Check for failures and report
    failures = [r for r in results if isinstance(r, Exception)]
    if failures:
        print(f"Warning: {len(failures)} trajectories failed to post after retries")
        # Log failures for debugging but continue execution
        for failure in failures:
            print(f"  Error: {failure}")

    return new_trajectories
```

#### Step 4: Process Mutations
```python
for trajectory in new_trajectories:
    # Initialize FormulaIsodegrees for this trajectory
    fisod = FormulaIsodegrees(config.num_vars, [])
    
    for i, step in enumerate(trajectory.steps):
        token_type, litint, cur_avgQ = step
        
        # Incrementally update formula based on token type
        if token_type == 0:  # ADD operation
            fisod.add_gate(litint)
        elif token_type == 1:  # DELETE operation
            fisod.remove_gate(litint)
        elif token_type == 2:  # EOS (end of sequence)
            break
        
        # Get current formula feature
        cell_id = fisod.feature  # Immutable tuple identifier
        
        # Update archive if improvement or new cell
        if should_update_cell(archive, cell_id, cur_avgQ, config.cell_density):
            update_cell(archive, cell_id, trajectory.traj_id, i, cur_avgQ)
```

#### Step 5: Archive Update
```python
def update_cell(archive, cell_id, traj_id, traj_slice, avgQ):
    cell_elites = archive[cell_id]
    
    if len(cell_elites) < cell_density:
        # Add to cell if under capacity
        cell_elites.append(Elite(traj_id, traj_slice, avgQ))
        cell_elites.sort(key=lambda x: x.avgQ, reverse=True)
    elif avgQ > min(e.avgQ for e in cell_elites):
        # Replace worst elite if better
        cell_elites[-1] = Elite(traj_id, traj_slice, avgQ)
        cell_elites.sort(key=lambda x: x.avgQ, reverse=True)
```

### Phase 3: Results and Analysis

After completing all iterations:

1. **Archive Statistics**
   - Total cells discovered
   - Cell coverage (occupied vs possible cells)
   - Performance distribution across cells
   - Diversity metrics

2. **Final Data Collection**
   ```python
   async with AsyncWarehouseClient() as warehouse:
       # Retrieve all trajectories generated during the run
       final_dataset = await warehouse.get_trajectory_dataset(
           since=run_start_time,
           until=run_end_time
       )
   ```
   - Collect all trajectories generated during MAP-Elites execution
   - Generate visualization of the feature space
   - Export best formulas per complexity level

## Data Flow Integration

### Service Architecture

```
graph LR
    A[Clusterbomb Service with MAP-Elites] --> B[Warehouse Service]
    B --> A
```

The clusterbomb service now directly implements MAP-Elites algorithm with multi-instance support:

1. **Clusterbomb Service (MAP-Elites Engine)**
   - Runs as an autonomous container executing MAP-Elites algorithm
   - **Supports multiple concurrent instances** for distributed exploration
   - Directly generates and evaluates trajectory mutations
   - Maintains a local elite archive that synchronizes with global state
   - No external API endpoints - operates as a job executor

2. **Warehouse Service Integration**
   - **Initialization**: Retrieves existing trajectories via `AsyncWarehouseClient`
   - **Periodic Synchronization**: Fetches new trajectories from all clusterbomb instances
   - **Continuous Updates**: Pushes new trajectories asynchronously during execution
   - **Final Collection**: Gathers all generated trajectories at completion
   - All communication uses async patterns for optimal performance

3. **Multi-Instance Coordination**
   ```python
   # At startup - load global state filtered by configuration
   trajectories = await warehouse.get_trajectory_dataset(
       num_vars=config.num_vars,
       width=config.width
   )
   
   # Each iteration - sync with other instances
   last_sync = await sync_archive_with_warehouse(archive, last_sync_time)
   
   # During execution - share discoveries immediately
   # NOTE: Each trajectory is sent as soon as it's collected,
   # not batched until iteration end
   for trajectory in new_trajectories:
       await warehouse.post_trajectory(trajectory)
   
   # At completion - gather collective results
   final_dataset = await warehouse.get_trajectory_dataset(
       since=start_time,
       num_vars=config.num_vars,
       width=config.width
   )
   ```
   
   Multiple clusterbomb instances can run simultaneously:
   - Each maintains its own local archive
   - Archives are periodically synchronized through the warehouse
   - Discoveries from one instance benefit all others
   - Enables massive parallel exploration of the solution space

## Implementation Considerations

### Performance Optimizations

1. **Multiprocessing for Mutations**
   - Utilizes `multiprocessing.Pool` to parallelize mutation jobs across CPU cores
   - Configurable worker count via `num_workers` parameter (None uses all available cores)
   - Each mutation runs in a separate process for true parallelism
   - Pool.map automatically distributes jobs evenly across workers

2. **Batch Processing**
   - Send multiple elites for mutation in parallel
   - Process trajectory responses in batches
   - Update archive in bulk operations

3. **Caching**
   - Cache FormulaIsodegrees computations
   - Maintain formula hash → feature mapping
   - Store frequently accessed elites in memory

4. **Incremental Updates**
   - Process trajectories step-by-step
   - Update features incrementally along trajectory
   - Avoid redundant feature recalculation

### Cell Density Management

The `cell_density` parameter controls diversity within cells:
- `cell_density = 1`: Classic MAP-Elites (one elite per cell)
- `cell_density > 1`: Multiple elites per cell, maintaining top-k performers
- Benefits: Preserves variation within structurally similar formulas
- Trade-off: Memory usage vs diversity preservation

### Mutation Strategies

Different mutation approaches via clusterbomb:
1. **Random Walk**: Uniform random exploration from elite
2. **Guided Search**: Bias toward promising directions
3. **Local Optimization**: Fine-tune around elite
4. **Structural Variation**: Focus on formula structure changes

### Convergence Criteria

The algorithm terminates when:
1. Fixed iteration count reached (`num_iterations`)
2. Archive coverage plateaus (no new cells discovered)
3. Performance improvement stagnates
4. User-defined stopping condition met

## Algorithm Advantages for Boolean Formula Optimization

1. **Structural Diversity**: FormulaIsodegrees ensures diverse formula structures
2. **Isomorphism Handling**: Equivalent formulas map to same cell, avoiding redundancy
3. **Stepping Stones**: Maintains suboptimal solutions that may lead to breakthroughs
4. **Parallelizable**: Elite selection and mutation can be parallelized
5. **Incremental Learning**: Builds on existing trajectory data from warehouse

## Future Enhancements

1. **Adaptive Parameters**
   - Dynamic cell_density based on archive coverage
   - Adaptive mutation rates based on improvement trends
   - Variable num_steps based on formula complexity

2. **Advanced Feature Spaces**
   - Multi-dimensional features beyond FormulaIsodegrees
   - Learned feature representations
   - Dynamic feature space expansion

3. **Hybrid Approaches**
   - Combine with local search for elite refinement
   - Integrate with SAT solvers for feasibility checking
   - Use neural networks for mutation guidance

## References

- Mouret, J.B. and Clune, J., 2015. Illuminating search spaces by mapping elites. arXiv preprint arXiv:1504.04909
- Vassiliades, V., Chatzilygeroudis, K. and Mouret, J.B., 2017. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation
- Cully, A. and Demiris, Y., 2017. Quality and diversity optimization: A unifying modular framework. IEEE Transactions on Evolutionary Computation