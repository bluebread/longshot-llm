# MAP-Elites Algorithm

## Overview

This document outlines the implementation plan for adapting the **MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) algorithm** to optimize boolean formulas in the gym-longshot framework. MAP-Elites is a quality-diversity algorithm that maintains a collection of diverse, high-performing solutions organized in a multi-dimensional feature space.

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

In our implementation, each boolean formula is characterized by its `FormulaFeature`:
- **Primary Mapping**: FormulaFeature → Cell (isomorphism-invariant representation)
- **Cell Identity**: The sorted tuple of literal occurrence counts serves as the cell identifier
- **Uniqueness**: Isomorphic formulas map to the same cell, ensuring structural diversity

### Algorithm Parameters

```python
class MAPElitesConfig:
    # Core algorithm parameters
    num_iterations: int          # Total number of iterations to run
    cell_density: int            # Maximum organisms per cell (elites)
    
    # Formula space parameters
    max_num_vars: int                # Number of boolean variables
    max_width: int                   # Formula width constraint
    max_size: int                    # Formula size constraint
    
    # Mutation parameters
    mutate_length: int           # Steps to run in clusterbomb simulation
    num_mutate: int              # Number of trajectories to collect per mutation
    
    # Optional parameters
    batch_size: int = 10         # Number of parallel mutations
    elite_selection_strategy: str = "uniform"  # How to select elites
    initialization_strategy: str = "warehouse"  # Source of initial population
```

## Algorithm Process Flow

### Phase 1: Initialization

1. **Download Existing Data**
   ```
   warehouse.download_trajectory_dataset() → List[Trajectory]
   ```
   - Retrieve all existing trajectories from the warehouse service
   - Extract formulas and their performance metrics (avgQ values)

2. **Build Initial Archive**
   ```python
    ff = FormulaFeature(num_vars, [])
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
            if used_variables.bit_length() > num_vars:
                break
            if token_type == 0 and lits.width > width:
                break

            if token_type == 0:
                ff.add_gate(litint)
            elif token_type == 1:
                ff.remove_gate(litint)
            else:
                raise Exception()

            cell_id = ff.feature  # Immutable tuple identifier
           
            if cell_id not in archive or cur_avgQ > archive[cell_id].avgQ:
                archive[cell_id] = Elite(
                    traj_id=tid, 
                    traj_slice=i,
                    avgQ=cur_avgQ
                )
   ```
   - Process trajectories incrementally to build FormulaFeatures
   - Map formulas to cells based on their feature representation
   - Keep only the best-performing formula per cell (elite selection)

### Phase 2: Main Evolution Loop

Repeat for `num_iterations`:

#### Step 1: Elite Selection
```python
def select_elites(archive, strategy="uniform"):
    populated_cells = [cell for cell in archive if archive[cell] is not None]
    
    if strategy == "uniform":
        # Uniform random selection from populated cells
        return random.sample(populated_cells, min(batch_size, len(populated_cells)))
    elif strategy == "curiosity":
        # Prefer less-explored or boundary cells
        return select_curious_cells(populated_cells)
    elif strategy == "performance":
        # Bias toward high-performing cells
        return select_high_performing_cells(populated_cells)
```

#### Step 2: Mutation via ClusterbombAgent
```python
for selected_cell in selected_cells:
    elite = archive[selected_cell]
    traj = trajectories_lookup[elite.traj_id]
    
    # Send elite formula to clusterbomb for mutation
    request = RolloutRequest(
        num_vars=num_vars,
        width=width,
        size=size,
        num_trajectories=num_mutate,
        steps_per_trajectory=mutate_length,
        prefix_traj=traj[:elite.traj_slice+1],
        early_stop=True
    )
    
    response = clusterbomb_agent.weapon_rollout(request)
    new_trajectories = response.trajectories
```

#### Step 3: Process Mutations
```python
for trajectory in new_trajectories:
    for step in trajectory:
        # Extract mutated formula and its performance
        mutant_formula = step.formula
        mutant_avgQ = step.cur_avgQ
        
        # Calculate feature for placement in archive
        feature = FormulaFeature(num_vars, mutant_formula.gates)
        cell_id = feature.feature
        
        # Update archive if improvement or new cell
        if should_update_cell(archive, cell_id, mutant_avgQ, cell_density):
            update_cell(archive, cell_id, mutant_formula, mutant_avgQ)
```

#### Step 4: Archive Update
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

2. **Elite Export**
   - Save all elites to warehouse
   - Generate visualization of the feature space
   - Export best formulas per complexity level

## Data Flow Integration

### Service Interactions

```
graph LR
    A[MAP-Elites Controller] --> B[Warehouse Service]
    A --> C[ClusterbombAgent]
    C --> D[Clusterbomb Service]
    D --> E[TrajectoryProcessor]
    E --> C
    C --> A
    B --> A
    A --> B
```

1. **Warehouse Service**
   - Initial data retrieval: `GET /trajectories`
   - Elite storage: `POST /formulas` with avgQ metadata
   - Graph updates: `PUT /evolution-graph` with parent-child relationships

2. **ClusterbombAgent**
   - Mutation requests: Send elite formulas for exploration
   - Trajectory retrieval: Receive mutated formulas with performance metrics

3. **TrajectoryProcessor** (within Clusterbomb)
   - Computes avgQ values for trajectory steps
   - Ensures consistent performance evaluation

## Implementation Considerations

### Performance Optimizations

1. **Batch Processing**
   - Send multiple elites to clusterbomb in parallel
   - Process trajectory responses in batches
   - Update archive in bulk operations

2. **Caching**
   - Cache FormulaFeature computations
   - Maintain formula hash → feature mapping
   - Store frequently accessed elites in memory

3. **Incremental Updates**
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

1. **Structural Diversity**: FormulaFeature ensures diverse formula structures
2. **Isomorphism Handling**: Equivalent formulas map to same cell, avoiding redundancy
3. **Stepping Stones**: Maintains suboptimal solutions that may lead to breakthroughs
4. **Parallelizable**: Elite selection and mutation can be parallelized
5. **Incremental Learning**: Builds on existing trajectory data from warehouse

## Future Enhancements

1. **Adaptive Parameters**
   - Dynamic cell_density based on archive coverage
   - Adaptive mutation rates based on improvement trends
   - Variable mutate_length based on formula complexity

2. **Advanced Feature Spaces**
   - Multi-dimensional features beyond FormulaFeature
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