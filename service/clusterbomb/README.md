# Clusterbomb Service

MAP-Elites algorithm implementation for boolean formula optimization through quality-diversity search.

## Overview

The Clusterbomb service implements the MAP-Elites algorithm to discover diverse, high-performing boolean formulas. It maintains an archive of elite solutions distributed across a behavioral feature space, enabling both exploration and exploitation of the formula search space.

## Features

- **Quality-Diversity Optimization**: Maintains diverse archives of high-performing formulas
- **Behavioral Feature Space**: Uses formula isodegrees for diversity preservation
- **Parallel Processing**: Multiprocessing support for efficient mutation generation
- **Distributed Execution**: Optional synchronization with warehouse service for multi-instance runs
- **Flexible Strategies**: Multiple elite selection and initialization strategies
- **Batch Experiments**: Automated experiment runner for systematic parameter exploration

## Installation

1. Install the longshot library:
```bash
cd ../../library
pip install -e .
cd ../service/clusterbomb
```

2. Install service dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the warehouse service is running (see main README for setup).

## Quick Start

### Running Single MAP-Elites Execution

Basic execution with default settings:
```bash
python run_map_elites.py
```

Custom configuration example:
```bash
python run_map_elites.py \
    --iterations 500 \
    --num-vars 5 \
    --width 4 \
    --batch-size 20 \
    --strategy performance
```

### Running Batch Experiments

The `run_map_elites_experiments.sh` script automates running experiments across multiple parameter configurations:

```bash
# Make script executable
chmod +x run_map_elites_experiments.sh

# Run experiments (currently configured for n=8, w=7-8)
./run_map_elites_experiments.sh
```

The script runs experiments with:
- Two strategies: uniform (random init) and performance (warehouse init)
- Calculated iterations: 3 × width²
- Calculated steps: 2 × 2^width
- Fixed cell density: 3
- Fixed batch size: 32

## Usage Guide

### run_map_elites.py - Main Execution Script

```bash
python run_map_elites.py [OPTIONS]
```

#### Core Algorithm Parameters
- `--iterations N`: Number of MAP-Elites iterations (default: 100)
- `--cell-density N`: Maximum elites per cell (default: 1)

#### Formula Space Parameters
- `--num-vars N`: Number of boolean variables (default: 4)
- `--width N`: Maximum formula width (default: 3)

#### Mutation Parameters
- `--num-steps N`: Steps per trajectory mutation (default: 10)
- `--num-trajectories N`: Trajectories per mutation (default: 5)

#### Parallelization
- `--num-workers N`: Number of parallel workers (default: all CPU cores)
- `--batch-size N`: Number of elites to mutate per iteration (default: 10)

#### Algorithm Strategy
- `--strategy TYPE`: Elite selection strategy: uniform, curiosity, performance (default: uniform)
- `--init-strategy TYPE`: Initialization: warehouse, random (default: warehouse)

#### Synchronization (for distributed runs)
- `--enable-sync`: Enable synchronization with other instances via warehouse
- `--sync-interval N`: Iterations between syncs when enabled (default: 10)

#### Warehouse Configuration
- `--warehouse-host HOST`: Warehouse service host (default: localhost)
- `--warehouse-port PORT`: Warehouse service port (default: 8000)
- `--timeout SECONDS`: HTTP timeout for warehouse requests (default: 30.0)

#### Output Options
- `--output PATH`: Archive output file (default: output/archive-n{N}w{W}.json)
- `--quiet`: Reduce output verbosity
- `--no-save`: Don't save archive to file

### Examples

#### Basic Exploration
```bash
# Explore small formula space with default settings
python run_map_elites.py --num-vars 3 --width 2 --iterations 100
```

#### High-Performance Search
```bash
# Intensive search with performance-focused strategy
python run_map_elites.py \
    --num-vars 5 \
    --width 4 \
    --iterations 1000 \
    --batch-size 50 \
    --num-workers 16 \
    --strategy performance \
    --cell-density 5
```

#### Distributed Execution
```bash
# Run multiple instances with synchronization
# Instance 1
python run_map_elites.py --enable-sync --sync-interval 10 &

# Instance 2 (on same or different machine)
python run_map_elites.py --enable-sync --sync-interval 10 \
    --warehouse-host warehouse.example.com
```

#### Long-Running Exploration
```bash
# Extended exploration with periodic sync and large archive
python run_map_elites.py \
    --num-vars 6 \
    --width 5 \
    --iterations 5000 \
    --cell-density 10 \
    --enable-sync \
    --sync-interval 50 \
    --timeout 60.0 \
    --output archives/exploration-n6w5.json
```

## Algorithm Details

### MAP-Elites Overview

MAP-Elites is a quality-diversity algorithm that:
1. Maintains an archive divided into behavioral niches
2. Each niche stores elite solutions with unique behavioral characteristics
3. New solutions are created by mutating existing elites
4. Solutions compete only within their behavioral niche

### Behavioral Features

The service uses **formula isodegrees** as behavioral features, which capture structural properties of boolean formulas while being invariant to variable permutations.

### Selection Strategies

- **uniform**: Randomly selects elites from the archive
- **curiosity**: Prioritizes less-explored regions of the feature space
- **performance**: Biases selection toward higher-performing elites

### Initialization Strategies

- **random**: Starts with randomly generated formulas
- **warehouse**: Seeds archive with existing high-quality trajectories from warehouse

## Output Format

Archives are saved as JSON files containing:
- Elite trajectories and their scores
- Behavioral feature mappings
- Metadata (iterations, parameters, timestamp)

Example structure:
```json
{
  "metadata": {
    "num_vars": 4,
    "width": 3,
    "iterations": 100,
    "timestamp": "2024-01-15T10:30:00"
  },
  "archive": {
    "cell_0": [
      {
        "trajectory": [5, 12, 33, ...],
        "score": 0.85,
        "features": [0.2, 0.5, 0.3]
      }
    ]
  }
}
```

## Monitoring

Progress is logged to stdout with information about:
- Current iteration
- Archive size and coverage
- Best score found
- Synchronization events (if enabled)

Set `--quiet` to reduce verbosity or check logs for detailed information.

## Troubleshooting

### Common Issues

1. **Warehouse Connection Failed**
   - Ensure warehouse service is running
   - Check host and port configuration
   - Verify network connectivity

2. **Out of Memory**
   - Reduce `--batch-size`
   - Decrease `--cell-density`
   - Lower `--num-workers`

3. **Slow Performance**
   - Increase `--num-workers` for more parallelization
   - Adjust `--batch-size` for better CPU utilization
   - Consider using `--strategy performance` for faster convergence

## Development

### Running Tests
```bash
pytest test/
```

### Extending the Service

1. **New Selection Strategies**: Add to `MAPElitesService.select_elites()`
2. **Custom Features**: Modify `FormulaIsodegrees` class
3. **Alternative Mutations**: Extend `TrajectoryGenerator`

## Architecture

```
clusterbomb/
├── run_map_elites.py          # Main execution script
├── run_map_elites_experiments.sh  # Batch experiment runner
├── map_elites_service.py      # Core MAP-Elites implementation
├── trajectory_generator.py    # Mutation generation
├── isodegrees.py              # Behavioral feature extraction
├── models.py                  # Data models and configuration
└── test/                      # Test suite
```

## Related Services

- **Warehouse Service**: Stores and retrieves trajectories
- **Trainer Service**: Trains neural models on collected trajectories

See the main project README for integration details.