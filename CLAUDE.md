# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Longshot LLM is a C++/Python library for boolean function manipulation and optimization, using MAP-Elites algorithms and trajectory-based learning. The project combines low-level C++ performance with Python accessibility through pybind11 bindings.

## Core Commands

### Building the Library
```bash
# Install the longshot library in development mode
cd library && pip install -e . && cd ..
```

### Running Tests
```bash
# Run Python tests with pytest
pytest test/

# Run specific test file
pytest test/service/test_warehouse.py

# Run C++ tests
cd test/library && make test
```

### Running Services
```bash
# Start warehouse service (trajectory storage)
cd service/warehouse && uvicorn main:app --host 0.0.0.0 --port 8000

# Start clusterbomb service (formula evaluation)
cd service/clusterbomb && uvicorn main:app --host 0.0.0.0 --port 8060

# Or use docker-compose
cd service && docker-compose up
```

### MAP-Elites Experiments
```bash
# Run MAP-Elites experiments with dataset export
cd archive/v2/script
python map_elites_with_dataset.py --num-vars 4 --width 3 --iterations 100

# Run batch experiments
./run_map_elites_experiments.sh

# Download trajectory dataset
python download_trajectory_dataset.py --output trajectories.json

# Export trajectories from MongoDB
MONGO_USER=user MONGO_PASSWORD=pass python export_trajectories.py --output-file data.json --all
```

## Architecture

### Library Structure (`library/longshot/`)
- **`_core`**: C++ extension module (built from `core/core.cpp`) providing high-performance boolean operations
- **`formula/`**: Boolean formula manipulation - `FormulaIsodegrees`, decision trees, reward models
- **`literals/`**: Term and clause representations for boolean formulas
- **`service/`**: Client interfaces for warehouse and clusterbomb services
- **`utils/`**: Utilities for trajectory processing, base64 encoding, formula parsing

### Services (`service/`)
- **Warehouse Service**: FastAPI service for trajectory storage using MongoDB/Redis
  - Stores and retrieves formula evaluation trajectories
  - Provides dataset export endpoints
- **Clusterbomb Service**: FastAPI service for formula evaluation
  - Evaluates boolean formulas with various strategies
  - Computes isodegrees and other formula metrics

### Archive (`archive/v2/`)
- Contains MAP-Elites implementation and experimental scripts
- **`script/`**: Core experiment runners and utilities
  - `map_elites.py`: Main MAP-Elites algorithm
  - `map_elites_with_dataset.py`: MAP-Elites with trajectory dataset export
  - `export_trajectories.py`: MongoDB to JSON exporter
  - `show_formula.py`: Formula visualization utility

## Key Concepts

### Trajectories
Trajectories represent sequences of formula evaluations with three components:
- `type`: Token types (integers)
- `litint`: Token literals (integers)  
- `avgQ`: Average Q values (floats)

### Formula Representation
Formulas use gate integer representations that can be parsed into tree structures. The system supports:
- Variable counting and isodegree computation
- Decision tree conversion
- Graph-based analysis

### MAP-Elites Algorithm
The implementation uses multi-dimensional archives to optimize boolean formulas:
- Cells indexed by formula characteristics (num_vars, width, size)
- Elite selection strategies: uniform, curiosity, performance
- Configurable mutation and batch processing

## Environment Variables

For MongoDB operations:
- `MONGO_USER`: MongoDB username
- `MONGO_PASSWORD`: MongoDB password  
- `MONGO_HOST`: MongoDB host (default: localhost)
- `MONGO_PORT`: MongoDB port (default: 27017)
- `MONGO_DB`: Database name (default: LongshotWarehouse)
- `LONGSHOT_TEST_MODE`: Set to 1 for test mode with default credentials

## Development Notes

- The library uses pybind11 for C++/Python bindings with C++17 standard
- OpenMP is enabled for parallel processing in C++ code
- Services use FastAPI with Pydantic for API validation
- Test coverage includes both Python (pytest) and C++ unit tests
- The project supports batch processing for large-scale experiments