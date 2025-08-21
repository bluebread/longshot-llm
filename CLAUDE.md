# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gym-longshot is a C++/Python reinforcement learning framework for boolean function optimization. It combines formula generation, trajectory processing, and microservice architecture to explore propositional logic formulas within RL environments.

## Build and Test Commands

### Library Build (C++ Extensions)
```bash
# Build the Python library with C++ extensions
cd library
pip install -e .
```

### Testing
```bash
# Run Python tests
cd test
pytest .

# Run C++ tests and Python tests
make test

# Run specific test file
pytest test_trajectory_processor.py
```

### Services
```bash
# Start infrastructure services (Neo4j, Redis, MongoDB, RabbitMQ)
cd service
docker-compose up -d

# Run warehouse service
cd service/warehouse
pip install -r requirements.txt
python main.py  # Runs on localhost:8000

# Run clusterbomb service  
cd service/clusterbomb
pip install -r requirements.txt
python main.py  # Runs on localhost:8060
```

### Demo System
```bash
# Run V2 system demonstration
python script/demo_v2_system.py
```

## Architecture

### Core Components

1. **Library (`library/longshot/`)**
   - **agent/**: RL agents and client wrappers for microservices
     - `ClusterbombAgent`: Client for weapon rollout operations
     - `WarehouseAgent`: Client for data storage/retrieval  
     - `TrajectoryProcessor`: Processes trajectories and computes metrics
   - **circuit/**: Logic circuit representations and operations
   - **env/**: RL environment for formula optimization
   - **models/**: Pydantic models for API data structures
   - **utils/**: Utility functions and formula operations
   - **core/**: C++ implementation for performance-critical operations

2. **Microservices (`service/`)**
   - **warehouse**: Data storage service (FastAPI)
     - Manages Neo4j (evolution graph), MongoDB (trajectories), Redis (isomorphism cache)
     - Provides unified API for data operations
   - **clusterbomb**: Trajectory generation service (FastAPI)  
     - Generates trajectories using random exploration
     - Processes results using TrajectoryProcessor locally

3. **Testing (`test/`)**
   - Comprehensive test suite with pytest
   - C++ unit tests with Makefile build system
   - Tests for all major components and integrations

### Data Flow

1. **Trajectory Generation**: Clusterbomb service generates trajectories from boolean formulas
2. **Processing**: TrajectoryProcessor computes avgQ values and formula metrics locally
3. **Storage**: Warehouse stores processed data in Neo4j (graph), MongoDB (trajectories), Redis (hashes)
4. **Retrieval**: Evolution graph analysis and visualization via warehouse API

### V2 System Refactor

The V2 architecture simplifies the original microservice design:
- Removes separate trajproc and ranker services
- Integrates processing directly into weapon services (clusterbomb)
- Consolidates formula and graph data in single Neo4j nodes
- Uses unified trajectory storage in MongoDB with cur_avgQ field

### Database Schema

- **Neo4j**: Evolution graph with FormulaNode labels containing integrated formula data
- **MongoDB**: Trajectories with simplified schema including cur_avgQ per step
- **Redis**: WL hash → formula ID mappings for isomorphism detection

### Key Files for Development

- `library/longshot/agent/trajectory_processor.py`: Core trajectory processing logic
- `service/warehouse/main.py`: Main data service API
- `service/clusterbomb/main.py`: Trajectory generation service
- `script/demo_v2_system.py`: Complete system demonstration
- `doc/microservice-v2.md`: Detailed V2 API specification

## Development Notes

- The system uses both Python and C++ components - ensure C++ compilation works before running tests
- Microservices require infrastructure services (Docker Compose) to be running
- V2 refactor is current architecture - older V1 components may be found in archive/
- Tests expect specific database configurations - check service/docker-compose.yml for credentials