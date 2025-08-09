# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

This is a complex reinforcement learning system for boolean formula generation and optimization, consisting of:

1. **Library** (`library/`): Core Python package with C++ extensions
   - `longshot/agent/`: RL agents and environment management
   - `longshot/circuit/`: Boolean formula and circuit representations
   - `longshot/models/`: Neural network architectures and data models
   - `longshot/utils/`: Utility functions and base64 encoding
   - `longshot/core/`: C++ backend for performance-critical operations

2. **Services** (`service/`): Microservices architecture
   - `warehouse/`: FastAPI service for formula storage (MongoDB, Neo4j, Redis)
   - `trajproc/`: Trajectory processing service
   - `ranker/`: Formula ranking service
   - `clusterbomb/`: Additional processing service

3. **Archive** (`archive/`): Research code and Jupyter notebooks for PPO training
4. **Test** (`test/`): Unit tests and C++ test files
5. **Doc** (`doc/`): Sphinx documentation

## Key Architecture

- **Core C++ Engine**: Performance-critical boolean operations in `library/longshot/core/`
- **RL Environment**: `EnvironmentAgent` manages multiple formula games, transforms data to tensors
- **Formula Representation**: `Literals` class represents boolean literals, `GateToken` for operations
- **Microservices**: Docker-compose infrastructure with Neo4j, MongoDB, Redis, RabbitMQ
- **Hybrid Language**: Python frontend with C++ backend via pybind11

## Development Commands

### Library Build & Development
```bash
# Build the C++ extension library (from library/ directory)
cd library && python setup.py build_ext --inplace

# Install in development mode
cd library && pip install -e .
```

### Testing
```bash
# Run all tests (from test/ directory)
cd test && make test

# Run Python tests only
cd test && pytest .

# Run C++ tests
cd test && make $(find . -name '*.cpp' | sed 's/.cpp/.out/g')
```

### Services
```bash
# Start infrastructure services
cd service && docker-compose up -d

# Individual service testing
cd service/trajproc && pytest test/
```

### Documentation
```bash
# Build documentation (from doc/ directory)
cd doc && make html
```

## Important Implementation Details

- **Token Dimension**: `GateToken.dim_token(num_vars) = 2 * num_vars + 3`
- **Environment Management**: `EnvironmentAgent` handles multiple formula games, uses arm filtering
- **C++ Integration**: Core boolean operations compiled with `-Ofast -fopenmp` optimizations
- **Database Schema**: Formulas stored in MongoDB, graph relationships in Neo4j, caching in Redis
- **RL Rewards**: Based on average-case deterministic query complexity of resulting formulas

## Configuration Files

- `library/pyproject.toml`: Python package configuration
- `library/setup.py`: C++ extension build configuration with pybind11
- `service/docker-compose.yml`: Infrastructure services setup
- `test/pytest.ini`: Test configuration with deprecation warning filters