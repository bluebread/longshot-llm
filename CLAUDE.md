# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
- Run all tests in a specific test file: `python -m pytest test/test_map_elites.py -v`
- Run a specific test class: `python -m pytest test/test_map_elites.py::TestFormulaIsodegrees -v`
- Run a single test method: `python -m pytest test/test_map_elites.py::TestTrajectoryGenerator::test_run_mutations_sync -v`
- Run tests for data cleaning: `python -m pytest test/service/test_data_cleaning.py -v`
- Run tests for a specific function: `python -m pytest test/service/test_data_cleaning.py::TestDataCleaningFunctions::test_truncate_trajectory_to_max_avgq -v`

### Training
- Train a model: `cd service/trainer && python train.py`
- Evaluate a model: `cd service/trainer && python eval.py --model-path ./models/MODEL_NAME`

### MAP-Elites Algorithm
- Run MAP-Elites with default settings: `cd service/clusterbomb && python run_map_elites.py`
- Run with custom parameters: `python run_map_elites.py --iterations 500 --num-vars 5 --width 4`
- Run with synchronization for distributed execution: `python run_map_elites.py --enable-sync --sync-interval 10`

### Utility Scripts
- Export trajectories: `python script/export_trajectories.py`
- Show formula details: `python script/show_formula.py`
- Analyze avgQ distribution: `python script/avgq_distribution.py`
- Validate trajectories: `python script/validate_trajectories.py`

## Architecture

### Core Library (library/longshot/)
The `longshot` library is a C++/Python hybrid package providing core functionality for boolean formula manipulation:
- **_core.cpython**: C++ extension module for high-performance operations
- **formula/**: Boolean formula representations and operations
- **literals/**: Literal and gate handling
- **utils/**: Utility functions including gate parsing and encoding
- **service/**: Service layer for warehouse client and trajectory management
- **error/**: Custom exceptions

### Services Architecture

#### MAP-Elites Service (service/clusterbomb/)
Implements the MAP-Elites evolutionary algorithm for formula optimization:
- **map_elites_service.py**: Core MAP-Elites algorithm implementation with archive management
- **trajectory_generator.py**: Generates trajectory mutations for formula evolution
- **isodegrees.py**: Feature extraction using formula isodegrees for behavioral characterization
- **models.py**: Pydantic models for configuration and data structures
- **run_map_elites.py**: Standalone runner script for autonomous execution

Key concepts:
- Uses behavioral feature space based on formula isodegrees
- Maintains an archive of diverse, high-performing solutions (elites)
- Supports distributed execution with warehouse synchronization
- Multiprocessing support for parallel mutation generation

#### Training Service (service/trainer/)
Handles neural network training for trajectory generation:
- **model.py**: GPT2ForLongshot model implementation with custom embeddings
- **dataset.py**: TrajectoryDataset for loading trajectory data
- **collator.py**: TrajectoryCollator for batch processing with permutation augmentation
- **custom_trainer.py**: LongshotTrainer extending HuggingFace Trainer
- **eval.py**: Model evaluation with avgQ analysis and distribution metrics
- **data_cleaning.py**: Functions for trajectory preprocessing and truncation
- **reward_function.py**: LongshotRewardFunction for TRL integration

Training pipeline:
1. Load trajectories from JSON datasets (format: n{num_vars}w{width}.json)
2. Apply data augmentation through input permutation
3. Train GPT-2 based model to predict next gate in trajectory
4. Evaluate using avgQ metrics and trajectory quality measures

#### Warehouse Service (service/warehouse/)
Centralized storage for trajectories (implementation details in main.py)

### Key Data Structures

**Trajectory**: Sequence of gate operations represented as integers
- Each gate encoded as integer combining operation type and variable indices
- Trajectories evolve formulas step-by-step

**Elite**: High-performing trajectory with behavioral features
- Contains trajectory, performance score (avgQ), and feature vector
- Stored in MAP-Elites archive cells based on behavioral characteristics

**FormulaIsodegrees**: Behavioral feature extraction
- Computes isomorphism-invariant features from formula structure
- Used for diversity preservation in MAP-Elites

## Important Patterns

### Gate Encoding
Gates are encoded as integers using `parse_gate_integer_representation` from longshot.utils. The encoding combines operation type and variable indices.

### Async Operations
Services use AsyncWarehouseClient for non-blocking communication with the warehouse. Always use async/await patterns when interacting with warehouse.

### Multiprocessing
MAP-Elites uses multiprocessing.Pool for parallel mutation generation. The `run_mutations_sync` function handles worker pool management.

### Model Serialization
Models are saved in HuggingFace-compatible format using `save_pretrained()` and loaded with `from_pretrained()`. Model directories include timestamp in name: `n{n}w{w}-{datetime}`.