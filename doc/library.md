# Library Documentation

## Overview

1. Agent Module:
    1. `class TrajectoryQueueAgent`:
        - Manages the trajectory queue using RabbitMQ.
        - Provides methods to push and pop trajectories.
        - Main functions:
            1. `push()`: Pushes a trajectory to the RabbitMQ queue.
            2. `pop()`: Pops a trajectory from the RabbitMQ queue.
            3. `start_consuming()`: Starts consuming messages from the RabbitMQ queue.
            4. `close()`: Closes the connection to RabbitMQ.
        - This class is a local module, which is accessible for any other components in the RL system.
    2. `class WarehouseAgent`:
        - Manages the warehouse microservice.
        - Provides methods to interact with the warehouse API.
        - Main functions:
            1. `get_X()`: Retrieves data from the warehouse.
            2. `post_X()`: Posts data to the warehouse.
            3. `put_X()`: Updates data in the warehouse.
            4. `delete_X()`: Deletes data from the warehouse.
        - This class is a local module, which is accessible for any other components in the RL system.
    3. `class ClusterbombAgent`:
        - Client for the Clusterbomb microservice.
        - Provides methods for weapon rollout operations.
        - Main functions:
            1. `weapon_rollout()`: Triggers trajectory generation with specified parameters.
            2. `health_check()`: Checks the health status of the Clusterbomb service.
        - Supports both synchronous and asynchronous operations.
    4. `class TrajectoryProcessor`:
        - Processes trajectories and updates the warehouse databases.
        - Operates locally within services that generate trajectories.
        - Main functions:
            1. `process_trajectory()`: Processes trajectory with V2 context (prefix + suffix).
            2. `reconstruct_base_formula()`: Rebuilds formula from prefix trajectory.
            3. `check_base_formula_exists()`: Verifies if formula already exists.
            4. `isomorphic_to()`: Checks for isomorphic formulas using WL hashes. 

2. Environment Module: **Formula Game**
    - Implements the RL environment that simulates the process of adding/deleting gates in a normal formed formula. 
    - Calculates *average-case deterministic query complexity*, the optimization target.
    - Main functions:
        1. `reset()`: Resets internal variables. 
        2. `step()`: Given the passed token (which indicates adding or deleting a gate), simulates a step and returns the reward. 
        3. `class GateToken`: 
            - Represents a token indicating an operation in the formula game.
        4. base64 encoding/decoding functions:
            1. `encode_float64_to_base64(value: float) -> str`: Encodes a float64 value to a base64 string.
            2. `decode_base64_to_float64(value: str) -> float`: Decodes a base64 string to a float64 value.
            3. `class Float64Base64`: A Pydantic model that validates base64-encoded float64 strings.

3. [Removed] ~~Ranking Module: **ArmRanker**~~
    - This module has been removed from the project as it is no longer needed.

4. Circuit Module: **FormulaGraph**
    - Implements graph representation of boolean formulas.
    - Provides methods for formula manipulation and analysis.
    - Main functions:
        1. `FormulaGraph(gates)`: Constructor accepting gate definitions.
        2. `wl_hash()`: Computes Weisfeiler-Lehman hash for isomorphism detection.
        3. `add_gate()`: Adds a new gate to the formula.
        4. `delete_gate()`: Removes a gate from the formula.
        5. `cur_avgQ`: Property returning average-case query complexity.

5. Models Module:
    - **API Models**: Pydantic models for API data structures.
        - `TrajectoryInfo`: Trajectory data with steps as tuples `(token_type, token_literals, cur_avgQ)`.
        - `TrajectoryProcessingContext`: V2 context for processing (prefix + suffix trajectories).
        - `WeaponRolloutRequest/Response`: Models for weapon rollout operations.
    - **Data Format**:
        - Trajectory steps stored as tuples for efficiency: `list[tuple[int, int, float]]`
        - MongoDB storage uses lists for BSON optimization.
        - Backward compatibility maintained for legacy dict format.