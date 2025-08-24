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
    3. **Environment Agent**
        - Manages environments. 
        - Transforms data in Tensor/TensorDict format. 
        - Main functions:
            1. `replace_arms()`: Replaces all arms/environments using the arm filter. 
            2. `reset()`: Resets formula games and saves trajectories to the trajectory queue (except the first time calling `reset()`). 
            3. `step()`: Executes a step of formula games. 

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

4. Processing Module: **TrajectoryProcessor**
    - Local class that processes trajectories and updates the databases.
    - Responsible for managing the data flow between trajectory collection and the warehouse.
    - Stateless and operates as a local class.
    - Main functions:
        1. `longshot.processing.TrajectoryProcessor(**config)`:
            - Constructor that takes configuration parameters for the processor
        2. `TrajectoryProcessor.process_trajectory(self, trajectory: dict) -> dict[str, Any]`:
            - Processes a single trajectory and updates the evolution graph accordingly.
            - Called when a new trajectory is collected and tries to break down the trajectory into smaller parts if necessary.
            - Current implementation processes the trajectory by extracting the formulas with top `traj_num_summits` avgQ values and breaking the trajectory into parts smaller than the number `granularity` defined in the configuration.
            - If a formula is isomorphic to an existing formula in the warehouse, it will not be added to the evolution graph, and the corresponding trajectory piece would be merged into the next one.
            - Parameters:
                - `trajectory` (dict): The trajectory data to process
            - Returns: Dictionary containing `new_formulas` list and `evo_path` list