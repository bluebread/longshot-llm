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