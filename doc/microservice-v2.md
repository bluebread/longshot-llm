# Microservice Documentation V2

This document outlines the structure and content of the API documentation for the microservices of this project in Version 2. It serves as a guide for developers to understand how to use the simplified API architecture effectively.

## Overview

1. **Warehouse**
    - Manages the storage and retrieval of data with simplified schema. 
    - Contains the following tables:
        1. *Isomorphism Hash Table*: Maps a WL hash to IDs of all isomorphic formulas in the evolution graph.
        2. *Evolution Graph Database*: Combined formula and graph data in single nodes.
        3. *Trajectory Tables*: Simplified entries with cur_avgQ field.

2. **Weapons**
    - Collect trajectories and process them locally using library components.
    - Public API:
        - `POST /weapon/rollout`: Given the number of steps and the initial formula's definition, it will run the environment for the specified number of steps and collect trajectories.
    - Including the following types of weapons:
        - **Cluster Bomb**: Randomly collects trajectories from the environment and processes them locally.
        - **Guided Missile**: Collects trajectories from the environment through RL policy and processes them locally.

## Database Schema

Because MongoDB does not have a native UUID type, we use UUIDs as strings in the database.

### Evolution Graph (Neo4j)

Each node is labeled with `FormulaNode`, and each edge is labeled with `EVOLVED_TO`. If the two adjacent nodes have the same avgQ value, they will be connected with an edge labeled `SAME_Q`. Nodes now contain integrated formula data:

| Attribute   | Type   | Description                   |
| ---------------- | :----: | ----------------------------- |
| formula\_id      | UUID   | The node's ID, corresponding to the formula ID (used as unique index)    |
| num\_vars      | int    | The number of variables in the formula represented by this node |
| width         | int    | The width of the formula represented by this node |
| size        | int    | The size of the formula represented by this node (number of nodes) |
| avgQ         | float  | The average-case deterministic query complexity of the formula represented by this node |
| wl_hash      | string | Weisfeiler-Lehman hash value |
| timestamp    | datetime | Insertion time |
| traj_id | UUID | Trajectory ID from which this formula/node can be reconstructed |
| traj_slice | int | Slice index within the trajectory for reconstruction |

### Trajectory Table (MongoDB)

Each trajectory is either a partial trajectory or the full definition of a formula.

| Column Name      | Type     | Description                                      |
|------------------|:----------:|--------------------------------------------------|
| id               | UUID     | Primary key                                      |
| timestamp        | datetime | The time when the trajectory was generated       |
| steps            | list     | List of steps in the trajectory, each step is a dictionary with the following fields: |
| step.token_type       | int   | The type of the token, 0 for ADD, 1 for DELETE    |
| step.token_literals   | int   | The binary representation for the literals of the token |
| step.cur_avgQ    | float    | Current average Q-value for this specific step |

### Isomorphism Hash Table (Redis)

- Key (*string*): WL hash value of a formula.
- Value (*List[UUID]*): the indices of probably isomorphic formulas with the same WL hash value.

## API Endpoints

### Warehouse

The Warehouse is a microservice that manages the storage and retrieval of data related to formulas, trajectories, and the evolution graph. It abstracts the complexity of data management and provides a simple interface for data access.

#### Data Format

When request is not passed in a correct format, the server will return a `422 Unprocessable Entity` error.

For every token literals in the request body, it should be represented as a 64-bits integer, the first 32 bits for the positive literals and the last 32 bits for the negative literals.

#### `GET /evolution_graph/node`
Retrieve a node in the evolution graph by its formula ID. Returns integrated formula and graph data.

- **Query Parameters:**  
    - `id` (string, required): Node UUID.

- **Response:**  
    ```json
    {
        "formula_id": "f123",
        "avgQ": 6.3,
        "num_vars": 3,
        "width": 2,
        "size": 5,
        "in_degree": 2,
        "out_degree": 3,
        "wl_hash": "abcd1234...",
        "timestamp": "2025-07-21T12:00:00Z",
        "traj_id": "t456",
        "traj_slice": 3
    }
    ```
- **Status Codes:**  
    - `200 OK`, `404 Not Found`

#### `POST /evolution_graph/node`
Add a new node to the evolution graph with integrated formula data.

- **Request Body:**  
    ```json
    {
        "formula_id": "f123",
        "avgQ": 6.3,
        "num_vars": 3,
        "width": 2,
        "size": 5,
        "wl_hash": "abcd1234...",
        "traj_id": "t456",
        "traj_slice": 3
    }
    ```
- **Response:**  
    ```json
    {
        "formula_id": "f123"
    }
    ```
- **Status Codes:**  
    - `201 Created`, `422 Unprocessable Entity`

#### `PUT /evolution_graph/node`
Update an existing node with integrated formula data.

- **Request Body:**  
    ```json
    {
        "formula_id": "f123",
        "inc_visited_counter": 11,
        "avgQ": 4.56,
        "wl_hash": "newHash123"
    }
    ```
- **Constraints**:
    - `formula_id` (string) is required to identify the node to be updated.
    - `avgQ`, `num_vars`, `width`, `size`, `wl_hash`, `traj_id`, `traj_slice` are optional fields that can be updated.
- **Response:**  
    - Success message.
- **Status Codes:**  
    - `200 OK`, `422 Unprocessable Entity`, `404 Not Found`

#### `DELETE /evolution_graph/node`
Delete a node.

- **Query Parameters:**  
    - `formula_id` (string, required): Node UUID.

- **Response:**  
    - Success message.
- **Status Codes:**  
    - `200 OK`, `404 Not Found`

#### `GET /formula/likely_isomorphic`
Retrieve IDs of likely isomorphic formulas.

- **Query Parameters:**  
    - `wl_hash` (string, required): Weisfeiler-Lehman hash.

- **Response:**  
    ```json
    {
        "isomorphic_ids": ["f123", "f124"]
    }
    ```
- **Status Codes:**  
    - `200 OK`, `404 Not Found`

#### `POST /formula/likely_isomorphic`

Add a likely isomorphic formula. If the WL hash already exists, the formula ID will be added to the existing list of isomorphic IDs. Otherwise, a new list will be created.

- **Request Body:**  
    ```json
    {
        "wl_hash": "abcd1234...",
        "formula_id": "f125"
    }
    ```
- **Response:**  
    - Success message.
- **Status Codes:**  
    - `201 Created`, `422 Unprocessable Entity`

#### `DELETE /formula/likely_isomorphic`

Delete the specified key.

- **Query Parameters:**  
    - `wl_hash` (string, required): Weisfeiler-Lehman hash.
- **Response:**  
    - Success message.
- **Status Codes:**  
    - `200 OK`, `404 Not Found`

#### `GET /trajectory`
Retrieve a trajectory by its ID.

- **Query Parameters:**  
    - `id` (string, required): Trajectory UUID.

- **Response:**  
    ```json
    {
        "id": "t456",
        "steps": [
            {
                "token_type": 0,
                "token_literals": 5,
                "cur_avgQ": 2.3
            }
        ],
        "timestamp": "2025-07-21T12:00:00Z"
    }
    ```
- **Status Codes:**  
    - `200 OK`, `404 Not Found`

#### `POST /trajectory`
Add a new trajectory. The fields `id` and `timestamp` are automatically generated by the server.

- **Request Body:**  
    ```json
    {
        "steps": [
            {
                "token_type": 0,
                "token_literals": 5,
                "cur_avgQ": 2.3
            }
        ]
    }
    ```
- **Response:**  
    ```json
    {
        "id": "t456"
    }
    ```
- **Status Codes:**  
    - `201 Created`, `422 Unprocessable Entity`

#### `PUT /trajectory`
Update an existing trajectory.

- **Request Body:**  
    ```json
    {
        "id": "t456",
        "steps": [
            {
                "order": 3,
                "token_type": 1,
                "token_literals": 3,
                "cur_avgQ": 2.8
            }
        ]
    }
    ```
- **Response:**  
    - Success message.
- **Status Codes:**  
    - `200 OK`, `422 Unprocessable Entity`, `404 Not Found`

#### `DELETE /trajectory`
Delete a trajectory.

- **Query Parameters:**  
    - `id` (string, required): Trajectory UUID.

- **Response:**  
    - Success message.
- **Status Codes:**  
    - `200 OK`, `404 Not Found`

#### `GET /formula/definition`
Retrieve the full definition of a formula by its ID.

- **Query Parameters:**  
    - `id` (string, required): Formula UUID.

- **Response:**  
    ```json
    {
        "formula_id": "f123",
        "definition": [
            ["x1", "x2", "x3"],
            ["x4", "x5"]
        ]
    }
    ```
- **Status Codes:**  
    - `200 OK`, `404 Not Found`

#### `POST /evolution_graph/path`

Add a new path to the evolution graph. The path is a list of formula IDs that represent the evolution path of the formulas in the trajectory.

- **Request Body:**  
    ```json
    {
        "path": [
            "f123",
            "f124", 
            "f125"
        ]
    }
    ```
- **Response:**  
    - Success message.
- **Status Codes:**  
    - `201 Created`, `422 Unprocessable Entity`

#### `GET /evolution_graph/download_nodes`

Get the evolution subgraph of nodes satisfying the given conditions (e.g. `num_vars`, `width`, etc.). Returns nodes with integrated formula data.

- **Query Parameters:**  
    - `num_vars` (int): The number of variables in the formula.
    - `width` (int): The width of the formula.
    - `size_constraint` (int, optional): The maximum size of the formula. Default: None.
- **Response:**  
    ```json
    {
        "nodes": [
            {
                "formula_id": "f123",
                "avgQ": "QAkh+1RBF0Q=",
                "num_vars": 3,
                "width": 2,
                "size": 5,
                "in_degree": 2,
                "out_degree": 3,
                "wl_hash": "abcd1234...",
                "timestamp": "2025-07-21T12:00:00Z",
                "traj_id": "t456",
                "traj_slice": 3
            }
        ]
    }
    ```
- **Status Codes:**
    - `200 OK`, `422 Unprocessable Entity`

#### `GET /evolution_graph/download_hypernodes`

Get the hypernodes of the evolution graph. A hypernode is a connected component of nodes with the same avgQ. A hypernode of size 1 would be omitted.

- **Query Parameters:**
    - `num_vars` (int): The number of variables in the formula.
    - `width` (int): The width of the formula.
    - `size_constraint` (int, optional): The maximum size of the formula. Default: None.

- **Response:**
    ```json
    {
        "hypernodes": [
            {
                "hnid": 1,
                "nodes": [
                    "f123",
                    "f124",
                    "f125"
                ]
            }
        ]
    }
    ```
- **Status Codes:**
    - `200 OK`, `422 Unprocessable Entity`

---

### Weapons

The Weapons microservice is responsible for collecting trajectories and processing them locally using the integrated library components. It provides a public API for external clients to interact with the system and collect trajectories from the environment.

#### `POST /weapon/rollout`

Collects trajectories from the environment and processes them locally using TrajectoryProcessor and ArmRanker modules. The request should specify the number of steps to run and the initial formula's definition.

- Request Body:
    ```json
    {
        "num_vars": 3,
        "width": 2,
        "size": 10,
        "steps_per_trajectory": 100,
        "num_trajectories": 10,
        "initial_formula_id": "f123",
        "initial_definition": [
            1, 2, 3
        ],
        "seed": 42
    }
    ```
- Request Field Descriptions:
    - `num_vars` (int): Number of variables in the formula.
    - `width` (int): Width of the formula.
    - `size` (int): Size of the formula (number of nodes).
    - `steps_per_trajectory` (int): Number of steps to run in a single trajectory.
    - `num_trajectories` (int): Number of trajectories to collect.
    - `initial_formula_id` (string): ID of the initial formula to start with. If not provided, a random formula will be used.
    - `initial_definition` (list[int]): Initial definition of the formula, represented as a list of lists of literals (represented by integers).
    - `seed` (int, optional): Random seed for reproducibility. Default: None.
- Response:
    ```json
    {
        "total_steps": 1000,
        "num_trajectories": 10,
        "processed_formulas": 25,
        "new_nodes_created": 12
    }
    ```
- Status Codes:
    - `200 OK`: Successfully collected trajectories and processed them locally.
    - `422 Unprocessable Entity`: Invalid request parameters. This includes:
        - Missing required fields or invalid data types
        - Formula parameter conflicts (e.g., `initial_definition` incompatible with specified `width` or `size` constraints)
        - Formula definition that results in a width greater than the specified `width` parameter
    - `500 Internal Server Error`: Unexpected server issues (e.g., warehouse connection failures).