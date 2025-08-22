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

**V2 Schema Changes**: The V2 architecture eliminates linked list structures in trajectory data by embedding complete formula reconstruction information within each trajectory record.

Because MongoDB does not have a native UUID type, we use UUIDs as strings in the database.

### Evolution Graph (Neo4j)

Each node is labeled with `FormulaNode`, and each edge is labeled with `EVOLVED_TO`. If the two adjacent nodes have the same avgQ value, they will be connected with an edge labeled `SAME_Q`. Nodes now contain integrated formula data:

| Attribute   | Type   | Description                   |
| ---------------- | :----: | ----------------------------- |
| node\_id      | UUID   | The node's ID, corresponding to the formula ID (used as unique index)    |
| num\_vars      | int    | The number of variables in the formula represented by this node |
| width         | int    | The width of the formula represented by this node |
| size        | int    | The size of the formula represented by this node (number of nodes) |
| avgQ         | float  | The average-case deterministic query complexity of the formula represented by this node |
| wl_hash      | string | Weisfeiler-Lehman hash value |
| timestamp    | datetime | Insertion time |
| traj_id | UUID | Trajectory ID from which this formula/node can be reconstructed |
| traj_slice | int | **V2**: Index pointing to the final step in the complete trajectory (prefix + suffix) that represents this formula state |

### Trajectory Table (MongoDB)

**V2 Schema**: Each trajectory contains the COMPLETE formula construction sequence, combining both base formula reconstruction (prefix) and new exploration steps (suffix). This eliminates the linked list structure and enables direct formula reconstruction.

| Column Name      | Type     | Description                                      |
|------------------|:----------:|--------------------------------------------------|
| traj_id               | UUID     | Primary key                                      |
| timestamp        | datetime | The time when the trajectory was generated       |
| steps            | list     | **V2**: Complete trajectory steps including both prefix (base formula) and suffix (new exploration) |
| step.token_type       | int   | The type of the token, 0 for ADD, 1 for DELETE    |
| step.token_literals   | int   | The binary representation for the literals of the token |
| step.cur_avgQ    | float    | **V2**: Current average Q-value for this specific step (required for all steps) |

**V2 Key Changes**:
- **Complete Trajectories**: No more partial/linked trajectories - each record contains full reconstruction sequence
- **Embedded Base Formulas**: Prefix trajectory steps embedded within each trajectory record
- **Direct Reconstruction**: Formula definitions reconstructed directly from trajectory steps without database lookups
- **Consistent avgQ**: All steps include `cur_avgQ` field for trajectory analysis

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
        "node_id": "f123",
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
Add a new node to the evolution graph with integrated formula data. The node_id is automatically generated by the system using UUID4.

- **Request Body:**  
    ```json
    {
        "avgQ": 2.75,
        "num_vars": 3,
        "width": 2,
        "size": 5,
        "wl_hash": "abc123def456",
        "traj_id": "67890abcdef",
        "traj_slice": 3
    }
    ```
- **Constraints:**
    - All fields are required:
        - `avgQ` (float): Average Q-value complexity metric
        - `num_vars` (int): Number of variables in the formula
        - `width` (int): Width constraint of the formula
        - `size` (int): Size (number of gates) of the formula
        - `wl_hash` (string): Weisfeiler-Lehman hash for isomorphism detection
        - `traj_id` (string): ID of the associated trajectory
        - `traj_slice` (int): Slice position within the trajectory
- **Response:**  
    ```json
    {
        "node_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    ```
- **Status Codes:**  
    - `201 Created`: Node created successfully
    - `422 Unprocessable Entity`: Invalid request data
    - `500 Internal Server Error`: Failed to create node

#### `PUT /evolution_graph/node`
Update an existing node with integrated formula data in the V2 system.

- **Request Body:**  
    ```json
    {
        "node_id": "node_f123e456_7",
        "avgQ": 2.75,
        "num_vars": 3,
        "width": 2,
        "size": 8,
        "wl_hash": "abc123def456",
        "traj_id": "67890abcdef",
        "traj_slice": 5
    }
    ```
- **Constraints**:
    - `node_id` (string) is required to identify the node to be updated.
    - All other fields are optional and can be updated independently:
        - `avgQ` (float): Average Q-value complexity metric
        - `num_vars` (int): Number of variables in the formula
        - `width` (int): Width constraint of the formula
        - `size` (int): Size (number of gates) of the formula
        - `wl_hash` (string): Weisfeiler-Lehman hash for isomorphism detection
        - `traj_id` (string): ID of the associated trajectory
        - `traj_slice` (int): Slice position within the trajectory
- **Response:**  
    ```json
    {
        "message": "Node updated successfully"
    }
    ```
- **Status Codes:**  
    - `200 OK`: Node updated successfully
    - `400 Bad Request`: No update data provided
    - `404 Not Found`: Node with specified ID not found
    - `422 Unprocessable Entity`: Invalid request data

#### `DELETE /evolution_graph/node`
Delete a node.

- **Query Parameters:**  
    - `node_id` (string, required): Node UUID.

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
        "wl_hash": "abcd1234...",
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
        "node_id": "f125"
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
    - `traj_id` (string, required): Trajectory UUID.

- **Response:**  
    ```json
    {
        "traj_id": "t456",
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
Add a new trajectory. The fields `traj_id` and `timestamp` are automatically generated by the server.

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
        "traj_id": "t456"
    }
    ```
- **Status Codes:**  
    - `201 Created`, `422 Unprocessable Entity`

#### `PUT /trajectory`
Update an existing trajectory.

- **Request Body:**  
    ```json
    {
        "traj_id": "t456",
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
    - `traj_id` (string, required): Trajectory UUID.

- **Response:**  
    - Success message.
- **Status Codes:**  
    - `200 OK`, `404 Not Found`

#### `GET /formula/definition`
Retrieve the full definition of a formula by its ID.

- **Query Parameters:**  
    - `node_id` (string, required): Formula node UUID.

- **Response:**  
    ```json
    {
        "node_id": "f123",
        "definition": [
            5431, 53, 54
        ]
    }
    ```
- **Status Codes:**  
    - `200 OK`, `404 Not Found`

#### `POST /evolution_graph/path`

Add a new path to the evolution graph. The path is a list of formula IDs that represent the evolution path of the formulas in the trajectory. Adjacent nodes on the path would be connected by EVOLVED_TO edge. When the two adjacent nodes have the same avgQ, they wil be connected a SAME_Q edge. 

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
                "node_id": "f123",
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

#### `GET /evolution_graph/dataset`

Get the complete evolution graph dataset including all nodes and edges with optional field filtering to reduce data redundancy.

- **Query Parameters:**
    - `required_fields` (list[str], optional): List of required node fields to include. Default: `["node_id"]`. Available fields: `node_id`, `avgQ`, `num_vars`, `width`, `size`, `wl_hash`, `timestamp`, `traj_id`, `traj_slice`. Note: `node_id` is always included.

- **Response:**
    ```json
    {
        "nodes": [
            {
                "node_id": "f123",
                "avgQ": 0.85,
                "num_vars": 3,
                "width": 2,
                "size": 5,
                "wl_hash": "abc123",
                "timestamp": "2023-01-01T00:00:00Z",
                "traj_id": "t456",
                "traj_slice": 10
            }
        ],
        "edges": [
            {
                "src": "f123",
                "dst": "f124",
                "type": "EVOLVED_TO"
            },
            {
                "src": "f123",
                "dst": "f125",
                "type": "SAME_Q"
            }
        ]
    }
    ```
- **Status Codes:**
    - `200 OK`

#### `GET /trajectory/dataset`

Get the complete trajectory dataset with all trajectories using optimized tuple format for steps.

- **Query Parameters:** None

- **Response:**
    ```json
    {
        "trajectories": [
            {
                "traj_id": "t456",
                "timestamp": "2023-01-01T00:00:00Z",
                "steps": [
                    [0, 5, 0.75],
                    [1, 3, 0.85]
                ]
            }
        ]
    }
    ```
    Note: Each step is represented as a tuple `[token_type, token_literals, cur_avgQ]` for reduced data redundancy.

- **Status Codes:**
    - `200 OK`

---

### Weapons

The Weapons microservice is responsible for collecting trajectories and processing them locally using the integrated library components. It provides a public API for external clients to interact with the system and collect trajectories from the environment.

#### `POST /weapon/rollout`

Collects trajectories from the environment and processes them locally using V2 TrajectoryProcessor with embedded formula reconstruction. The V2 schema eliminates the linked list structure by providing complete trajectory data for base formula reconstruction.

- Request Body:
    ```json
    {
        "num_vars": 3,
        "width": 2,
        "size": 10,
        "steps_per_trajectory": 100,
        "num_trajectories": 10,
        "prefix_traj": [
            {
                "token_type": 0,
                "token_literals": 3,
                "cur_avgQ": 0.5
            },
            {
                "token_type": 0,
                "token_literals": 4,
                "cur_avgQ": 1.0
            }
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
    - `prefix_traj` (list[TrajectoryStep]): **V2 REQUIRED** - Complete trajectory for base formula reconstruction. Contains all steps needed to build the initial state, eliminating the need for linked list traversal.
    - `seed` (int, optional): Random seed for reproducibility. Default: None.
- TrajectoryStep Format:
    - `token_type` (int): Type of operation (0=ADD, 1=DEL, 2=EOS)
    - `token_literals` (int): 64-bit integer representation of literals (first 32 bits positive, last 32 bits negative)
    - `cur_avgQ` (float): Average Q-value after this step
- Response:
    ```json
    {
        "total_steps": 1000,
        "num_trajectories": 10,
        "processed_formulas": 25,
        "new_nodes_created": 12,
        "base_formula_exists": false
    }
    ```
- Response Field Descriptions:
    - `total_steps` (int): Total number of steps executed across all trajectories
    - `num_trajectories` (int): Number of trajectories actually collected
    - `processed_formulas` (int): **V2** - Number of unique formulas processed from trajectory segments
    - `new_nodes_created` (int): **V2** - Number of new nodes created in the evolution graph
    - `base_formula_exists` (bool): **V2** - Whether the base formula (from prefix_traj) already exists in the database
- Status Codes:
    - `200 OK`: Successfully collected trajectories and processed them using V2 schema.
    - `422 Unprocessable Entity`: Invalid request parameters. This includes:
        - Missing required `prefix_traj` field
        - Invalid trajectory step format
        - Formula parameter conflicts
    - `500 Internal Server Error`: Unexpected server issues (e.g., warehouse connection failures).

##### Clusterbomb Algorithm

1. Reconstruct base formula from prefix trajectory. Initialize temporary processor for base formula reconstruction. 
2. Check if base formula already exists in database. If not, it would be saved to the warehouse. 
3. Initialize the RL environment (FormulaGame), warehouse agent and trajectory processor.
4. Run the environment simulation.
5. Process all trajectories.


## V2 Architecture Changes

### Trajectory Schema Evolution

**V1 Problems (Deprecated)**:
- Trajectories formed a linked list structure via `initial_node_id` references
- Required complex backtracking to reconstruct formula definitions
- Empty formula initialization caused database consistency issues
- Performance degradation due to recursive formula retrieval

**V2 Solutions**:
- **Embedded Reconstruction**: Each trajectory contains complete `prefix_traj` for base formula reconstruction
- **No Linked Lists**: Eliminates dependency chains and backtracking requirements
- **Complete Trajectories**: Combined prefix + suffix stored as single trajectory record
- **Improved Performance**: Direct formula reconstruction without warehouse queries
- **Database Consistency**: Base formula existence checking prevents orphaned references

### Processing Flow

1. **Base Formula Reconstruction**: Use `prefix_traj` to rebuild initial formula state locally
2. **Duplicate Detection**: Check if base formula already exists in database
3. **Trajectory Segmentation**: Process `suffix_traj` in granulated pieces as before
4. **Complete Storage**: Store combined (prefix + suffix) trajectory with correct `traj_slice` references
5. **Node Creation**: Create evolution graph nodes with proper trajectory references

### Migration Notes

- **Backward Compatibility**: V1 fields (`initial_definition`, `initial_node_id`) are deprecated but temporarily supported
- **Required Migration**: New integrations must use `prefix_traj` field
- **Performance Benefits**: V2 eliminates O(n) traversal costs for formula definition retrieval
- **Data Consistency**: V2 prevents empty formula initialization problems in demo scripts