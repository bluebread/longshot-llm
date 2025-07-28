# RL-API Documentation

This document outlines the structure and content of the API documentation for the RL part of this project. It serves as a guide for developers to understand how to use the API effectively.

## Overview

### Local Modules

1. **Formula Game**
    - Implements the RL environment that simulates the process of adding/deleting gates in a normal formed formula. 
    - Calculates *average-case deterministic query complexity*, the optimization target.
    - Main functions:
        1. `reset()`: Resets internal variables. 
        2. `step()`: Given the passed token (which indicates adding or deleting a gate), simulates a step and returns the reward. 
2. **Environment Agent**
    - Manages environments. 
    - Transforms data in Tensor/TensorDict format. 
    - Main functions:
        1. `replace_arms()`: Replaces all arms/environments using the arm filter. 
        2. `reset()`: Resets formula games and saves trajectories to the trajectory queue (except the first time calling `reset()`). 
        3. `step()`: Executes a step of formula games. 
3. **Trajectory Queue Agent**
    - Manages the trajectory queue using RabbitMQ.
    - Provides methods to push and pop trajectories.
    - Main functions:
        1. `push()`: Pushes a trajectory to the RabbitMQ queue.
        2. `pop()`: Pops a trajectory from the RabbitMQ queue.
        3. `start_consuming()`: Starts consuming messages from the RabbitMQ queue.
        4. `close()`: Closes the connection to RabbitMQ.
    - This class is a local module, which is accessible for any other components in the RL system.
4. **Trainer**
    - Trains a RL policy that learns how to build a CNF/DNF formula with the largest average-case deterministic query complexity.
    - Retrieves dataset (trajectories) from the environment (wrapper). 
    - Includes but not limited to the following modules:
        1. Replay Buffer (able to handle both Markovian and non-Markovian sequences)
        2. General Advantage Estimation（optional）
        3. Optimizer
        4. Scheduler
        5. Loss Function
        6. Gumbel-Topk Distribution
        7. Policy Network
        8. Critic Network

### Microservices

1. **Trajectory Queue**
    - RabbitMQ interface.
    - Handles the queuing of trajectories for processing.
2. **Warehouse**
    - Manages the storage and retrieval of data. It hides the complexity of data management and provides a simple interface for data access.
    - Contains the following tables:
        1. *Isomorphism Hash Table*: Maps a WL hash to IDs of all isomorphic formuals in the formula table. 
        2. *Formula Table*: Each entry records (ID, timestamp, BaseFormulaID, TrajectoryID, complexity, hash, #vars, width, size, NodeID, FullTrajectoryID). 
        3. *Trajectory Tables*: Each tables contains entries of (order, token_type, token_literals, reward). 
        4. *Evolution Graph Database*:
            - graph label: (#vars, width)
            - nodes: (FormulaTableID, visited_counter)
            - edges: (BaseFormulaID, NewFormulaID, distance)
    - Low-level API:
        - `GET /formula/info`: Retrieves information about a formula by its ID.
        - `POST /formula/info`: Adds a new formula entry to the formula table.
        - `PUT /formula/info`: Updates an existing formula entry in the formula table.
        - `DELETE /formula/info`: Deletes a formula entry from the formula table.
        - `GET /formula/likely_isomorphic`: Retrieves likely isomorphic formulas' IDs.
        - `POST /formula/likely_isomorphic`: Adds a likely isomorphic formula.
        - `GET /trajectory`: Retrieves a trajectory by its ID.
        - `POST /trajectory`: Adds a new trajectory to the warehouse.
        - `PUT /trajectory`: Updates an existing trajectory in the warehouse.
        - `DELETE /trajectory`: Deletes a trajectory from the warehouse.
        - `GET /evolution_graph/node`: Retrieves a node in the evolution graph by its ID.
        - `POST /evolution_graph/node`: Adds a new node to the evolution graph.
        - `PUT /evolution_graph/node`: Updates an existing node in the evolution graph.
        - `DELETE /evolution_graph/node`: Deletes a node from the evolution graph.
        - `GET /evolution_graph/edge`: Retrieves an edge in the evolution graph by its ID.
        - `POST /evolution_graph/edge`: Adds a new edge to the evolution graph.
        - `PUT /evolution_graph/edge (unused)`: Updates an existing edge in the evolution graph.
        - `DELETE /evolution_graph/edge`: Deletes an edge from the evolution graph.
    - High-level API:
        - `GET /formula/definition`: Retrieves the full definition of a formula by its ID.
        - `GET /evolution_graph/subgraph`: Retrieves the evolution subgraph of active nodes. 
        - `POST /formula/add`: Adds a new formula to the warehouse, including updating the isomorphism hash table and the evolution graph.
        - `POST /evolution_graph/subgraph`: Adds a new subgraph to the evolution graph of a formula.
        - `POST /evolution_graph/contract_edge`: Contracts an edge in the evolution graph of a formula. One of the nodes will be deactivated.

3. Arm Filter
    - Filters and selects the best arms (formulas) based on the trajectories and the evolution graph.
    - Maintains the evolution graph of formulas.
    - Implements policies for arm selection (e.g., UCB algorithm).
    - Submodules:
        1. *Trajectory Processor*: Processes trajectories and updates the evolution graph.
        2. *Evolution Graph Manager*: Manages the evolution graph and its updates.
        3. *Arm Ranker*: Ranks the arms (formulas) based on their performance and potential.
    - Main API:
        - `GET /topk_arms`: Return the current best top-K arms.


## Database Schema

Because MongoDB does not have a native UUID type, we use UUIDs as strings in the database. 

### Formula Table (MongoDB)

In the database, the formula table is labeled as `FormulaTable`. Each entry in the formula table represents a formula and contains the following columns:

| Column Name       | Type        | Description                          |
|:------------------|:-----------:|:--------------------------------------|
| id                | UUID        | Primary key                                 |
| base_formula_id   | UUID        | Parent formula ID                             |
| trajectory_id     | UUID        | Associated trajectory table ID            |
| avgQ        | string       | Average-case deterministic query complexity (in base64 format)                             |
| wl_hash              | string      | Weisfeiler-Lehman hash value              |
| num_vars          | int         | Number of variables                               |
| width             | int         | Formula width                               |
| size              | int         | Formula size (number of nodes)                     |
| timestamp         | datetime    | Insertion time                               |
| node_id         | UUID        | Node ID in the evolution graph  |

### Definition Cache Table (Redis)

- Key (*UUID*): the ID of the formula.
- Value (*List[UUID]*): the indices of trajectories that can be used to reconstruct the full definition of the formula.

### Trajectory Table (MongoDB)

Each trajectory is either a partial trajectory or the full definition of a formula, and its table ID must be recorded somewhere in the formula table.

| Column Name      | Type     | Description                                      |
|------------------|:----------:|--------------------------------------------------|
| id               | UUID     | Primary key                                      |
| steps            | list     | List of steps in the trajectory, each step is a dictionary with the following fields: |
| step.token_type       | bool   | The type of the token, 0 for ADD, 1 for DELETE    |
| step.token_literals   | int   | The binary representation for the literals of the token |
| step.reward           | string    | The immediate reward received after this step (in base64 format)                      |


### Isomorphism Hash Table (Redis)

- Key (*string*): WL hash value of a formula.
- Value (*List[UUID]*): the indices of probably isomorphic formulas with the same WL hash value.

### Evolution Graph (Neo4j)

Each graph is labeled with `N<num_vars>W<width>`, where `num_vars` is the number of variables and `width` is the width of the formula.

#### Node Entity

| Attribute   | Type   | Description                   |
| ---------------- | :----: | ----------------------------- |
| formula\_id      | UUID   | The node's ID, corresponding to the primary key of the FormulaTable    |
| avgQ         | float  | The average-case deterministic query complexity of the formula represented by this node |
| visited\_counter | int    | The number of times this node has been touched by a trajectory                   |
| inactive         | bool   | Whether this node is inactive. If inactive, it will not be collected while traversing the graph.                    |
| in_degree        | int    | The in-degree of this node in the evolution graph  |
| out_degree       | int    | The out-degree of this node in the evolution graph  |

#### Edge Entity

| Attribute   | Type   | Description                   |
| ---------------- | :----: | ----------------------------- |
| base_formula_id | UUID   | The ID of the base formula, corresponding to the primary key of FormulaTable |
| new_formula_id  | UUID   | The ID of the new formula, corresponding to the primary key of FormulaTable    |

## Microservice


### Trajectory Queue

The Trajectory Queue is a RabbitMQ queue used for temporarily buffering trajectories generated by the reinforcement learning (RL) environment. This allows asynchronous decoupling between the environment that produces trajectories and the components (e.g., training workers or data processors) that consume them.

It provides a reliable mechanism for pushing and popping serialized trajectory data for downstream processing.



#### Message Queue Definition

| Property | 	Value |
| -------- | ------- |
| Queue Name | trajectory.queue |
| Exchange | trajectory.exchange |
| Exchange Type | direct |
| Routing Key | trajectory.push |
| Durable | true |
| Auto-delete | false |
| Ack Mode | Manual acknowledgment |
| Content Type | application/json |


#### Message Schema (JSON)

```json
{
  "num_vars": 3,
  "width": 2,
  "timestamp": "2025-07-21T12:00:00Z",
  "trajectory": {
    "base_formula_id": "f123",
    "steps": [
      {
        "order": 0,
        "token_type": "ADD",
        "token_literals": ["x1", "x2"],
        "reward": 0.1,
        "avgQ": 2.5
      },
      {
        "order": 1,
        "token_type": "DEL",
        "token_literals": ["x2"],
        "reward": -0.05,
        "avgQ": 3.0
      }
    ]
  }
}
```


##### Field Descriptions

| Field                 | Type   | Description                                                 |
| --------------------- | :------: | ----------------------------------------------------------- |
| `num_vars`            | int    | Number of variables in the formula                          |
| `width`               | int    | Width of the formula                                        |
| `timestamp`           | string | ISO 8601 timestamp of when the trajectory was generated   |
| `trajectory`        | object | Contains the trajectory data                                |
| `base_formula_id`    | string | The unique ID of the base formula that the trajectory applies to |
| `steps`               | array  | List of step objects in the trajectory                      |
| `step.order`          | int    | The position of the action in the sequence                  |
| `step.token_type`     | string | One of `"ADD"`, `"DEL"`, `"EOS"`                                     |
| `step.token_literals` | array  | The binary representation for literals involved in the operation                          |
| `step.reward`         | float  | The reward received at this step                            |
| `step.avgQ`           | float  | The average-case deterministic query complexity of the formula after this step |


#### Message Consuming Setting (Pop) 

| Property | 	Value |
| -------- | ------- |
|	Queue | trajectory.queue
|	Auto Ack | false
|	Expected Content Type | application/json


#### Example (Push)

```python
# Push a trajectory to the RabbitMQ queue
address = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(address)
channel = connection.channel()

# Declare the exchange and queue
channel.exchange_declare(
    exchange='trajectory.exchange', 
    exchange_type='direct'
    )
channel.queue_declare(
    queue='trajectory.queue',
    auto_delete=False,
    durable=True
    )

# Push a trajectory
message = json.dumps(trajectory)
channel.basic_publish(
    exchange='trajectory.exchange',
    routing_key='trajectory.push',
    body=message,
    properties=pika.BasicProperties(
        content_type='application/json',
        delivery_mode=2  # make message persistent
    ))

print(" [x] Sent trajectory")
connection.close()
```

#### Example (Pop)

```python
# Receive and process messages from the RabbitMQ queue
address = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(address)
channel = connection.channel()

# Declare the queue
channel.queue_declare(queue='trajectory.queue', durable=True)

# Set up the callback function to process messages
def callback(ch, method, properties, body):
    data = json.loads(body)
    print(" [x] Received data:", data)
    # Acknowledge the message
    ch.basic_ack(delivery_tag=method.delivery_tag)  

channel.basic_consume(
    queue='trajectory.queue', 
    on_message_callback=callback,
    auto_ack=False  # Set to False to manually acknowledge messages
    )

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
```

---

### Warehouse

The Warehouse is a microservice that manages the storage and retrieval of data related to formulas, trajectories, and the evolution graph. It abstracts the complexity of data management and provides a simple interface for data access. 

When request is not passed in a correct format, the server will return a `422 Unprocessable Entity` error.

#### `GET /formula/info`
Retrieve information about a formula by its ID.

- **Query Parameters:**  
    - `id` (string, required): Formula UUID.

- **Response:**  
    ```json
    {
        "id": "f123",
        "base_formula_id": "f122",
        "trajectory_id": "t456",
        "avgQ": 2.5,
        "wl_hash": "abcd1234...",
        "num_vars": 3,
        "width": 2,
        "size": 5,
        "timestamp": "2025-07-21T12:00:00Z",
        "node_id": "n789",
    }
    ```
- **Status Codes:**  
    - `200 OK`, `404 Not Found`


#### `POST /formula/info`
Add a new formula entry to the formula table. The fields `id` and `timestamp` are optional and would be automatically generated by the server if not provided. 


- **Request Body:**  
    ```json
    {
        "base_formula_id": "f122",
        "trajectory_id": "t456",
        "avgQ": 2.5,
        "wl_hash": "abcd1234...",
        "num_vars": 3,
        "width": 2,
        "size": 5,
        "node_id": "n789"
    }
    ```
- **Response:**  
    ```json
    {
        "id": "f123"
    }
    ```
- **Status Codes:**  
    - `201 Created`, `422 Unprocessable Entity`


#### `PUT /formula/info`
Update an existing formula entry.

- **Request Body:**  
    ```json
    {
        "id": "f123",
        "avgQ": 2.7,
        "size": 6
    }
    ```
- **Response:**  
    - Success message or updated formula object.
- **Status Codes:**  
    - `200 OK`, `422 Unprocessable Entity`, `404 Not Found`


#### `DELETE /formula/info`
Delete a formula entry.

- **Query Parameters:**  
    - `id` (string, required): Formula UUID.

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
Add a likely isomorphic formula.

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
                "order": 0,
                "token_type": 0,
                "token_literals": 5,
                "reward": 0.1
            }
        ]
    }
    ```
- **Status Codes:**  
    - `200 OK`, `404 Not Found`


#### `POST /trajectory`
Add a new trajectory.

- **Request Body:**  
    ```json
    {
        "steps": [
            {
                "order": 0,
                "token_type": 0,
                "token_literals": 5,
                "reward": 0.1
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
        "steps": [ ... ]
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


#### `GET /evolution_graph/node`
Retrieve a node in the evolution graph by its ID.

- **Query Parameters:**  
    - `id` (string, required): Node UUID.

- **Response:**  
    ```json
    {
        "formula_id": "f123",
        "avgQ": 2.5,
        "visited_counter": 10,
        "inactive": false,
        "in_degree": 2,
        "out_degree": 3
    }
    ```
- **Status Codes:**  
    - `200 OK`, `404 Not Found`


#### `POST /evolution_graph/node`
Add a new node to the evolution graph. The visited counter is set to 1, and the inactive flag is set to false by default.

- **Request Body:**  
    ```json
    {
        "formula_id": "f123",
        "avgQ": 2.5
    }
    ```
- **Response:**  
    ```json
    {
        "node_id": "n789"
    }
    ```
- **Status Codes:**  
    - `201 Created`, `422 Unprocessable Entity`


#### `PUT /evolution_graph/node`
Update an existing node. The parameter ``inc_visited_counter`` is used to increment the visited counter by a specified amount and suppose to be greater than 0. 

- **Request Body:**  
    ```json
    {
        "node_id": "n789",
        "inc_visited_counter": 11,
        "inactive": true
    }
    ```
- **Response:**  
    - Success message.
- **Status Codes:**  
    - `200 OK`, `422 Unprocessable Entity`, `404 Not Found`


#### `DELETE /evolution_graph/node`
Delete a node.

- **Query Parameters:**  
    - `node_id` (string, required): Node UUID.

- **Response:**  
    - Success message.
- **Status Codes:**  
    - `200 OK`, `404 Not Found`


#### `GET /evolution_graph/edge`
Retrieve an edge in the evolution graph by its ID.

- **Query Parameters:**  
    - `edge_id` (string, required): Edge UUID.

- **Response:**  
    ```json
    {
        "base_formula_id": "f123",
        "new_formula_id": "f124"
    }
    ```
- **Status Codes:**  
    - `200 OK`, `404 Not Found`


#### `POST /evolution_graph/edge`
Add a new edge.

- **Request Body:**  
    ```json
    {
        "base_formula_id": "f123",
        "new_formula_id": "f124"
    }
    ```
- **Response:**  
    ```json
    {
        "edge_id": "e456"
    }
    ```
- **Status Codes:**  
    - `201 Created`, `422 Unprocessable Entity`


#### `PUT /evolution_graph/edge  (unused)`
Update an existing edge. Unused for now since no attribute is stored in the edge entity.


#### `DELETE /evolution_graph/edge`
Delete an edge.

- **Query Parameters:**  
    - `edge_id` (string, required): Edge UUID.

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
        "id": "f123",
        "definition": [
            ["x1", "x2", "x3"],
            ["x4", "x5"]
        ]
    }
    ```
- **Status Codes:**  
    - `200 OK`, `404 Not Found`


##### TODO: Lazy Reconstruction and Caching
- Warehouse does not store the full definition of each formula initially.
- Instead, formulas are reconstructed recursively using `BaseFormulaID` and `TrajectoryID`.
- A lazy caching system is adopted: once a formula is reconstructed, its `FullTrajectoryID` may be stored for future faster access.
- Periodic or opportunistic checkpointing can be applied to avoid long chains of reconstruction.
- How frequently to reconstruct and cache is a design choice that can be tuned based on performance needs.

#### `GET /evolution_graph/subgraph`
Retrieve the evolution subgraph of active nodes.

- **Query Parameters:**  
    - `num_vars` (int, required)
    - `width` (int, required)

- **Response:**  
    ```json
    {
        "nodes": [ ... ],
        "edges": [ ... ]
    }
    ```
- **Status Codes:**  
    - `200 OK`, `422 Unprocessable Entity`


#### `POST /formula/add`
Add a new formula to the warehouse, updating the isomorphism hash table and evolution graph.

- **Request Body:**  
    ```json
    {
        "base_formula_id": "f122",
        "trajectory_id": "t456",
        "avgQ": 2.5,
        "wl_hash": "abcd1234...",
        "num_vars": 3,
        "width": 2,
        "size": 5
    }
    ```
- **Response:**  
    ```json
    {
        "id": "f123"
    }
    ```
- **Status Codes:**  
    - `201 Created`, `422 Unprocessable Entity`


#### `POST /evolution_graph/subgraph`
Add a new subgraph to the evolution graph of a formula.

- **Request Body:**  
    ```json
    {
        "nodes": [ ... ],
        "edges": [ ... ]
    }
    ```
- **Response:**  
    - Success message.
- **Status Codes:**  
    - `201 Created`, `422 Unprocessable Entity`


#### `POST /evolution_graph/contract_edge`
Contract an edge in the evolution graph; one node will be deactivated.

- **Request Body:**  
    ```json
    {
        "edge_id": "e456"
    }
    ```
- **Response:**  
    - Success message.
- **Status Codes:**  
    - `200 OK`, `422 Unprocessable Entity`, `404 Not Found`





---

### Arm Filter


The Arm Filter is a microservice responsible for filtering and selecting the best arms (formulas) based on the trajectories and the evolution graph. It maintains the evolution graph of formulas and implements policies for arm selection, such as the Upper Confidence Bound (UCB) algorithm.

#### `GET /topk_arms`

##### Description

Returns the current best top-K arms based on the latest trajectories and evolution graph. If parameter `size` is provided, it will return only the formulas of size less than or equal to `size`. If `size` is not provided, it will return the formulas of any size.

##### Endpoint
```
GET /topk_arms
```

##### Query Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| num_vars | int | The number of variables in the formula  |
| width    | int | The width of the formula           |
| k       | int | The number of top arms to return  |
| size     | int | The maximum size of the formula. Default: None |


##### Response (Success)

```json
{
  "top_k_arms": [
    {
      "formula_id": "f123",
      "definition": [
        ["x1", "x2", "x3"],
        ["x4", "x5"]
      ]
    },
    {
      "formula_id": "f124",
      "definition": [
        ["x6", "x7"],
        ["x8", "x9", "x10"]
      ],
    }
  ]
}
```

##### Response Field Descriptions

| Field          | Type   | Description                                   |
| -------------- | :-----: | --------------------------------------------- |
| `top_k_arms`   | array  | List of top-K arms                             |
| `arm.formula_id`   | string | The unique ID of the formula                   |
| `arm.definition`   | array  | The definition of the formula, represented as a list of lists of literals |


##### Status Codes

* `200 OK`: Successfully returned top-K arms
* `422 Unprocessable Entity`: Missing required parameters or invalid values
* `500 Internal Server Error`: Failed to retrieve data



## Local Modules

### `Class service.utils.TrajectoryQueueAgent(host: str, port: int = 5672)`

The `TrajectoryQueueAgent` class provides a high-level interface for managing trajectory data using RabbitMQ. It handles the connection setup, message publishing, and consumption for trajectory processing in the RL system.

##### Constructor Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `host`    | str    | The RabbitMQ server host address             |
| `port`    | int    | The RabbitMQ server port (default: 5672)     |

#### `TrajectoryQueueAgent.__init__(self, host: str, port: int = 5672) -> None`

Initializes the TrajectoryQueueAgent with the specified RabbitMQ host and port. This constructor establishes a connection to the RabbitMQ server, declares the necessary exchange and queue, and sets up the binding between them.

The following RabbitMQ components are automatically configured:
- Queue Name: `trajectory.queue`
- Exchange Name: `trajectory.exchange`
- Exchange Type: `direct`
- Routing Key: `trajectory.routing`
- Durability: `true` (both queue and exchange persist across server restarts)

#### `TrajectoryQueueAgent.push(self, trajectory: dict) -> None`

Pushes a trajectory dictionary to the RabbitMQ queue. The trajectory is serialized to JSON format before being published to the queue.

##### Parameters

| Parameter    | Type | Description                                   |
| ------------ | :--: | --------------------------------------------- |
| `trajectory` | dict | The trajectory data to be queued              |

##### Raises

- `ValueError`: If the trajectory parameter is not a dictionary.

#### `TrajectoryQueueAgent.pop(self) -> dict | None`

Pops a trajectory from the RabbitMQ queue using basic_get for immediate retrieval. If a message is available, it is acknowledged and returned as a dictionary. If the queue is empty, returns None.

##### Returns

| Type         | Description                                   |
| :----------: | --------------------------------------------- |
| `dict | None` | The popped trajectory data or None if queue is empty |

#### `TrajectoryQueueAgent.start_consuming(self, callback: callable) -> None`

Starts continuous consumption of messages from the RabbitMQ queue. This method blocks and processes incoming messages using the provided callback function.

##### Parameters

| Parameter  | Type     | Description                                   |
| ---------- | :------: | --------------------------------------------- |
| `callback` | callable | Function to process each received trajectory  |

The callback function should accept a single parameter of type `dict` containing the trajectory data.


#### `TrajectoryQueueAgent.close(self) -> None`

Closes the connection to the RabbitMQ server. This method should be called to properly clean up resources when the TrajectoryQueue is no longer needed.

### `Class service.utils.Deduplicator()`

The `Deduplicator` class is responsible for identifying and removing duplicate trajectories from the queue. It ensures that only unique trajectories are processed, improving the efficiency of the RL system.

#### `Deduplicator.__init__(self) -> None`

Initializes the Deduplicator with an empty set to keep track of seen trajectories.

#### `Deduplicator.is_duplicate(self, trajectory: dict) -> bool`

Checks if the given trajectory is a duplicate. If it is unique, it adds it to the set of seen trajectories.

##### Parameters

| Parameter    | Type | Description                                   |
| ------------ | :--: | --------------------------------------------- |
| `trajectory` | dict | The trajectory data to check for duplicates   |

##### Returns

| Type    | Description                                   |
| :------: | --------------------------------------------- |
| `bool` | True if the trajectory is a duplicate, False otherwise |

### `Class GateToken(literals, *, type: str)`

The `GateToken` class represents a token that indicates an operation (adding or deleting a gate) in the formula game. It contains information about the type of operation and the literals involved.

##### Constructor Parameters

| Parameter       | Type   | Description                                   |
| --------------- | :-----: | --------------------------------------------- |
| `literals`      | Literals   | A list of literals involved in the operation   |
| `type`    | str    | The type of operation, either "ADD" or "DEL" or "EOS" |


#### `GateToken.dim_token(num_vars: int) -> int`

Returns `2 * num_vars + 3`, the dimension of the token tensor based on the number of variables in the formula, where `num_vars` is the number of variables in the formula. The first `num_vars` elements represent the literals, the next `num_vars` elements represent the negated literals, and the last three elements represent the token type.

#### `GateToken.to_tensor(self) -> torch.Tensor`

Converts the `GateToken` instance to a PyTorch tensor representation. The tensor will have a shape of `(dim_token,)`. 

#### `GateToken.from_tensor(torch.Tensor) -> GateToken`

Converts a PyTorch tensor back to a `GateToken` instance. The tensor should have a shape of `(dim_token,)`. This is a class method. 


### Formula Game

#### `Class FormulaGame(init_formula_def: list[GateToken], **config)`

The `FormulaGame` class implements the RL environment that simulates the process of adding or deleting gates in a normal form formula. It calculates the average-case deterministic query complexity, which is the optimization target.

##### Constructor Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `init_formula_def` | list[GateToken] | The formula's definition to be manipulated in the game |
| `config`  | dict   | Configuration parameters for the game         |

#### `FormulaGame.reset(self) -> None`

Resets the internal variables of the formula game. This method is called at the beginning of each episode to prepare the environment for a new game.

#### `FormulaGame.step(self, token: GateToken) -> float`

Simulates a step in the formula game by applying the given token (which indicates adding or deleting a gate) to the formula. It returns the reward for this step, which is based on the average-case deterministic query complexity of the resulting formula.

##### Parameters

| Parameter | Type     | Description                                   |
| --------- | :-------: | --------------------------------------------- |
| `token`   | GateToken | The token representing the gate operation     |

##### Returns

| Type    | Description                                   |
| :------: | --------------------------------------------- |
| `float` | The reward received after applying the token, based on the average-case deterministic query complexity of the formula. |


### `Class EnvironmentAgent(num_env: int, num_vars: int, width: int, size: int, device: torch.device = None, **config)`

The `EnvironmentAgent` class manages multiple environments (formula games) and transforms data into Tensor/TensorDict format. It is responsible for replacing arms/environments using the arm filter, resetting formula games, and executing steps in the formula games. During initialization, it will create `num_env` games of formulas with the given number of variables and width and call `replace_arms()` to initialize the arms.

##### Constructor Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `num_env` | int    | The number of environments to manage          |
| `num_vars` | int    | The number of variables in the formula        |
| `width`   | int    | The width of the formula                      |
| `size`    | int    | The maximum size of the formula      |
| `device`  | torch.device | The device to run the agent on (default: CPU) |
| `config`  | dict   | Configuration parameters for the agent        |

#### `EnvironmentAgent.replace_arms(self) -> None`

Replaces all arms/environments using the arm filter. This method is called to update the set of available arms based on the latest trajectories and evolution graph.

#### `EnvironmentAgent.reset(self) -> tuple[torch.Tensor, torch.Tensor]`

Resets the formula games and saves trajectories to the trajectory queue unless the agent has not called `step()` method before. This method prepares the environment for a new episode by resetting the state of all formula games, and returns the gates and lengths of the initial formulas in Tensor format.

##### Returns

| Name   | Type          | Description                                   |
| :-----: | :------------: | --------------------------------------------- |
| `gates` | torch.Tensor  | A tensor representing the gates of the initial formulas, with shape `(num_env, size, dim_token)` |
| `lengths` | torch.Tensor | A tensor representing the lengths of the initial formulas, with shape `(num_env,)` |


#### `EnvironmentAgent.step(self, tokens: torch.Tensor) -> torch.Tensor`

Executes a step of the formula games by applying the tensor of the given token. It returns the reward for this step, which is based on the average-case deterministic query complexity of the resulting formula.

##### Parameters

| Parameter | Type          | Description                                   |
| --------- | :------------: | --------------------------------------------- |
| `tokens`  | torch.Tensor  | A tensor representing the operations to be applied to the formula games, with shape `(num_env, dim_token)` |

##### Returns

| Type    | Description                                   |
| :------: | --------------------------------------------- |
| `torch.Tensor` | A tensor representing the rewards for this step, based on the average-case deterministic query complexity of the resulting formula. |

---

### Trajectory Queue

Here are the methods that wrapper the RabbitMQ interface for the Trajectory Queue microservice. These methods allow for pushing and popping serialized trajectory data for downstream processing.

#### Class `TrajectoryQueue(host: str, port: int, **config)`

##### Constructor Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `host`  | str    | The IP address of the RabbitMQ server |
| `port`  | int    | The port number of the RabbitMQ server        |
| `config`  | dict   | Configuration parameters for the queue        |

##### Exceptions

| Exception | Description                                   |
| :--------: | --------------------------------------------- |
| `pika.exceptions.AMQPConnectionError` | Raised when the connection to the RabbitMQ server fails |

#### `TrajectoryQueue.push(self, trajectory: dict) -> None`

Pushes a serialized trajectory to the RabbitMQ queue for downstream processing.

##### Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `trajectory` | dict | The serialized trajectory data to push to the queue |

#### `TrajectoryQueue.pop(self) -> dict | None`

Pops a serialized trajectory from the RabbitMQ queue for processing.

##### Returns

| Type    | Description                                   |
| :------: | --------------------------------------------- |
| `dict`  | The serialized trajectory data popped from the queue. Returns `None` if the queue is empty. |

#### `TrajectoryQueue.close(self) -> None`

Closes the connection to the RabbitMQ server. This method should be called when the queue is no longer needed to release resources.

#### `TrajectoryQueue.start_consuming(self, callback: callable) -> None`

Starts consuming messages from the RabbitMQ queue.

##### Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `callback` | callable | A function to call with each message. A message in the TrajectoryQueue message schema would be passed to this function. |

### Arm Filter

The internal components of the Arm Filter microservice are responsible for processing trajectories, managing the evolution graph, and ranking arms (formulas). These components work together to maintain the evolution graph of formulas and implement policies for arm selection.

#### `Class TrajectoryProcessor(**config)`


The `TrajectoryProcessor` class processes trajectories and updates the evolution graph. It is responsible for managing the data flow between the trajectory queue and the evolution graph.

##### Constructor Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `config`  | dict   | Configuration parameters for the processor    |

#### `TrajectoryProcessor.check_if_duplicate(self, formula: NormalFormFormula) -> bool`

Checks if a given formula is isomorphic to any existing formula in the warehouse. This method uses the Weisfeiler-Lehman hash to determine if the formula is a duplicate.

##### Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `formula` | NormalFormFormula | The formula to check for isomorphism         |

##### Returns

| Type    | Description                                   |
| :------: | --------------------------------------------- |
| `bool`  | `True` if the formula is isomorphic to any existing formula


#### `TrajectoryProcessor.process_trajectory(self, data: dict) -> None`

Processes a single trajectory and updates the evolution graph accordingly. This method is called when a new trajectory is received from the trajectory queue and would try to break down the trajectory into smaller parts if necessary. The result is then saved to the warehouse.

##### Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `data`    | dict   | The trajectory data to process in the message schema (JSON) defined in Trajectory Queue.                  |

#### `Class EvolutionGraphManager(**config)`

The `EvolutionGraphManager` class manages the evolution graph and its updates. It is responsible for keeping the graph's size manageable using graph contraction techniques, which adds skipping edges to the graph. 

##### Constructor Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `config`  | dict   | Configuration parameters for the manager      |

#### `EvolutionGraphManager.check(self) -> bool`

Checks if the evolution graph satisfies the size constraints defined in the configuration. If the graph is too large, it will trigger a contraction process.

##### Returns

| Type    | Description                                   |
| :------: | --------------------------------------------- |
| `bool`  | `True` if the graph is within size constraints, `False` otherwise. |

#### `EvolutionGraphManager.contract_graph(self) -> None`

Contracts the evolution graph by merging nodes and edges based on the provided data. This method is called when the graph exceeds the size constraints. The result is then saved to the warehouse.

#### `Class ArmRanker(**config)`

The `ArmRanker` class ranks the arms (formulas) based on their performance and potential. It uses the evolution graph and trajectories to determine the best arms.

##### Constructor Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `config`  | dict   | Configuration parameters for the ranker       |

#### `ArmRanker.rank_arms(self, arms: list[int]) -> list[int]`

Ranks the provided arms based on their performance and potential. This method uses the evolution graph and trajectories to determine the best arms.

##### Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `arms`    | list[int] | A list of indices representing the arms to be ranked |

##### Returns

| Type    | Description                                   |
| :------: | --------------------------------------------- |
| `list[int]`  | A list of indices representing the ranked arms, sorted by their performance and potential. |
