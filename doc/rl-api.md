# RL-API Documentation

This document outlines the structure and content of the API documentation for the RL part of this project. It serves as a guide for developers to understand how to use the API effectively.

## Overview

### Local Modules

1. **Formula Game**
    - Implements the RL environment that simulates the process of adding/deleting gates in a normal formed formula. 
    - Calcultes *average-case deterministic query complexity*, the optimization target.
    - Main functions:
        1. `reset()`: Resets internal variables. 
        2. `step()`: Given the passed token (which indicates adding or deleting a gate), simulates a step and returns the reward. 
2. **Environment Agent**
    - Manages environments. 
    - Transforms data in Tensor/TensorDict format. 
    - Main functions:
        1. `replace_arms()`: Replaces all arms/environments using arm filter. 
        2. `reset()`: Resets formula games and saves trajectories to the trajectory queue (except the first time calling `reset()`). 
        3. `step()`: Executes a step of formula games. 
3. **Trainer**
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
        - `PUT /evolution_graph/edge`: Updates an existing edge in the evolution graph.
        - `DELETE /evolution_graph/edge`: Deletes an edge from the evolution graph.
    - High-level API:
        - `GET /formula/is_duplicate`: Checks if a formula is isomorphic to any existing formula.
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
        1. *Arm Filter Gateway*: Provides a public interface for the arm filter service.
        2. *Trajectory Processor*: Processes trajectories and updates the evolution graph.
        3. *Arm Deduplicator*: Checks if a formula is isomorphic to any existing formula using the isomorphism hash table.
        4. *Graph Change Detector*: Detects changes in the evolution graph and calls the evolution graph manager and the arm ranker to update the graph and the arm ranking.
        5. *Evolution Graph Manager*: Manages the evolution graph and its updates.
        6. *Arm Ranker*: Ranks the arms (formulas) based on their performance and potential.
    - Main API:
        - `GET /topk_arms`: Return the current best top-K arms.


## Database Schema

### Formula Table

| Column Name       | Type        | Description                          |
|:------------------|:-----------:|:--------------------------------------|
| id                | UUID        | Primary key                                 |
| base_formula_id   | UUID        | Parent formula ID                             |
| trajectory_id     | UUID        | Associated trajectory table ID (can be NULL)            |
| avgQ        | float       | Average-case deterministic query complexity                             |
| wl-hash              | char(32)      | Weisfeiler-Lehman hash value              |
| num_vars          | int         | Number of variables                               |
| width             | int         | Formula width                               |
| size              | int         | Formula size (number of nodes)                     |
| timestamp         | datetime    | Insertion time                               |
| node_id         | UUID        | Node ID in the evolution graph (can be NULL) |
| full_trajectory_id | UUID        | Full trajectory table ID (default NULL)                     |


### Trajectory Table


| Column Name      | Type     | Description                                      |
|------------------|:----------:|--------------------------------------------------|
| order            | int      | The order of this token in the entire trace (0-based)             |
| token_type       | bool   | The type of the token, 0 for ADD, 1 for DELETE    |
| token_literals   | int   | The binary representation for the literals of the token |
| reward           | float    | The immediate reward received after this step                      |


### Isomorphism Hash Table

- Key (*string*): WL hash value of a formula.
- Value (*List[UUID]*): the indices of probably isomorphic formulas with the same WL hash value.

### Evolution Graph

Each graph is labeled with `N<num_vars>W<width>`, where `num_vars` is the number of variables and `width` is the width of the formula.

#### Node Entity

| Attribute   | Type   | Description                   |
| ---------------- | :----: | ----------------------------- |
| formula\_id      | UUID   | The node's ID, corresponding to the primary key of the FormulaTable    |
| avgQ         | float  | The average-case deterministic query complexity of the formula represented by this node |
| visited\_counter | int    | The number of times this node has been touched by a trajectory                   |
| inactive         | bool   | Whether this node is inactive                    |
| in_degree        | int    | The in-degree of this node in the evolution graph  |
| out_degree       | int    | The out-degree of this node in the evolution graph  |

#### Edge Entity

| Attribute   | Type   | Description                   |
| ---------------- | :----: | ----------------------------- |
| base_formula_id | UUID   | The ID of the base formula, corresponding to the primary key of FormulaTable |
| new_formula_id  | UUID   | The ID of the new formula, corresponding to the primary key of FormulaTable    |
| distance         | int    | The Hamming distance between the base formula and the new formula (e.g., adding or removing a gate) |


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


### Warehouse API Reference

Below are the RESTful API endpoints provided by the Warehouse microservice. Each endpoint is described with its purpose, request/response formats, and expected status codes.

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
        "wl-hash": "abcd1234...",
        "num_vars": 3,
        "width": 2,
        "size": 5,
        "timestamp": "2025-07-21T12:00:00Z",
        "node_id": "n789",
        "full_trajectory_id": "t999"
    }
    ```
- **Status Codes:**  
    - `200 OK`, `404 Not Found`


#### `POST /formula/info`
Add a new formula entry to the formula table.

- **Request Body:**  
    ```json
    {
        "base_formula_id": "f122",
        "trajectory_id": "t456",
        "avgQ": 2.5,
        "wl-hash": "abcd1234...",
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
    - `201 Created`, `400 Bad Request`


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
    - `200 OK`, `400 Bad Request`, `404 Not Found`


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
    - `wl-hash` (string, required): Weisfeiler-Lehman hash.

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
        "wl-hash": "abcd1234...",
        "formula_id": "f125"
    }
    ```
- **Response:**  
    - Success message.
- **Status Codes:**  
    - `201 Created`, `400 Bad Request`


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
    - `201 Created`, `400 Bad Request`


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
    - `200 OK`, `400 Bad Request`, `404 Not Found`


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
Add a new node to the evolution graph.

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
    - `201 Created`, `400 Bad Request`


#### `PUT /evolution_graph/node`
Update an existing node.

- **Request Body:**  
    ```json
    {
        "node_id": "n789",
        "visited_counter": 11,
        "inactive": true
    }
    ```
- **Response:**  
    - Success message.
- **Status Codes:**  
    - `200 OK`, `400 Bad Request`, `404 Not Found`


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
        "new_formula_id": "f124",
        "distance": 1
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
        "new_formula_id": "f124",
        "distance": 1
    }
    ```
- **Response:**  
    ```json
    {
        "edge_id": "e456"
    }
    ```
- **Status Codes:**  
    - `201 Created`, `400 Bad Request`


#### `PUT /evolution_graph/edge`
Update an existing edge.

- **Request Body:**  
    ```json
    {
        "edge_id": "e456",
        "distance": 2
    }
    ```
- **Response:**  
    - Success message.
- **Status Codes:**  
    - `200 OK`, `400 Bad Request`, `404 Not Found`


#### `DELETE /evolution_graph/edge`
Delete an edge.

- **Query Parameters:**  
    - `edge_id` (string, required): Edge UUID.

- **Response:**  
    - Success message.
- **Status Codes:**  
    - `200 OK`, `404 Not Found`


#### `GET /formula/is_duplicate`
Check if a formula is isomorphic to any existing formula.

- **Query Parameters:**  
    - `wl-hash` (string, required): Weisfeiler-Lehman hash.

- **Response:**  
    ```json
    {
        "is_duplicate": true,
        "duplicate_ids": ["f123", "f124"]
    }
    ```
- **Status Codes:**  
    - `200 OK`, `400 Bad Request`


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
    - `200 OK`, `400 Bad Request`


#### `POST /formula/add`
Add a new formula to the warehouse, updating the isomorphism hash table and evolution graph.

- **Request Body:**  
    ```json
    {
        "base_formula_id": "f122",
        "trajectory_id": "t456",
        "avgQ": 2.5,
        "wl-hash": "abcd1234...",
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
    - `201 Created`, `400 Bad Request`


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
    - `201 Created`, `400 Bad Request`


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
    - `200 OK`, `400 Bad Request`, `404 Not Found`





---

### Arm Filter


The Arm Filter is a microservice responsible for filtering and selecting the best arms (formulas) based on the trajectories and the evolution graph. It maintains the evolution graph of formulas and implements policies for arm selection, such as the Upper Confidence Bound (UCB) algorithm.

#### `GET /topk_arms`

##### Description

Returns the current best top-K arms based on the latest trajectories and evolution graph.

##### Endpoint
```
GET /topk_arms
```

##### Query Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| num_vars | int | The number of variables in the formula  |
| width    | int | The width of the formula           |
| k       | int | The number of top arms to return  

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
| `formula_id`   | string | The unique ID of the formula                   |
| `definition`   | array  | The definition of the formula, represented as a list of lists of literals |


##### Status Codes

* `200 OK`: Successfully returned top-K arms
* `400 Bad Request`: Missing required parameters or invalid values
* `500 Internal Server Error`: Failed to retrieve data



## Local Modules

### Formula Game

---

### Environment Agent

---

### Trainer

---

### Arm Filter

#### Arm Filter Gateway

#### Arm Deduplicator

#### Trajectory Processor

#### Graph Change Detector

#### Evolution Graph Manager

#### Arm Ranker