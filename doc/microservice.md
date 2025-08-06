# Microservice Documentation

This document outlines the structure and content of the API documentation for the microservices of this project. It serves as a guide for developers to understand how to use the API effectively.

## Overview

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
            - nodes: (FormulaTableID, visited_counter, ...)
            - edges: (BaseFormulaID, NewFormulaID)
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
    - High-level API:
        - `GET /formula/definition`: Retrieves the full definition of a formula by its ID.
        - `POST /evolution_graph/path`: Adds a new path to the evolution graph.
        - `GET /evolution_graph/download_nodes`: Retrieves the evolution subgraph of nodes satisfying the given conditions.

3. **Trajectory Processor**
    - Processes incoming trajectories and updates the warehouse with new data.
    - Extracts relevant information from trajectories and stores it in the appropriate tables.

4. **Arm Ranker**
    - Filters and selects the best arms (formulas) based on the trajectories and the evolution graph.
    - Maintains the evolution graph of formulas.
    - Implements policies for arm selection (e.g., UCB algorithm).
    - Main API:
        - `GET /topk_arms`: Return the current best top-K arms.

5. **Trainer**
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

        
## Database Schema

Because MongoDB does not have a native UUID type, we use UUIDs as strings in the database. 

### Formula Table (MongoDB)

In the database, the formula table is labeled as `FormulaTable`. Each entry in the formula table represents a formula and contains the following columns:

| Column Name       | Type        | Description                          |
|:------------------|:-----------:|:--------------------------------------|
| id                | UUID        | Primary key                                 |
| base_formula_id   | UUID         | Parent formula ID (can be NULL)                            |
| trajectory_id     | UUID         | Associated trajectory table ID (can be NULL)           |
| avgQ              | float       | Average-case deterministic query complexity                             |
| wl_hash           | string      | Weisfeiler-Lehman hash value              |
| num_vars          | int         | Number of variables                               |
| width             | int         | Formula width                               |
| size              | int         | Formula size (number of nodes)                     |
| timestamp         | datetime    | Insertion time                               |
| node_id         | UUID        | Node ID in the evolution graph  |

A entry represents a empty formula if `base_formula_id` and `trajectory_id` are both NULL.

<!-- ### Definition Cache Table (Redis)

- Key (*UUID*): the ID of the formula.
- Value (*List[UUID]*): the indices of trajectories that can be used to reconstruct the full definition of the formula. -->

### Trajectory Table (MongoDB)

Each trajectory is either a partial trajectory or the full definition of a formula, and its table ID must be recorded somewhere in the formula table.

| Column Name      | Type     | Description                                      |
|------------------|:----------:|--------------------------------------------------|
| id               | UUID     | Primary key                                      |
| base_formula_id | UUID     | The ID of the base formula, corresponding to the primary key of FormulaTable |
| timestamp        | datetime | The time when the trajectory was generated       |
| steps            | list     | List of steps in the trajectory, each step is a dictionary with the following fields: |
| step.token_type       | int   | The type of the token, 0 for ADD, 1 for DELETE    |
| step.token_literals   | int   | The binary representation for the literals of the token |
| step.reward           | float    | The immediate reward received after this step                      |


### Isomorphism Hash Table (Redis)

- Key (*string*): WL hash value of a formula.
- Value (*List[UUID]*): the indices of probably isomorphic formulas with the same WL hash value.

### Evolution Graph (Neo4j)

Each graph is labeled with `N<num_vars>W<width>`, where `num_vars` is the number of variables and `width` is the width of the formula.

#### Node Attributes

| Attribute   | Type   | Description                   |
| ---------------- | :----: | ----------------------------- |
| formula\_id      | UUID   | The node's ID, corresponding to the primary key of the FormulaTable (used as unique index)    |
| num\_vars      | int    | The number of variables in the formula represented by this node |
| width         | int    | The width of the formula represented by this node |
| size        | int    | The size of the formula represented by this node (number of nodes) |
| avgQ         | float  | The average-case deterministic query complexity of the formula represented by this node |
| visited\_counter | int    | The number of times this node has been touched by a trajectory                   |

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
  "size": 5,
  "timestamp": "2025-07-21T12:00:00Z",
  "trajectory": {
    "base_formula_id": "f123",
    "steps": [
      {
        "order": 0,
        "token_type": "ADD",
        "token_literals": 53456,
        "reward": 0.1,
        "avgQ": 2.5
      },
      {
        "order": 1,
        "token_type": "DEL",
        "token_literals": 358768,
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
| `base_size`               | int    | Size of the base formula (number of nodes)                       |
| `timestamp`           | string | ISO 8601 timestamp of when the trajectory was generated   |
| `trajectory`        | object | Contains the trajectory data                                |
| `base_formula_id`    | string | The unique ID of the base formula that the trajectory applies to |
| `steps`               | array  | List of step objects in the trajectory                      |
| `step.order`          | int    | The position of the action in the sequence                  |
| `step.token_type`     | string | One of `"ADD"`, `"DEL"`, `"EOS"`                                     |
| `step.token_literals` | int  | The binary representation for literals involved in the operation                          |
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

### Trajectory Processor


The `TrajectoryProcessor` class processes trajectories and updates the databases. It is responsible for managing the data flow between the trajectory queue and the warehouse. It is stateless and has no endpoints. 

#### Current Implementation

The current implementation processes the trajectory by extracting the formulas with top `traj_num_summits` avgQ values and breaking the trajectory into parts smaller than the number `granularity` defined in the configuration. If a formula is isomorphic to an existing formula in the warehouse, it will not be added to the evolution graph, and the corresponding trajectory piece would be merged into the next one. The final piece would be just discarded if the formula is duplicate.  

<!-- 
### `Class TrajectoryProcessor(**config)`


The `TrajectoryProcessor` class processes trajectories and updates the evolution graph. It is responsible for managing the data flow between the trajectory queue and the evolution graph.

#### Constructor Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `config`  | dict   | Configuration parameters for the processor    |


#### `TrajectoryProcessor.isomorphic_to(self, formula_graph: networkx.Graph, wl_hash: str | None = None) -> str | None`

Returns the ID of an existing formula in the warehouse that is isomorphic to the given formula graph. If no isomorphic formula is found, it returns `None`. 
This method uses the Weisfeiler-Lehman hash to determine if the formula is a duplicate. If the `wl_hash` is provided, it will be used to check for isomorphism; otherwise, the hash will be computed from the `formula_graph`.

##### Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `formula_graph` | networkx.Graph | The graph representation of the formula to check for isomorphism |
| `wl_hash` | str | The Weisfeiler-Lehman hash of the formula (optional) |

##### Returns

| Type    | Description                                   |
| :------: | --------------------------------------------- |
| str  | The ID of the isomorphic formula if found, otherwise `None` |


#### `TrajectoryProcessor.process_trajectory(self, trajectory: dict) -> dict[str, Any]`

Processes a single trajectory and updates the evolution graph accordingly. This method is called when a new trajectory is received from the trajectory queue and would try to break down the trajectory into smaller parts if necessary. The result is then saved to the warehouse and also returned as a list of new formulas' information.

##### Current Implementation

The current implementation processes the trajectory by extracting the formulas with top `traj_num_summits` avgQ values and breaking the trajectory into parts smaller than the number `granularity` defined in the configuration. If a formula is isomorphic to an existing formula in the warehouse, it will not be added to the evolution graph, and the corresponding trajectory piece would be merged into the next one. The final piece would be just discarded if the formula is duplicate.  

##### Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `data`    | dict   | The trajectory data to process in the message schema (JSON) defined in Trajectory Queue.                  |

##### Returns

```JSON
{
    "new_formulas": [
        {
            "id": "f123",
            "base_formula_id": "f122",
            "trajectory_id": "t456",
            "avgQ": 1.5,
            "num_vars": 3,
            "width": 2,
            "size": 5,
            "wl_hash": "abcd1234...",
        },
    ],
    "evo_path": [
        "f46", "f27", "f68", "f16"
    ]
}
```

| Attribute | Type    | Description                                   |
| :------: | :------: | --------------------------------------------- |
| new_formulas | `list[dict]`  | A list of dictionaries representing the new formulas' information, each containing the formula ID, base formula ID, trajectory ID, average-case deterministic query complexity, number of variables, width, size and wl-hash value.  |
| evo_path | `list[str]` | A list of formula IDs representing the evolution path of the formulas in the trajectory. This is used to track the evolution of formulas over time. | 
-->


---

### Warehouse

The Warehouse is a microservice that manages the storage and retrieval of data related to formulas, trajectories, and the evolution graph. It abstracts the complexity of data management and provides a simple interface for data access. 

#### Data Format

When request is not passed in a correct format, the server will return a `422 Unprocessable Entity` error.

For every token literals in the request body, it should be represented as a 64-bits integer, the first 32 bits for the positive literals and the last 32 bits for the negative literals.

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
        "avgQ": 1.5,
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
Add a new formula entry to the formula table. The fields `id` and `timestamp` are automatically generated by the server. The fields ``base_formula_id`` and ``trajectory_id`` can be NULL if the formula is an empty formula.


- **Request Body:**  
    ```json
    {
        "base_formula_id": "f122",
        "trajectory_id": "t456",
        "avgQ": 1.5,
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
        "avgQ": 1.5,
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
        "base_formula_id": "f123",
        "steps": [
            {
                "token_type": 0,
                "token_literals": 5,
                "reward": 1.0,
            }
        ]
    }
    ```
- **Status Codes:**  
    - `200 OK`, `404 Not Found`


#### `POST /trajectory`
Add a new trajectory. The fields `id` and `timestamp` are automatically generated by the server. The field `base_formula_id` can be NULL if the trajectory is not associated with any formula.


- **Request Body:**  
    ```json
    {
        "base_formula_id": "f123",
        "steps": [
            {
                "token_type": 0,
                "token_literals": 5,
                "reward": 1.0,
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
        "base_formula_id": "f123",
        "steps": [
            {
                "order": 3, // 0-based
                "token_type": 1,
                "token_literals": 3,
                "reward": 1.0
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


#### `GET /evolution_graph/node`
Retrieve a node in the evolution graph by its formula ID.

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
        "visited_counter": 10,
        "in_degree": 2,
        "out_degree": 3
    }
    ```
- **Status Codes:**  
    - `200 OK`, `404 Not Found`


#### `POST /evolution_graph/node`
Add a new node to the evolution graph. The visited counter is set to 0, and the inactive flag is set to false by default.

- **Request Body:**  
    ```json
    {
        "formula_id": "f123",
        "avgQ": 6.3,
        "num_vars": 3,
        "width": 2,
        "size": 5
    }
    ```
- **Response:**  
    - Success message.
- **Status Codes:**  
    - `201 Created`, `422 Unprocessable Entity`


#### `PUT /evolution_graph/node`
Update an existing node. The parameter ``inc_visited_counter`` is used to increment the visited counter by a specified amount and suppose to be greater than 0. 

- **Request Body:**  
    ```json
    {
        "formula_id": "f123",
        "inc_visited_counter": 11,
        "avgQ": 4.56
    }
    ```
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


##### TODO: Lazy Reconstruction and Caching
- Warehouse does not store the full definition of each formula initially.
- Instead, formulas are reconstructed recursively using `BaseFormulaID` and `TrajectoryID`.
- A lazy caching system is adopted: once a formula is reconstructed, its `FullTrajectoryID` may be stored for future faster access.
- Periodic or opportunistic checkpointing can be applied to avoid long chains of reconstruction.
- How frequently to reconstruct and cache is a design choice that can be tuned based on performance needs.


#### `POST /evolution_graph/path`

Add a new path to the evolution graph. The path is a list of formula IDs that represent the evolution path of the formulas in the trajectory.

- **Request Body:**  
    ```json
    {
        "path": [
            "f123",
            "f124",
            "f125",
        ]
    }
    ```
- **Response:**  
    - Success message.
- **Status Codes:**  
    - `201 Created`, `422 Unprocessable Entity`


#### `GET /evolution_graph/download_nodes`

Get the evolution subgraph of nodes satisfying the given conditions (e.g. `num_vars`, `width`, etc.). The subgraph is a list of nodes and edges that represent the evolution of formulas over time.


- **Query Parameters:**  
    - `num_vars` (int): The number of variables in the formula.
    - `width` (int): The width of the formula.
- **Response:**  
    ```json
    {
        "nodes": [
            {
                "formula_id": "f123",
                "avgQ": "QAkh+1RBF0Q=",
                "visited_counter": 10,
                "num_vars": 3,
                "width": 2,
                "in_degree": 2,
                "out_degree": 3
            },
        ]
    }
    ```
- **Status Codes:**
    - `200 OK`, `422 Unprocessable Entity`


---

### Arm Ranker


The Arm Ranker is a microservice responsible for filtering and selecting the best arms (formulas) based on the trajectories and the evolution graph. It maintains the evolution graph of formulas and implements policies for arm selection, such as the Upper Confidence Bound (UCB) algorithm.

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
        246, 123, 456
      ]
    },
    {
      "formula_id": "f124",
      "definition": [
        683, 234, 567
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
| `arm.definition`   | list[int]  | The definition of the formula, represented as a list of lists of literals (represented by integers) |


##### Status Codes

* `200 OK`: Successfully returned top-K arms
* `422 Unprocessable Entity`: Missing required parameters or invalid values

<!-- ## Local Modules: Arm Filter

The internal components of the Arm Filter microservice are responsible for processing trajectories, managing the evolution graph, and ranking arms (formulas). These components work together to maintain the evolution graph of formulas and implement policies for arm selection.

### Main Program Flow

The main program flow of the Arm Filter microservice involves the following components:

- Scheduled Tasks (TrajectoryProcessor): These tasks periodically process the trajectories from the queue, check/update the evolution graph, perform necessary updates (such as graph contraction), rank arms, and save the ranking to a file with timestamp. 
- API Endpoint (Arm Ranker): The /topk_arms endpoint allows users to retrieve the current best top-K arms based on the latest trajectories and evolution graph. The only thing that the API does is to read the latest ranking file and return the top-K arms' definition (obained from the warehouse). The ranking file is updated by the scheduled tasks.


### `Class ArmRanker(max_num_arms: int, **config)`

The `ArmRanker` class ranks the arms (formulas) based on their performance and potential. It uses the evolution graph and formulas' information to determine the best arms. This class is stateless and just provides a method to rank arms.

#### Constructor Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `max_num_arms` | int | The maximum number of arms to return in the ranking. |
| `config`  | dict   | Configuration parameters for the ranker       |

#### `ArmRanker.score(self, arm: dict, total_visited: int) -> float`

Scores a single arm (formula) based on its properties such as average-case deterministic query complexity, visited counter, in-degree, and out-degree. The score is calculated using a weighted formula that combines these properties.

##### Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `arm`     | dict   | A dictionary representing the arm (formula) to be scored, containing fields like `avgQ`, `visited_counter`, `in_degree`, and `out_degree`. |
| `total_visited` | int | The total number of visited counters across all arms, used to normalize the score. |

#### `ArmRanker.rank_arms(self, arms: list[dict], total_visited: int) -> list[tuple[int, float]]`

Ranks the provided arms based on their performance and potential. This method uses the evolution graph and trajectories to determine the best arms. In the case that the UCB algorithm is adopted, the current time step is the sum of visited counters of all arms.

##### Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `arms`    | list[dict] | A list of dict objects representing the arms to be ranked from EvolutionGraphManager.get_active_nodes method |
| `total_visited` | int | The total number of visited counters across all arms, used to normalize the scores. |

##### Returns

| Type    | Description                                   |
| :------: | --------------------------------------------- |
| list[tuple[int, float]] | A list of tuples, each containing the arm's ID and its score, sorted in descending order of score. The first element is the arm's ID, and the second element is the score. | 
-->