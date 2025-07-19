# RL-API Documentation

This document outlines the structure and content of the API documentation for the RL part of this project. It serves as a guide for developers to understand how to use the API effectively.

## Overview

### Main Program

1. **Trainer**
    - Trains a RL policy that learns how to build a CNF/DNF formula with the largest *average-case deterministic query complexity* of a formula, which is the optimization target.
    - Retrieves dataset (trajectories) from the environment (wrapper). 
    - Includes but not limited to the following modules:
        1. Replay Buffer (should be able to handle both Markovian and non-Markovian sequences)
        2. General Advantage Estimation（GAE, optional）
        3. Optimizer
        4. Scheduler
        5. Loss Function
        6. Policy
        7. Critic
2. **Environment Wrapper**
    - Manages environments. 
    - Transforms data in Tensor/TensorDict format. 
    - Main functions:
        1. `replace_arms()`: Replaces all arms/environments using arm selector. 
        2. `reset()`: Resets formula games and saves trajectories to the trajectory queue (except the first time calling `reset()`). 
        3. `step()`: Executes a step of formula games. 
3. **Formula Game**
    - Implements the RL environment that simulates the process of adding/deleting gates from a normal formed formula. 
    - Calcultes the average-case deterministic query complexity.
    - Main functions:
        1. `reset()`: Resets internal variables. 
        2. `step()`: Given the passed token (which indicates adding or deleting a gate), simulates a step and returns the reward. 
4. **Arms Selector**
    - Selects the best or most potential arms, which are used to intialize the pool of environments.
    - Implements policies for arm selection (e.g. the UCB algorithm).
    - Does **NOT** maintain arms or keep them in memory. 
    - Main functions:
        - `select_arms()`: Downloads arms from the warehouse and selects the best arms based on the current state of the environment. 

### Microservices

1. **Trajectory Queue**
    - Handles the queuing of trajectories for processing
2. **Warehose**
    - Manages the storage and retrieval of data. It hides the complexity of data management and provides a simple interface for data access.
    - Contains the following tables:
        1. *Isomorphism Hash Table*: Maps a WL hash to IDs of all isomorphic formuals in *Formula Table*. 
        2. *Formula Table*: Each entry records (ID, timestamp, BaseFormulaID, TrajectoryID, complexity, hash, #vars, width, size). 
        3. *Trajectory Tables*: Each tables contains entries of (order, token_type, token_literals, reward). 
        4. *Evolution Graph Database*:
            - graph label: (#vars, width)
            - nodes: (FormulaTableID, visited_counter)
            - edges: (BaseFormulaID, NewFormulaID, distance)
3. **Gardener**
    - Builds/Filters formulas from trajectories.
    - Maintains evolution graphs. 
    - Checks if a formula is duplicate using the isomorphism has table. 
    - Increments the counter if a formula is touched in a trajectory. 
    - Could have several policy implementations.

## Local Modules

## Database Schema

### Formula Table

| Column Name       | Type        | Description                          |
|:------------------|:-----------:|:--------------------------------------|
| id                | UUID        | Primary key                                 |
| base_formula_id   | UUID        | Parent formula ID                             |
| trajectory_id     | UUID        | Associated trajectory ID (can be NULL)            |
| avgQ        | float       | Average-case deterministic query complexity                             |
| wl-hash              | string      | Weisfeiler-Lehman hash value              |
| num_vars          | int         | Number of variables                               |
| width             | int         | Formula width                               |
| size              | int         | Formula size (number of nodes)                     |
| timestamp         | datetime    | Insertion time                               |


### Trajectory Table


| Column Name      | Type     | Description                                      |
|------------------|:----------:|--------------------------------------------------|
| order            | int      | The order of this token in the entire trace (0-based)             |
| token_type       | string   | The type of the token, such as “ADD”, ‘DELETE’, “EOS”     |
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
| visited\_counter | int    | The number of times this node has been touched by a trajectory                   |

#### Edge Entity

| Attribute   | Type   | Description                   |
| ---------------- | :----: | ----------------------------- |
| base_formula_id | UUID   | The ID of the base formula, corresponding to the primary key of FormulaTable |
| new_formula_id  | UUID   | The ID of the new formula, corresponding to the primary key of FormulaTable    |
| distance         | int    | The distance between the base formula and the new formula (e.g., adding or removing a gate) |


## Microservice


### Trajectory Queue

The Trajectory Queue is a microservice responsible for temporarily storing trajectories produced by the RL environment. It provides endpoints to push new trajectories into the queue and pop all trajectories for processing. 

#### Base URL

```
http://<host>:<port>/trajectory-queue
```

---

#### `POST /push`

##### Description

Push one or more trajectories into the queue.

##### Endpoint

```
POST /push
```

##### Request Body (JSON)

```json
{
  "trajectories": [
    {
      "formula_id": "f123",
      "steps": [
        {
          "order": 0,
          "token_type": "ADD",
          "token_literals": ["x1", "x2"],
          "reward": 0.1
        },
        {
          "order": 1,
          "token_type": "DEL",
          "token_literals": ["x2"],
          "reward": -0.05
        }
      ]
    },
    {
      "formula_id": "f124",
      "steps": [
        {
          "order": 0,
          "token_type": "ADD",
          "token_literals": ["x3", "x4"],
          "reward": 0.2
        }
      ]
    }
  ]
}
```

##### Request Field Descriptions

| Field                 | Type   | Description                                                 |
| --------------------- | :------: | ----------------------------------------------------------- |
| `formula_id`          | string | The unique ID of the formula that the trajectory applies to |
| `steps`               | array  | List of step objects in the trajectory                      |
| `step.order`          | int    | The position of the action in the sequence                  |
| `step.token_type`     | string | One of `"ADD"`, `"DEL"`, `"EOS"`                                     |                                     |
| `step.token_literals` | int  | The binary representation for literals involved in the operation                          |
| `step.reward`         | float  | The reward received at this step                            |

##### Response (Success)

```json
{
  "status": "success",
  "num_received": 2
}
```

##### Response (Error)

```json
{
  "status": "error",
  "message": "Invalid request: missing 'trajectories' field"
}
```

##### Status Codes

* `200 OK`: Successful push
* `400 Bad Request`: Malformed request
* `500 Internal Server Error`: Server failed to store trajectories

---

#### `GET /pop`

##### Description

Pop all trajectories from the queue. This is a destructive operation — once called, the queue will be emptied.

##### Endpoint

```
GET /pop
```

##### Response (Success)

```json
{
  "trajectories": [
    {
      "formula_id": "f123",
      "steps": [
        {
          "order": 0,
          "token_type": "ADD",
          "token_literals": ["x1", "x2"],
          "reward": 0.1
        },
        {
          "order": 1,
          "token_type": "DEL",
          "token_literals": ["x2"],
          "reward": -0.05
        }
      ]
    },
    ...
  ]
}
```

##### Request Field Descriptions

| Field                 | Type   | Description                                                 |
| --------------------- | :------: | ----------------------------------------------------------- |
| `formula_id`          | string | The unique ID of the formula that the trajectory applies to |
| `steps`               | array  | List of step objects in the trajectory                      |
| `step.order`          | int    | The position of the action in the sequence                  |
| `step.token_type`     | string | One of `"ADD"`, `"DEL"`, `"EOS"`                                     |
| `step.token_literals` | int  | The binary representation for literals involved in the operation                          |
| `step.reward`         | float  | The reward received at this step                            |



##### Response (Empty Queue)

```json
{
  "trajectories": []
}
```

##### Status Codes

* `200 OK`: Successfully returned trajectories
* `204 No Content`: Queue is empty (optional, or just return `[]`)
* `500 Internal Server Error`: Failed to retrieve data

### Warehouse

### Gardener