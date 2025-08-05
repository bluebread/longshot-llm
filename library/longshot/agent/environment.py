"""
## Local Modules: Environment Agent

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
"""