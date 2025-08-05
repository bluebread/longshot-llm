"""
## Local Modules: Formula Game

### `Class FormulaGame(init_formula_def: list[GateToken], **config)`

The `FormulaGame` class implements the RL environment that simulates the process of adding or deleting gates in a normal form formula. It calculates the average-case deterministic query complexity, which is the optimization target.

#### Constructor Parameters

| Parameter | Type   | Description                                   |
| --------- | :-----: | --------------------------------------------- |
| `init_formula_def` | list[GateToken] | The formula's definition to be manipulated in the game |
| `config`  | dict   | Configuration parameters for the game         |

#### `FormulaGame.reset(self) -> None`

Resets the internal variables of the formula game. This method is called at the beginning of each episode to prepare the environment for a new game.

#### `FormulaGame.step(self, token: GateToken) -> float`

Simulates a step in the formula game by applying the given token (which indicates adding or deleting a gate) to the formula. It returns the reward for this step, which is based on the average-case deterministic query complexity of the resulting formula.

#### Parameters

| Parameter | Type     | Description                                   |
| --------- | :-------: | --------------------------------------------- |
| `token`   | GateToken | The token representing the gate operation     |

#### Returns

| Type    | Description                                   |
| :------: | --------------------------------------------- |
| `float` | The reward received after applying the token, based on the average-case deterministic query complexity of the formula. |

"""