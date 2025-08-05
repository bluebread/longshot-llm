"""

## `Class lsutils.GateToken(literals, *, type: str)`

The `GateToken` class represents a token that indicates an operation (adding or deleting a gate) in the formula game. It contains information about the type of operation and the literals involved.

### Constructor Parameters

| Parameter       | Type   | Description                                   |
| --------------- | :-----: | --------------------------------------------- |
| `literals`      | Literals   | A list of literals involved in the operation   |
| `type`    | str    | The type of operation, either "ADD" or "DEL" or "EOS" |


### `GateToken.dim_token(num_vars: int) -> int`

Returns `2 * num_vars + 3`, the dimension of the token tensor based on the number of variables in the formula, where `num_vars` is the number of variables in the formula. The first `num_vars` elements represent the literals, the next `num_vars` elements represent the negated literals, and the last three elements represent the token type.

### `GateToken.to_tensor(self) -> torch.Tensor`

Converts the `GateToken` instance to a PyTorch tensor representation. The tensor will have a shape of `(dim_token,)`. 

### `GateToken.from_tensor(torch.Tensor) -> GateToken`

Converts a PyTorch tensor back to a `GateToken` instance. The tensor should have a shape of `(dim_token,)`. This is a class method. 

"""