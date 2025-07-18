from tensordict.nn import TensorDictModule
from tensordict import TensorDict
import torch
import torch.nn as nn
from torch import nn
from torchrl.modules import ConvNet, MLP
from torchrl.modules.models.utils import SquashDims
from tensordict.nn import TensorDictSequential

data = TensorDict({"key1": torch.randn(10, 3)}, batch_size=[10])
module = nn.Linear(3, 4)
td_module = TensorDictModule(module, in_keys=["key1"], out_keys=["key2"])
td_module(data)
print(data)

#########################

net = MLP(num_cells=[32, 64], out_features=4, activation_class=nn.ELU)
print(net)
print(net(torch.randn(10, 3)).shape)

#########################

backbone_module = nn.Linear(5, 3)
backbone = TensorDictModule(
    backbone_module, in_keys=["observation"], out_keys=["hidden"]
)
actor_module = nn.Linear(3, 4)
actor = TensorDictModule(actor_module, in_keys=["hidden"], out_keys=["action"])
value_module = MLP(out_features=1, num_cells=[4, 5])
value = TensorDictModule(value_module, in_keys=["hidden", "action"], out_keys=["value"])

sequence = TensorDictSequential(backbone, actor, value)
print(sequence)

print(sequence.in_keys, sequence.out_keys)

data = TensorDict(
    {"observation": torch.randn(3, 5)},
    [3],
)
backbone(data)
# Note: data is modified in place, so the output of backbone is now in data["hidden"]
actor(data)
value(data)

data = TensorDict(
    {"observation": torch.randn(3, 5)},
    [3],
)
sequence(data)
print(data)