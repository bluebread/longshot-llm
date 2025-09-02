import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.objectives.value import GAE
import torch.nn as nn

# Commented out the value network as it is not used in this example.
# # value_net = TensorDictModule(
# #     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
# # )
# Reference: https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.value.GAE.html?highlight=gae#torchrl.objectives.value.GAE
# Create a GAE module without a value network.

module = GAE(
    gamma=0.98,
    lmbda=0.95,
    value_network=None,
    differentiable=False,
)
value = torch.randn(1, 10, 1)
next_value = torch.randn(1, 10, 1)
reward = torch.randn(1, 10, 1)
done = torch.zeros(1, 10, 1, dtype=torch.bool)
terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
tensordict = TensorDict(
    {
        # "obs": obs, # This is used in the original code, but not needed for GAE without a value network
        "state_value": value, # This is NOT in the original code, but needed for GAE
        "next": {
            # "obs": next_obs, # This is used in the original code, but not needed for GAE without a value network
            "done": done, 
            "reward": reward, 
            "terminated": terminated,
            "state_value": next_value, # This is NOT in the original code, but needed for GAE
        }, 
    }, 
    [1, 10]
)
_ = module(tensordict)
assert "advantage" in tensordict.keys()
# print('value:', value)
# print('next_value:', next_value)
# print('reward:', reward)
print('Advantage (GAE module):\n', tensordict['advantage'])

# backward pass
gamma, lam = 0.98, 0.95
value = value.squeeze(0)
value = value.squeeze(-1)
next_value = next_value.squeeze(0)
next_value = next_value.squeeze(-1)
reward = reward.squeeze(0)
reward = reward.squeeze(-1)
# Calculate GAE manually
deltas = reward + gamma * next_value - value # THIS IS THE REALLY FORMULA OF GAE!!!
advantages = torch.zeros(10)
gae = 0.0
for t in reversed(range(10)):
    gae = deltas[t] + gamma * lam * gae
    advantages[t] = gae

print("Advantages (manual):\n", advantages)