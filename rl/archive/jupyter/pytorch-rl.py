from torchrl.envs import GymEnv
from torchrl.envs import step_mdp
import gymnasium as gym
import pytest
import numpy as np

import longshot
import longshot.gym
from longshot.circuit import FormulaType, Literals

env = GymEnv("longshot/avgQ-d2-formula-v0", n=3, mono=True, ftype=FormulaType.Disjunctive)

reset = env.reset()

# reset_with_action = env.rand_action(reset)
# stepped_data = env.step(reset_with_action)
# data = step_mdp(stepped_data)
# print(data)

rollout = env.rollout(max_steps=10)
print(rollout)