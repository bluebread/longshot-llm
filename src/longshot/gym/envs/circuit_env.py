from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AvgQ_D2_FormulaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}