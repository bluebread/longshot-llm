from gymnasium.envs.registration import register
from ray.tune.registry import register_env

from .envs import AvgQ_D2_FormulaEnv

register(
    id="longshot/avgQ-d2-formula-v0",
    entry_point="longshot.gym.envs:AvgQ_D2_FormulaEnv",
)
register_env("longshot/avgQ-d2-formula-v0", lambda config: AvgQ_D2_FormulaEnv(**config))