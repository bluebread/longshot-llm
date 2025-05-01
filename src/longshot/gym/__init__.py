from gymnasium.envs.registration import register

from .envs import AvgQ_D2_FormulaEnv
from .envs import FlattenSequence, LambdaMixedReward

register(
    id="longshot/avgQ-d2-formula",
    entry_point="longshot.gym.envs:AvgQ_D2_FormulaEnv",
)