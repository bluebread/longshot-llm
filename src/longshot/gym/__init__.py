from gymnasium.envs.registration import register

from .circuit_env import AvgQ_D2_FormulaEnv
from .wrapper import (
    FlattenSequence, 
    LambdaMixedReward, 
    XORAction,
    XORObservation,
    SearchForXOR,
)

register(
    id="longshot/avgQ-d2-formula",
    entry_point="longshot.gym:AvgQ_D2_FormulaEnv",
)