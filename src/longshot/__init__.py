from gymnasium.envs.registration import register

register(
    id="longshot/avgQ-d2-formula-v0",
    entry_point="longshot.gym.envs:AvgQ_D2_FormulaEnv",
)