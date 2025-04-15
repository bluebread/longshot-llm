import gymnasium as gym
import pytest

import longshot.gym
from longshot.circuit import FormulaType
from longshot.gym.envs import Character, CharacterType

def test_gym_registration():
    # Check if the environment is registered
    env = gym.make("longshot/avgQ-d2-formula-v0", 
        n=3, 
        granularity="clause", 
        ftype=FormulaType.Conjunctive,
        )
    
    assert env is not None

@pytest.mark.repeat(3)
@pytest.mark.parametrize("n", [2,3,4,5])
@pytest.mark.parametrize("granularity", ["character", "clause"])
def test_random_loop(n, granularity):
    # Check if the environment can be reset and stepped through
    env = gym.make("longshot/avgQ-d2-formula-v0", 
        n=n, 
        granularity=granularity,
        ftype=FormulaType.Conjunctive
        )
        
    observation, info = env.reset()

    episode_over = False

    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

    env.close()
