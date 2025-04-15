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

@pytest.mark.parametrize("granularity", ["character"])
def test_random_loop(granularity):
    # Check if the environment can be reset and stepped through
    env = gym.make("longshot/avgQ-d2-formula-v0", 
        n=4, 
        granularity=granularity,
        ftype=FormulaType.Conjunctive
        )
    observation, info = env.reset(seed=1452)

    episode_over = False

    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        print(Character(CharacterType(action[0].item()), action[1]))
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation, reward, terminated, truncated, info)

        episode_over = terminated or truncated

    env.close()

if __name__ == "__main__":
    test_gym_registration()
    test_random_loop("character")
    # test_random_loop("clause")