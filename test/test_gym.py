import gymnasium as gym
import pytest
import numpy as np

import longshot.gym
from longshot.circuit import FormulaType, Literals
from longshot.gym.envs import Character, CharacterType

def test_gym_registration():
    # Check if the environment is registered
    env = gym.make("longshot/avgQ-d2-formula-v0", 
        n=3, 
        granularity="clause/term", 
        ftype=FormulaType.Conjunctive,
        )
    
    assert env is not None

@pytest.mark.repeat(3)
@pytest.mark.parametrize("n", [2,3,4,5])
@pytest.mark.parametrize("granularity", ["character", "clause/term"])
def test_random_loop(n, granularity):
    # Check if the environment can be reset and stepped through
    env = gym.make("longshot/avgQ-d2-formula-v0", 
        n=n, 
        granularity=granularity,
        ftype=FormulaType.Conjunctive
        )
        
    observation, info = env.reset(seed=587)

    episode_over = False

    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        # print(str(Character(CharacterType(action[0].item()), action[1].item())))
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation, reward, terminated, truncated, info)

        episode_over = terminated or truncated

    env.close()

def test_character_mode_1():
    env = gym.make("longshot/avgQ-d2-formula-v0", 
        n=3, 
        granularity="character",
        ftype=FormulaType.Conjunctive,
        )
        
    env.reset()
    
    a = Character(CharacterType.VAR, 1) # x1
    assert env.step(a) == (np.array([0.], dtype=np.float32), 0.0, False, False, {})
    a = Character(CharacterType.NVAR, 0) # ¬x0
    assert env.step(a) == (np.array([0.], dtype=np.float32), 0.0, False, False, {})
    a = Character(CharacterType.EOS, 0) # EOS
    assert env.step(a) == (np.array([1.5], dtype=np.float32), 1.5, True, False, {})

    env.close()

def test_character_mode_2():
    env = gym.make("longshot/avgQ-d2-formula-v0", 
        n=3, 
        granularity="character",
        ftype=FormulaType.Conjunctive,
        )
        
    env.reset()
    
    a = Character(CharacterType.EOC, 0) # x1
    assert env.step(a) == (np.array([0.], dtype=np.float32), 0.0, False, False, {'redundant': 1})
    a = Character(CharacterType.NEG, 0) # ¬
    assert env.step(a) == (np.array([0.], dtype=np.float32), 0.0, False, False, {})
    a = Character(CharacterType.NEG, 0) # ¬
    assert env.step(a) == (np.array([0.], dtype=np.float32), 0.0, False, False, {})
    a = Character(CharacterType.NVAR, 1) # ¬x1
    assert env.step(a) == (np.array([0.], dtype=np.float32), 0.0, False, False, {})
    a = Character(CharacterType.NEG, 0) # ¬
    assert env.step(a) == (np.array([0.], dtype=np.float32), 0.0, False, False, {})
    a = Character(CharacterType.VAR, 0) # x0
    assert env.step(a) == (np.array([0.], dtype=np.float32), 0.0, False, False, {})
    a = Character(CharacterType.NEG, 0) # ¬
    assert env.step(a) == (np.array([0.], dtype=np.float32), 0.0, False, False, {})
    a = Character(CharacterType.EOS, 0) # EOS
    # the 2-nd, 3-rd, and 7-th characters are redundant
    assert env.step(a) == (np.array([1.5], dtype=np.float32), 1.5, True, False, {'redundant': 3})

    env.close()
    
def test_clauseterm_mode_1():
    env = gym.make("longshot/avgQ-d2-formula-v0", 
        n=3, 
        granularity="clause/term",
        ftype=FormulaType.Disjunctive,
        )
        
    env.reset()
    
    a = Literals([0,1], [0]) # x0.¬x0.x1
    assert env.step(a) == (np.array([0.0], dtype=np.float32), 0.0, False, False, {"redundant": 1})
    a = Literals([0,1], []) # x0.x1
    assert env.step(a) == (np.array([1.5], dtype=np.float32), 1.5, False, False, {})
    a = Literals([2], [0]) # ¬x0.x2
    assert env.step(a) == (np.array([2.], dtype=np.float32), 0.5, False, False, {})
    a = Literals([2], [0]) # ¬x0.x2
    assert env.step(a) == (np.array([2.], dtype=np.float32), 0.0, False, False, {})
    a = Literals([0,1], [2]) # x0.x1.¬x2
    assert env.step(a) == (np.array([2.], dtype=np.float32), 0.0, False, False, {})
    a = Literals([], [1,2]) # ¬x1.¬x2
    assert env.step(a) == (np.array([2.5], dtype=np.float32), 0.5, False, False, {})
    a = Literals([], [1,2]) # ¬x1.¬x2
    assert env.step(a) == (np.array([2.5], dtype=np.float32), 0.0, False, False, {})
    a = Literals([], [0,1,2]) # ¬x0.¬x1.¬x2
    assert env.step(a) == (np.array([2.5], dtype=np.float32), 0.0, False, False, {})
    a = Literals([0], [1,2]) # x0.¬x1.¬x2
    assert env.step(a) == (np.array([2.5], dtype=np.float32), 0.0, False, False, {})
    a = Literals([], [0,2]) # ¬x0.¬x2
    assert env.step(a) == (np.array([1.75], dtype=np.float32), -0.75, False, False, {})
    a = Literals([], [0]) # ¬x0
    assert env.step(a) == (np.array([1.75], dtype=np.float32), 0.0, False, False, {})
    a = Literals([1], [0]) # ¬x0.x1
    assert env.step(a) == (np.array([1.75], dtype=np.float32), 0.0, False, False, {})
    a = Literals([0,1,2], []) # x0.x1.x2
    assert env.step(a) == (np.array([1.75], dtype=np.float32), 0.0, False, False, {})
    a = Literals([0,1], []) # x0.x1
    assert env.step(a) == (np.array([1.75], dtype=np.float32), 0.0, False, False, {})
    a = Literals([0], [1,2]) # x0.¬x1.¬x2
    assert env.step(a) == (np.array([1.75], dtype=np.float32), 0.0, False, False, {})
    a = Literals([2], []) # x2
    assert env.step(a) == (np.array([0.], dtype=np.float32), -1.75, True, True, {})
    
    env.close()


def test_clauseterm_mode_2():
    env = gym.make("longshot/avgQ-d2-formula-v0", 
        n=3, 
        granularity="clause/term",
        ftype=FormulaType.Disjunctive,
        )
        
    env.reset()
    
    a = Literals([0], [1,2]) # x0.¬x1.¬x2
    assert env.step(a) == (np.array([1.75], dtype=np.float32), 1.75, False, False, {})
    a = Literals([1], [0,2]) # ¬x0.x1.¬x2
    assert env.step(a) == (np.array([2.], dtype=np.float32), 0.25, False, False, {})
    a = Literals([2], [0,1]) # ¬x0.¬x1.x2
    # print(env.step(a))
    assert env.step(a) == (np.array([2.75], dtype=np.float32), 0.75, False, False, {})
    a = Literals([0,1,2], []) # x0.x1.x2
    assert env.step(a) == (np.array([3.], dtype=np.float32), 0.25, False, False, {})
    
    env.close()


def test_clauseterm_mode_3():
    env = gym.make("longshot/avgQ-d2-formula-v0", 
        n=4, 
        granularity="clause/term",
        ftype=FormulaType.Disjunctive,
        )
        
    env.reset()
    
    a = Literals([0], [1,2,3]) # x0.¬x1.¬x2
    assert env.step(a) == (np.array([1.875], dtype=np.float32), 1.875, False, False, {})
    a = Literals([1], [0,2,3]) # ¬x0.x1.¬x2
    assert env.step(a) == (np.array([2.], dtype=np.float32), 0.125, False, False, {})
    a = Literals([2], [0,1,3]) # ¬x0.¬x1.x2
    assert env.step(a) == (np.array([2.375], dtype=np.float32), 0.375, False, False, {})
    a = Literals([0,1,2], [3]) # x0.x1.x2
    assert env.step(a) == (np.array([2.5], dtype=np.float32), 0.125, False, False, {})
    a = Literals([3], [0,1,2]) # x0.¬x1.¬x2
    assert env.step(a) == (np.array([3.375], dtype=np.float32), 0.875, False, False, {})
    a = Literals([0,1,3], [2]) # ¬x0.x1.¬x2
    assert env.step(a) == (np.array([3.5], dtype=np.float32), 0.125, False, False, {})
    a = Literals([0,2,3], [1]) # ¬x0.¬x1.x2
    assert env.step(a) == (np.array([3.875], dtype=np.float32), 0.375, False, False, {})
    a = Literals([1,2,3], [0]) # x0.x1.x2
    assert env.step(a) == (np.array([4.], dtype=np.float32), 0.125, False, False, {})
    
    env.close()

if __name__ == "__main__":
    # pytest.main([__file__])
    # test_random_loop(3, "clause/term")
    test_clauseterm_mode_3()