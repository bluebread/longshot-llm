import gymnasium as gym
import pytest
import numpy as np

import longshot
from longshot.circuit import FormulaType, Literals
from longshot.gym import FlattenSequence, LambdaMixedReward, XORAction

def test_gym_registration():
    print("test_gym_registration")
    # Check if the environment is registered
    env = gym.make("longshot/avgQ-d2-formula", 
        n=3, 
        ftype=FormulaType.Conjunctive,
        )
    
    assert env is not None

@pytest.mark.repeat(3)
@pytest.mark.parametrize("n", [2,3,4,5])
def test_random_loop(n):
    print("test_random_loop")
    # Check if the environment can be reset and stepped through
    env = gym.make("longshot/avgQ-d2-formula", 
        n=n, 
        ftype=FormulaType.Conjunctive,
        mono=True
        )
        
    observation, info = env.reset()

    episode_over = False

    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        # print(str(Character(CharacterType(action[0].item()), action[1].item())))
        observation, reward, terminated, truncated, info = env.step(action)
        # print(observation, reward, terminated, truncated, info)

        episode_over = terminated or truncated

    env.close()

    
def test_mono_mode_1():
    env = gym.make("longshot/avgQ-d2-formula", 
        n=3, 
        ftype=FormulaType.Disjunctive,
        mono=True,
        )
        
    env.reset()
    
    a = Literals([0,1], [0]) # x0.¬x0.x1
    obs, *others = env.step(a)
    assert len(obs) == 0 and isinstance(obs, np.ndarray)
    assert others == [0.0, False, False, {'adding': False, 'removing': False, 'avgQ': 0.0}]
    
    a = Literals([0,1], []) # x0.x1
    obs, *others = env.step(a)
    assert len(obs) == 1
    assert others == [1.5, False, False, {'adding': True, 'removing': False, 'avgQ': 1.5}]
    
    a = Literals([2], [0]) # ¬x0.x2
    obs, *others = env.step(a)
    assert len(obs) == 2
    assert others == [0.5, False, False, {'adding': True, 'removing': False, 'avgQ': 2.0}]
    
    a = Literals([2], [0]) # ¬x0.x2
    obs, *others = env.step(a)
    assert len(obs) == 2
    assert others == [0.0, False, False, {'adding': True, 'removing': False, 'avgQ': 2.0}]
    
    a = Literals([0,1], [2]) # x0.x1.¬x2
    obs, *others = env.step(a)
    assert len(obs) == 3
    assert others == [0.0, False, False, {'adding': True, 'removing': False, 'avgQ': 2.0}]
    
    a = Literals([], [1,2]) # ¬x1.¬x2
    obs, *others = env.step(a)
    assert len(obs) == 4
    assert others == [0.5, False, False, {'adding': True, 'removing': False, 'avgQ': 2.5}]
    
    a = Literals([], [1,2]) # ¬x1.¬x2
    obs, *others = env.step(a)
    assert len(obs) == 4
    assert others == [0.0, False, False, {'adding': True, 'removing': False, 'avgQ': 2.5}]
    
    a = Literals([], [0,1,2]) # ¬x0.¬x1.¬x2
    obs, *others = env.step(a)
    assert len(obs) == 5
    assert others == [0.0, False, False, {'adding': True, 'removing': False, 'avgQ': 2.5}]
    
    a = Literals([0], [1,2]) # x0.¬x1.¬x2
    obs, *others = env.step(a)
    assert len(obs) == 6
    assert others == [0.0, False, False, {'adding': True, 'removing': False, 'avgQ': 2.5}]
    
    a = Literals([], [0,2]) # ¬x0.¬x2
    obs, *others = env.step(a)
    assert len(obs) == 7
    assert others == [-0.75, False, False, {'adding': True, 'removing': False, 'avgQ': 1.75}]
    
    a = Literals([], [0]) # ¬x0
    obs, *others = env.step(a)
    assert len(obs) == 8
    assert others == [0.0, False, False, {'adding': True, 'removing': False, 'avgQ': 1.75}]
    
    a = Literals([1], [0]) # ¬x0.x1
    obs, *others = env.step(a)
    assert len(obs) == 9
    assert others == [0.0, False, False, {'adding': True, 'removing': False, 'avgQ': 1.75}]
    
    a = Literals([0,1,2], []) # x0.x1.x2
    obs, *others = env.step(a)
    assert len(obs) == 10
    assert others == [0.0, False, False, {'adding': True, 'removing': False, 'avgQ': 1.75}]
    
    a = Literals([0,1], []) # x0.x1
    obs, *others = env.step(a)
    assert len(obs) == 10
    assert others == [0.0, False, False, {'adding': True, 'removing': False, 'avgQ': 1.75}]
    
    a = Literals([0], [1,2]) # x0.¬x1.¬x2
    obs, *others = env.step(a)
    assert len(obs) == 10
    assert others == [0.0, False, False, {'adding': True, 'removing': False, 'avgQ': 1.75}]
    
    a = Literals([2], []) # x2
    obs, *others = env.step(a)
    assert len(obs) == 11
    assert others == [-1.75, False, False, {'adding': True, 'removing': False, 'avgQ': 0.0}]
    
    env.close()


def test_mono_mode_2():
    env = gym.make("longshot/avgQ-d2-formula", 
        n=3, 
        ftype=FormulaType.Disjunctive,
        mono=True,
        )
        
    env.reset()
    
    a = Literals([0], [1,2]) # x0.¬x1.¬x2
    obs, *others = env.step(a)
    assert len(obs) == 1
    assert others == [1.75, False, False, {'adding': True, 'removing': False, 'avgQ': 1.75}]

    a = Literals([1], [0,2]) # ¬x0.x1.¬x2
    obs, *others = env.step(a)
    assert len(obs) == 2
    assert others == [0.25, False, False, {'adding': True, 'removing': False, 'avgQ': 2.0}]

    a = Literals([2], [0,1]) # ¬x0.¬x1.x2
    obs, *others = env.step(a)
    assert len(obs) == 3
    assert others == [0.75, False, False, {'adding': True, 'removing': False, 'avgQ': 2.75}]

    a = Literals([0,1,2], []) # x0.x1.x2
    obs, *others = env.step(a)
    assert len(obs) == 4
    assert others == [0.25, False, False, {'adding': True, 'removing': False, 'avgQ': 3.0}]

    env.close()


def test_mono_mode_3():
    env = gym.make("longshot/avgQ-d2-formula", 
        n=4, 
        ftype=FormulaType.Disjunctive,
        mono=True,
        )
        
    env.reset()
    
    a = Literals([0], [1,2,3])
    obs, *others = env.step(a)
    assert len(obs) == 1
    assert others == [1.875, False, False, {'adding': True, 'removing': False, 'avgQ': 1.875}]

    a = Literals([1], [0,2,3])
    obs, *others = env.step(a)
    assert len(obs) == 2
    assert others == [0.125, False, False, {'adding': True, 'removing': False, 'avgQ': 2.0}]

    a = Literals([2], [0,1,3])
    obs, *others = env.step(a)
    assert len(obs) == 3
    assert others == [0.375, False, False, {'adding': True, 'removing': False, 'avgQ': 2.375}]

    a = Literals([0,1,2], [3])
    obs, *others = env.step(a)
    assert len(obs) == 4
    assert others == [0.125, False, False, {'adding': True, 'removing': False, 'avgQ': 2.5}]

    a = Literals([3], [0,1,2])
    obs, *others = env.step(a)
    assert len(obs) == 5
    assert others == [0.875, False, False, {'adding': True, 'removing': False, 'avgQ': 3.375}]

    a = Literals([0,1,3], [2])
    obs, *others = env.step(a)
    assert len(obs) == 6
    assert others == [0.125, False, False, {'adding': True, 'removing': False, 'avgQ': 3.5}]

    a = Literals([0,2,3], [1])
    obs, *others = env.step(a)
    assert len(obs) == 7
    assert others == [0.375, False, False, {'adding': True, 'removing': False, 'avgQ': 3.875}]

    a = Literals([1,2,3], [0])
    obs, *others = env.step(a)
    assert len(obs) == 8
    assert others == [0.125, False, False, {'adding': True, 'removing': False, 'avgQ': 4.0}]

    env.close()


def test_counting_mode_1():
    env = gym.make("longshot/avgQ-d2-formula", 
        n=4, 
        ftype=FormulaType.Disjunctive,
        mono=False,
        )
        
    env.reset()

    a = Literals([0], [1])
    obs, *others = env.step(a)
    assert len(obs) == 1
    assert others == [1.5, False, False, {'adding': True, 'removing': False, 'avgQ': 1.5}]
    a = Literals([0,2], [3])
    obs, *others = env.step(a)
    assert len(obs) == 2
    assert others == [0.375, False, False, {'adding': True, 'removing': False, 'avgQ': 1.875}]
    a = Literals([3], [0,1,2])
    obs, *others = env.step(a)
    assert len(obs) == 3
    assert others == [0.875, False, False, {'adding': True, 'removing': False, 'avgQ': 2.75}]

    a = Literals([3], [0,1,2])
    obs, *others = env.step(a)
    assert len(obs) == 2
    assert others == [-0.875, False, False, {'adding': False, 'removing': True, 'avgQ': 1.875}]
    a = Literals([0,2], [3])
    obs, *others = env.step(a)
    assert len(obs) == 1
    assert others == [-0.375, False, False, {'adding': False, 'removing': True, 'avgQ': 1.5}]
    a = Literals([0], [1])
    obs, *others = env.step(a)
    assert len(obs) == 0
    assert others == [-1.5, False, False, {'adding': False, 'removing': True, 'avgQ': 0.0}]
    
    a = Literals([], [])
    obs, *others = env.step(a)
    assert others == [0.0, True, False, {'adding': False, 'removing': False, 'avgQ': 0.0}]
    
    env.close()

@pytest.mark.repeat(3)
def test_wrappers():
    env = gym.make("longshot/avgQ-d2-formula", 
        n=5, 
        ftype=FormulaType.Disjunctive,
        mono=False,
        )
        
    env = FlattenSequence(env)
    env = LambdaMixedReward(env)
    env = XORAction(env)
    
    observation, info = env.reset()

    for _ in range(100):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        # print(str(Character(CharacterType(action[0].item()), action[1].item())))
        observation, reward, terminated, truncated, info = env.step(action)
        # print(observation, reward, terminated, truncated, info)

        episode_over = terminated or truncated
        
        if episode_over:
            break

    env.close()
    
if __name__ == "__main__":
    pytest.main([__file__])
    # test_gym_registration()
    # test_mono_mode_1()
    # test_mono_mode_2()
    # test_mono_mode_3()
    # test_counting_mode_1()
    # test_wrappers()
    