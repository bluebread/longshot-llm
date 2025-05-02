from gymnasium import ObservationWrapper, Wrapper, ActionWrapper
from gymnasium import spaces
import numpy as np
import numpy.typing as NDarray

class FlattenSequence(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.MultiBinary(int(3 ** env.num_vars))
    
    def observation(self, seq: NDarray) -> NDarray:
        obs = np.zeros(int(3 ** self.env.num_vars), dtype=np.int8)
        exp3_v = 3 ** np.arange(self.env.num_vars).reshape(-1,1)
        indices = seq @ exp3_v.T
        obs[indices] = 1
        
        return obs
        
class LambdaMixedReward(Wrapper):
    def __init__(self, env, alpha: float = 1., beta: float = 1., gamma: float = 0., eps: float = 1e-5):
        super().__init__(env)
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        avgQ = self.env._cur_avgQ
        prev_avgQ = self.env._prev_avgQ
        lam = info["lambda"] = self._avgQ_to_lambda(avgQ)
        prev_lam = self._avgQ_to_lambda(prev_avgQ)
        r = self.alpha * reward + self.beta * (lam - prev_lam) - self.gamma
        
        return obs, r, terminated, truncated, info
    
    def _avgQ_to_lambda(self, avgQ: float) -> float:
        """
        Convert the average Q function to a lambda function.
        The conversion is done using the formula:
            avgQ = n * ( 1 - 1 / (1 + lambda) ) + eps
        Rearranging gives:
            lambda = 1 / ( 1 - (avgQ - eps) / n ) - 1
        where n is the number of variables.
        """
        if avgQ < self.eps:
            return 0.
        return 1 / (1 - ((avgQ - self.eps) / self.env.num_vars)) - 1
    
class XORAction(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.MultiBinary(env.num_vars + 1)
        self._EOS = np.array([2] * env.num_vars, dtype=np.int8)
    
    def action(self, action: NDarray) -> NDarray:
        if action[-1].item() >= 1:
            action = self._EOS
        else:
            action = action[:-1]
        
        return action
    