from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
from typing import Any

from ...error import LongshotError
from ...circuit import Literals, FormulaType, NormalFormFormula

class AvgQ_D2_FormulaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        n: int,
        stride: int = 1,
        init_state: NormalFormFormula | None = None,
        ftype: FormulaType | None = None,
        no_obs: bool = False,
        render_mode: str | None = None ,    
    ):
        """
        Initialize the environment with the given parameters.
        """
        # Check if types of the arguments are valid
        if not isinstance(n, int):
            raise LongshotError(f"Expected `n` to be int, got {type(n).__name__}")
        if not isinstance(stride, int):
            raise LongshotError(f"Expected `stride` to be int, got {type(stride).__name__}")
        if init_state is not None and not isinstance(init_state, NormalFormFormula):
            raise LongshotError(f"Expected `init_state` to be NormalFormFormula or None, got {type(init_state).__name__}")
        if render_mode is not None and not isinstance(render_mode, str):
            raise LongshotError(f"Expected `render_mode` to be str or None, got {type(render_mode).__name__}")
        if init_state is None and ftype is None:
            raise LongshotError("Either `init_state` or `ftype` should be provided.")
        
        super().__init__()
        
        if render_mode is not None:
            raise NotImplementedError("Rendering is not implemented yet.")
        
        self.num_vars = n
        self.stride = stride
        self.render_mode = render_mode
        self.init_state = init_state
        self.ftype = ftype if self.init_state is None else self.init_state.ftype
        self.no_obs = no_obs
        
        self.observation_space = spaces.MultiBinary(int(3**n - 1))
        self.action_space = spaces.MultiDiscrete([3] * n, dtype=np.int8)
        
        self._terminated = True
    
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if self.init_state is not None:
            self._formula = self.init_state
            self._cur_avgQ = self._formula.avgQ()
            self._prev_avgQ = 0.
        else:
            self._formula = NormalFormFormula(num_vars=self.num_vars, ftype=self.ftype)
            self._cur_avgQ = self._prev_avgQ = 0.
            
        self._step_count = 0
        self._char_buffer = []
        self._terminated = False
        self._has_new_literals = False
        self._closed = False
        self._clauses_set = np.zeros(3**self.num_vars - 1, dtype=np.int8) if not self.no_obs else None
        
        # TODO: initialize `literals_set` if `init_state` is not None 
        if self.init_state is not None:
            raise NotImplementedError("Initializing `_clauses_set` with `init_state` is not implemented yet.")
        
        super().reset(seed=seed, options=options)
        self.observation_space.seed(seed=seed)
        self.action_space.seed(seed=seed)
        
        return self._clauses_set, {}
    
    def step(
        self, 
        action: Literals | NDArray[np.integer[Any]]
    ) -> tuple[float, float, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise LongshotError("Environment has already terminated.")
        if self._closed:
            raise LongshotError("Environment has been closed.")
        
        truncated = False
        info = {'redundant': False}
        self._prev_avgQ = self._cur_avgQ
        self._step_count += 1
        
        if not isinstance(action, (Literals, np.ndarray)):
            raise LongshotError(f"Expected action to be Clause, got {type(action).__name__}")
        if isinstance(action, np.ndarray):
            if action.shape != (self.num_vars,):
                raise LongshotError(f"Expected action to be of shape ({self.num_vars},), got {action.shape}")
            pvs = [i for i in range(self.num_vars) if action[i] == 0]
            nvs = [i for i in range(self.num_vars) if action[i] == 1]

            if len(pvs) + len(nvs) == 0: # <End of Sequence>
                self._terminated = True
                return self._cur_avgQ, 0.0, self._terminated, True, info
            else:
                new_literals = Literals(pos=pvs, neg=nvs)
        else:
            new_literals = action

        if not new_literals.is_constant(): # TODO: and formula's truth table is unchanged
            self._formula.add(new_literals)
            self._has_new_literals = True
            
            if not self.no_obs:
                self._clauses_set[self._encode_literals(new_literals)] = 1
        else:
            info['redundant'] = True
        
        if self._step_count >= self.stride or self._terminated:
            if self._has_new_literals or self._terminated:
                self._cur_avgQ = self._formula.avgQ()
                self._step_count = 0
                self._has_new_literals = False
                
                if self._cur_avgQ <= 0.0:
                    self._terminated = truncated = True
            
        obs = self._clauses_set.copy() if not self.no_obs else None
        reward = self._cur_avgQ - self._prev_avgQ
        info['avgQ'] = self._cur_avgQ

        return obs, reward, self._terminated, truncated, info

    def render(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True
    
    def _encode_literals(self, ls: Literals) -> int:
        """
        Encode the literals into a integer.
        """
        code = 0
        
        for i in range(self.num_vars - 1, -1, -1):
            code *= 3
            
            if (ls.pos & (1 << i)) > 0:
                code += 0
            elif (ls.neg & (1 << i)) > 0:
                code += 1
            else:
                code += 2
        
        return code