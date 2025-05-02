from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
from typing import Any
import warnings

from ...error import LongshotError
from ...circuit import Literals, FormulaType, NormalFormFormula


class AvgQ_D2_FormulaEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(
        self,
        n: int,
        ftype: FormulaType | None = None,
        mono: bool = False,
        init_formula: NormalFormFormula | None = None,
        render_mode: str | None = None ,    
    ):
        """
        Initialize the environment with the given parameters.
        """
        super().__init__()
        
        # Check if types of the arguments are valid
        if init_formula is None:
            if not isinstance(n, int):
                raise LongshotError(f"Expected `n` to be int, got {type(n).__name__}")
            if render_mode is not None and not isinstance(render_mode, str):
                raise LongshotError(f"Expected `render_mode` to be str or None, got {type(render_mode).__name__}")
            if init_formula is None and ftype is None:
                raise LongshotError("Either `init_state` or `ftype` should be provided.")
            
            self.num_vars = n
            self.ftype = ftype
            self.mono = mono
            self.init_formula = None
            
        else:
            if not isinstance(init_formula, NormalFormFormula):
                raise LongshotError(f"Expected `init_state` to be NormalFormFormula or None, got {type(init_formula).__name__}")
            
            self.num_vars = init_formula.num_vars
            self.ftype = init_formula.ftype
            self.mono = init_formula.is_mono
            self.init_formula = init_formula.copy()
        
        self.action_space = spaces.MultiDiscrete([3] * n, dtype=np.int8)
        self.observation_space = spaces.Sequence(self.action_space, stack=True)
        self.render_mode = render_mode
        
        self._terminated = True
        self._closed = False
        self._formula = None
        self._literals_seq = None
        
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Reset the environment to an initial state.
        """
        self._terminated = False
        self._step_count = 0
        
        if self.init_formula is not None and isinstance(self.init_formula, NormalFormFormula):
            self._formula = self.init_formula.copy()
            self._cur_avgQ = self._formula.avgQ()
            self._prev_avgQ = 0.0
            self._literals_seq = np.vstack([ls.vectorize(self._formula.num_vars) for ls in self._formula])
        else:
            self._formula = NormalFormFormula(num_vars=self.num_vars, ftype=self.ftype, mono=self.mono)
            self._cur_avgQ = self._prev_avgQ = 0.
            self._literals_seq = np.empty((0,), dtype=np.int8)
            
        super().reset(seed=seed, options=options)
        self.observation_space.seed(seed=seed)
        self.action_space.seed(seed=seed)
        
        return self._literals_seq, {}
    
    def step(self, action: Literals | NDArray[np.integer[Any]]) -> tuple[float, float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment with the given action.
        """
        if self._terminated:
            raise LongshotError("Environment has already terminated.")
        if self._closed:
            raise LongshotError("Environment has been closed.")
        if not isinstance(action, (Literals, np.ndarray)):
            raise LongshotError(f"Expected action to be Literals or np.ndarray, got {type(action).__name__}")
        
        info = {'adding': False, 'removing': False}
        ls = None
        self._prev_avgQ = self._cur_avgQ
        self._step_count += 1
        
        if isinstance(action, np.ndarray):
            if action.shape != (self.num_vars,):
                raise LongshotError(f"Expected action to be of shape ({self.num_vars},), got {action.shape}")
            pvs = np.where(action == 0)[0].tolist()
            nvs = np.where(action == 1)[0].tolist()

            ls = Literals(pos=pvs, neg=nvs)
        else:
            ls = action
            
        if ls.is_empty: # <End of Sequence>
            self._terminated = True
            info['avgQ'] = self._cur_avgQ
            return self._literals_seq, 0.0, self._terminated, False, info
        elif ls.is_contradictory:
            pass
        else:
            if not self.mono and ls in self._formula:
                info['removing'] = True
                self._formula.remove(ls)
            else:
                info['adding'] = True
                self._formula.add(ls)
            
            if len(self._formula) > 0:
                # it is hard to optimize here because Numpy arrays are immutable
                self._literals_seq = np.vstack([
                    ls.vectorize(self._formula.num_vars) for ls in self._formula
                ])
            else:
                self._literals_seq = np.empty((0,))
                
            self._cur_avgQ = self._formula.avgQ()
            
        reward = self._cur_avgQ - self._prev_avgQ
        info['avgQ'] = self._cur_avgQ
        
        return self._literals_seq, reward, self._terminated, False, info
    
    def render(self) -> None:
        raise NotImplementedError("Rendering is not implemented yet.")
    
    def close(self) -> None:
        self._closed = True
        self._terminated = True
        
        del self._formula
        del self._literals_seq
        del self.init_formula
        
    @property
    def formula(self) -> NormalFormFormula:
        """
        Return the current formula.
        """
        return self._formula.copy()