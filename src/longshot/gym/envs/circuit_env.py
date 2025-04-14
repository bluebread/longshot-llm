from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
from typing import Any

from .char_type import CharacterType, Character
from ...error import LongshotError
from ...circuit import Clause, FormulaType, NormalFormFormula

class AvgQ_D2_FormulaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        n: int,
        max_size: int | None = None,
        max_width: int | None = None,
        granularity: str = "character",
        stride: int = 1,
        init_state: NormalFormFormula | None = None,
        render_mode: str | None = None ,    
    ):
        """
        Initialize the environment with the given parameters.
        """
        # Check if types of the arguments are valid
        if not isinstance(n, int):
            raise LongshotError(f"Expected `n` to be int, got {type(n).__name__}")
        if max_size is not None and not isinstance(max_size, int):
            raise LongshotError(f"Expected `max_size` to be int or None, got {type(max_size).__name__}")
        if max_width is not None and not isinstance(max_width, int):
            raise LongshotError(f"Expected `max_width` to be int or None, got {type(max_width).__name__}")
        if not isinstance(granularity, str):
            raise LongshotError(f"Expected `granularity` to be str, got {type(granularity).__name__}")
        if granularity not in ["character", "clause"]:
            raise LongshotError(f"Expected `granularity` to be 'character' or 'clause', got {granularity}")
        if not isinstance(stride, int):
            raise LongshotError(f"Expected `stride` to be int, got {type(stride).__name__}")
        if init_state is not None and not isinstance(init_state, NormalFormFormula):
            raise LongshotError(f"Expected `init_state` to be NormalFormFormula or None, got {type(init_state).__name__}")
        if render_mode is not None and not isinstance(render_mode, str):
            raise LongshotError(f"Expected `render_mode` to be str or None, got {type(render_mode).__name__}")
        
        super().__init__()
        
        if render_mode is not None:
            raise NotImplementedError("Rendering is not implemented yet.")
        
        self.num_vars = n
        self.max_size = max_size
        self.max_width = max_width
        self.granularity = granularity
        self.stride = stride
        self.render_mode = render_mode
        self.init_state = init_state
        
        self.observation_space = spaces.Box(low=0, high=n, shape=(1,), dtype=np.float32)
        
        if self.granularity == "character":
            self.action_space = spaces.MultiDiscrete(np.array([len(CharacterType), n]), dtype=np.int32)
        elif self.granularity == "clause":
            self.action_space = spaces.MultiDiscrete(np.full((n,), 3), dtype=np.int32)
        
        self._terminated = True
    
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if self.init_state is not None:
            self._formula = self.init_state
            self._cur_avgQ = self._formula.avgQ()
            self._prev_avgQ = 0
        else:
            self._formula = NormalFormFormula(num_vars=self.num_vars, ftype=FormulaType.Disjunctive)
            self._cur_avgQ = self._prev_avgQ = 0.0
            
        self._step_count = 0
        self._char_buffer = []
        self._terminated = False
        
        super().reset(seed=seed, options=options)
        
        return self._cur_avgQ, {}
    
    def step(
        self, 
        action: Character | Clause | NDArray[np.integer[Any]]
    ) -> tuple[float, float, bool, bool, dict[str, Any]]:
        truncated = False
        info = {}
        self._prev_avgQ = self._cur_avgQ
        self.step_count += 1
        
        if self._terminated:
            raise LongshotError("Environment has already terminated.")
        
        if self.granularity == "character":
            if not isinstance(action, [Character, NDArray[np.integer[Any]]]):
                raise LongshotError(f"Expected action to be Character, got {type(action).__name__}")
            if isinstance(action, NDArray[np.integer[Any]]):
                if action.shape != (2,):
                    raise LongshotError(f"Expected action to be of shape (2,), got {action.shape}")
                action = Character(CharacterType(action[0]), action[1])
            if action.char_type in [CharacterType.EOS, CharacterType.EOC]:
                self._terminated = (action.char_type == CharacterType.EOS)
                new_clause = self._merge_char_buffer()
                self._formula.add_clause(new_clause)
            elif action.char_type in [CharacterType.NEG, CharacterType.VAR, CharacterType.NVAR]:
                self._char_buffer.append(action)
            else:
                raise LongshotError(f"Unknown character type: {action.char_type}")
        elif self.granularity == "clause":
            if not isinstance(action, [Clause, NDArray[np.integer[Any]]]):
                raise LongshotError(f"Expected action to be Clause, got {type(action).__name__}")
            if isinstance(action, NDArray[np.integer[Any]]):
                if action.shape != (self.num_vars,):
                    raise LongshotError(f"Expected action to be of shape ({self.num_vars},), got {action.shape}")
                pos_vars = [i for i in range(self.num_vars) if action[i] == 0]
                neg_vars = [i for i in range(self.num_vars) if action[i] == 1]
                action = Clause(d_clause={"pos_vars": pos_vars, "neg_vars": neg_vars})

            self._formula.add_clause(action)
        else:
            raise LongshotError(f"Unknown granularity: {self.granularity}")
        
        if self._step_count >= self.stride or self._terminated:
            self._cur_avgQ = self._formula.avgQ()
            self._step_count = 0
            
        if self.max_size is not None and self._formula.size >= self.max_size:
            self._terminated = True
        if self.max_width is not None and self._formula.width > self.max_width:
            self._terminated = truncated = True
        
        return self._cur_avgQ, self._cur_avgQ - self._prev_avgQ, self._terminated, truncated, info

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
    
    def _merge_char_buffer(self) -> Clause:
        if len(self._char_buffer) == 0:
            return Clause()
        
        pos_vars = []
        neg_vars = []
        
        for idx, char in enumerate(self._char_buffer):
            if char is None:
                continue # this character has been removed by the previous negation
            if char.char_type == CharacterType.NEG:
                nxt_char = self._char_buffer[idx + 1] if idx + 1 < len(self._char_buffer) else None
                
                if nxt_char is None:
                    raise LongshotError("Negation character at the end of clause or sequence.")
                if nxt_char.char_type == CharacterType.VAR:
                    self._char_buffer[idx + 1] = Character(CharacterType.NVAR, nxt_char.value)
                elif nxt_char.char_type == CharacterType.NVAR:
                    self._char_buffer[idx + 1] = Character(CharacterType.VAR, nxt_char.value)
                elif nxt_char.char_type == CharacterType.NEG:
                    self._char_buffer[idx + 1] = None
                else:
                    raise LongshotError(f"Invalid character after negation: {nxt_char.char_type}")
                
                self._char_buffer[idx] = None
            elif char.char_type == CharacterType.VAR:
                pos_vars.append(char.value) 
            elif char.char_type == CharacterType.NVAR:
                neg_vars.append(char.value)
            else:
                raise LongshotError(f"Some character of invalid type in character buffer: {char.char_type}")
        
        self._char_buffer.clear()
        
        return Clause(d_clause={"pos_vars": pos_vars, "neg_vars": neg_vars})