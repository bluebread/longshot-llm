from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
from typing import Any

from .char_type import CharacterType, Character
from ...error import LongshotError
from ...circuit import Literals, FormulaType, NormalFormFormula

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
        ftype: FormulaType | None = None,
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
        if granularity not in ["character", "clause/term"]:
            raise LongshotError(f"Expected `granularity` to be 'character' or 'clause/term', got {granularity}")
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
        self.max_size = max_size
        self.max_width = max_width
        self.granularity = granularity
        self.stride = stride
        self.render_mode = render_mode
        self.init_state = init_state
        self.ftype = ftype if self.init_state is None else self.init_state.ftype
        
        self.observation_space = spaces.Box(low=0, high=n, shape=(1,), dtype=np.float32)
        
        if self.granularity == "character":
            self.action_space = spaces.MultiDiscrete(np.array([len(CharacterType), n]), dtype=np.int32)
        elif self.granularity == "clause/term":
            self.action_space = spaces.MultiDiscrete(np.full((n,), 3), dtype=np.int32)
        
        self._terminated = True
    
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if self.init_state is not None:
            self._formula = self.init_state
            self._cur_avgQ = np.array([self._formula.avgQ()])
            self._prev_avgQ = np.zeros_like(self._cur_avgQ)
        else:
            self._formula = NormalFormFormula(num_vars=self.num_vars, ftype=self.ftype)
            self._cur_avgQ = self._prev_avgQ = np.zeros(shape=(1,), dtype=np.float32)
            
        self._step_count = 0
        self._char_buffer = []
        self._terminated = False
        self._has_new_literals = False
        
        super().reset(seed=seed, options=options)
        self.action_space.seed(seed=seed)
        
        return self._cur_avgQ, {}
    
    def step(
        self, 
        action: Character | Literals | NDArray[np.integer[Any]]
    ) -> tuple[float, float, bool, bool, dict[str, Any]]:
        truncated = False
        info = {}
        self._prev_avgQ = self._cur_avgQ
        self._step_count += 1
        
        if self._terminated:
            raise LongshotError("Environment has already terminated.")
        
        if self.granularity == "character":
            if not isinstance(action, (Character, np.ndarray)):
                raise LongshotError(f"Expected action to be Character, got {type(action).__name__}")
            if isinstance(action, np.ndarray):
                if action.shape != (2,):
                    raise LongshotError(f"Expected action to be of shape (2,), got {action.shape}")
                action = Character(CharacterType(action[0].item()), action[1])
            if action.char_type in [CharacterType.EOS, CharacterType.EOC]:
                self._terminated = (action.char_type == CharacterType.EOS)
                
                if len(self._char_buffer) > 0:
                    new_literals = self._merge_char_buffer(info)
                    if not new_literals.is_constant():
                        self._formula.add(new_literals)
                        self._has_new_literals = True
                    else:
                        info['redundant'] = info.get('redundant', 0) + new_literals.width() + 1
                else:
                    info['redundant'] = info.get('redundant', 0) + 1
            elif action.char_type in [CharacterType.NEG, CharacterType.VAR, CharacterType.NVAR]:
                self._char_buffer.append(action)
            else:
                raise LongshotError(f"Unknown character type: {action.char_type}")
        elif self.granularity == "clause/term":
            if not isinstance(action, (Literals, np.ndarray)):
                raise LongshotError(f"Expected action to be Clause, got {type(action).__name__}")
            if isinstance(action, np.ndarray):
                if action.shape != (self.num_vars,):
                    raise LongshotError(f"Expected action to be of shape ({self.num_vars},), got {action.shape}")
                pvs = [i for i in range(self.num_vars) if action[i] == 0]
                nvs = [i for i in range(self.num_vars) if action[i] == 1]

                if len(pvs) + len(nvs) == 0:
                    self._terminated = True
                    print("<EOS>")
                    return self._cur_avgQ, 0.0, self._terminated, True, info
                else:
                    new_literals = Literals(pos=pvs, neg=nvs)
                    print(str(new_literals))
            else:
                new_literals = action

            if not new_literals.is_constant():
                self._formula.add(new_literals)
                self._has_new_literals = True
            else:
                info['redundant'] = info.get('redundant', 0) + 1
        else:
            raise LongshotError(f"Unknown granularity: {self.granularity}")
        
        if self._step_count >= self.stride or self._terminated:
            if self._has_new_literals or self._terminated:
                self._cur_avgQ = np.array([self._formula.avgQ()], dtype=np.float32)
                self._step_count = 0
                self._has_new_literals = False
                
                if self._cur_avgQ <= 0.0:
                    self._terminated = truncated = True
            
        if self.max_size is not None and self._formula.size >= self.max_size:
            self._terminated = True
        if self.max_width is not None and self._formula.width > self.max_width:
            self._terminated = truncated = True
        
        obs = self._cur_avgQ
        reward = (self._cur_avgQ - self._prev_avgQ).item()

        return obs, reward, self._terminated, truncated, info

    def render(self) -> None:
        pass

    def close(self) -> None:
        self._terminated = True
    
    def _merge_char_buffer(self, info: dict) -> Literals:
        if len(self._char_buffer) == 0:
            return Literals()
        
        pvs = []
        nvs = []
        
        for idx, char in enumerate(self._char_buffer):
            if char is None:
                continue # this character has been removed by the previous negation
            if char.char_type == CharacterType.NEG:
                nxt_char = self._char_buffer[idx + 1] if idx + 1 < len(self._char_buffer) else None
                
                if nxt_char is None or nxt_char.char_type in [CharacterType.EOC, CharacterType.EOS]:
                    info['redundant'] = info.get('redundant', 0) + 1
                elif nxt_char.char_type == CharacterType.VAR:
                    # TODO: check if the variable is already in the list
                    self._char_buffer[idx + 1] = Character(CharacterType.NVAR, nxt_char.var_id)
                elif nxt_char.char_type == CharacterType.NVAR:
                    # TODO: check if the variable is already in the list
                    self._char_buffer[idx + 1] = Character(CharacterType.VAR, nxt_char.var_id)
                    info['redundant'] = info.get('redundant', 0) + 1
                elif nxt_char.char_type == CharacterType.NEG:
                    self._char_buffer[idx + 1] = None
                    info['redundant'] = info.get('redundant', 0) + 2
                else:
                    raise LongshotError(f"Invalid character after negation: {nxt_char.char_type}")
                
                self._char_buffer[idx] = None
            elif char.char_type == CharacterType.VAR:
                if char.var_id in pvs:
                    info['redundant'] = info.get('redundant', 0) + 1
                else:
                    pvs.append(char.var_id) 
            elif char.char_type == CharacterType.NVAR:
                if char.var_id in nvs:
                    info['redundant'] = info.get('redundant', 0) + 1
                else:
                    nvs.append(char.var_id)
            else:
                raise LongshotError(f"Some character of invalid type in character buffer: {char.char_type}")
        
        self._char_buffer.clear()
        
        return Literals(pos=pvs, neg=nvs)