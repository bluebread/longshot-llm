from enum import auto, IntEnum 
from ...error import LongshotError

import numpy as np

class CharacterType(IntEnum):
    def _generate_next_value_(name, start, count, last_values):
        # Here, count starts at 0 for the first item, then increments.
        return count
    
    EOS = auto() # End of Sequence
    EOC = auto() # End of Clause
    NEG = auto() # Negative
    VAR = auto() # Variable
    NVAR = auto() # Negative variable

class Character:
    def __init__(self, char_type: CharacterType, var_id: int | np.integer | None = None):
        if char_type is None or not isinstance(char_type, CharacterType):
            raise LongshotError("`char_type` should be a CharacterType.")
        if char_type not in [CharacterType.EOS, CharacterType.EOC, CharacterType.NEG, CharacterType.VAR, CharacterType.NVAR]:
            raise LongshotError("`char_type` should be one of the defined CharacterType.")
        if var_id is not None and not isinstance(var_id, (int, np.integer)):
            raise LongshotError("`var_id` should be an integer or None.")
        
        self.char_type = char_type
        self.var_id = var_id

    def __repr__(self):
        return f"Character(type={self.char_type}, value={self.value})"

    def __str__(self):
        if self.char_type == CharacterType.EOS:
            return "<EOS>"
        elif self.char_type == CharacterType.EOC:
            return "<EOC>"
        elif self.char_type == CharacterType.NEG:
            return "¬"
        elif self.char_type == CharacterType.VAR:
            return f"x{self.var_id}"
        elif self.char_type == CharacterType.NVAR:
            return f"¬x{self.var_id}"
        else:
            raise LongshotError(f"Unknown character type: {self.char_type}")