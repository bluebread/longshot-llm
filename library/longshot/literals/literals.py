from collections.abc import Iterable
import numpy as np

from ..error import LongshotError
from .._core import (
    _Literals,
)

MAX_NUM_VARS = 32  # Maximum number of variables supported

class Literals(_Literals):
    """
    Represents a set of literals (e.g., `x1`, `¬x2`). It serves as a base class for `Clause` and `Term`.
    """
    def __init__(
        self, 
        pos: int | Iterable[int] | None = None,
        neg: int | Iterable[int] | None = None,
        d_literals: dict[int, bool] | None = None,
        ):
        """
        Initializes a `Literals` object. Can be initialized with positive and negative variable indices or a dictionary.
        :param pos: Positive variables.
        :param neg: Negative variables.
        :param d_literals: Dictionary of literals.
        :param ftype: Formula type.
        """
            
        if pos is not None or neg is not None:
            pos = 0 if pos is None else pos
            neg = 0 if neg is None else neg
            
            if isinstance(pos, Iterable):
                if not all(isinstance(i, (int, np.integer)) and i >= 0 and i < MAX_NUM_VARS for i in pos):
                    raise LongshotError("the argument `pos` is not a list of valid integers.")
                pos = sum((1 << i) for i in pos)
            if isinstance(neg, Iterable):
                if not all(isinstance(i, (int, np.integer)) and i >= 0 and i < MAX_NUM_VARS for i in neg):
                    raise LongshotError("the argument `neg` is not a list of valid integers.")
                neg = sum((1 << i) for i in neg)
            
            if isinstance(pos, (int, np.integer)) and isinstance(neg, (int, np.integer)):
                super().__init__(pos, neg)
            else:
                raise LongshotError("the arguments `pos` and `neg` should be integers.")
        elif d_literals is not None:
            if not isinstance(d_literals, dict):
                raise LongshotError("the argument `d_literals` is not a dictionary.")
            
            pos = neg = 0 
            
            for k, v in d_literals.items():
                if not isinstance(k, (int, np.integer)) or k < 0 or k >= MAX_NUM_VARS:
                    raise LongshotError("the key of the dictionary `d_literals` is not a valid integer.")
                if not isinstance(v, bool):
                    raise LongshotError("the value of the dictionary `d_literals` is not a boolean.")
                
                if v:
                    pos |= (1 << k)
                else:
                    neg |= (1 << k)
                
            super().__init__(pos, neg)
        else:
            super().__init__(0, 0)
    
    def _get_literals_str(self) -> list[str]:
        """
        Returns the string representation of the literals.
        """
        literals = []
        
        for i in range(MAX_NUM_VARS):
            if (self.pos & (1 << i)) > 0:
                literals.append(f"x{i}")
            if (self.neg & (1 << i)) > 0:
                literals.append(f"¬x{i}")
        
        return literals
    
    def __str__(self) -> str:
        """
        Returns a string representation of the literals, joined by `.` (e.g., `x0.¬x1`).
        """
        return '.'.join(self._get_literals_str())

    def __repr__(self) -> str:
        """
        Returns the string representation of the Literals object.
        """
        pos_str = ','.join(str(i) for i in range(MAX_NUM_VARS) if (self.pos & (1 << i)) > 0)
        neg_str = ','.join(str(i) for i in range(MAX_NUM_VARS) if (self.neg & (1 << i)) > 0)
        return f"Literals(pos=[{pos_str}], neg=[{neg_str}])"

    def __int__(self) -> int:
        """
        Returns the integer representation of the Literals object.
        """
        return self.pos + self.neg * (2 ** MAX_NUM_VARS)

    def to_dict(self) -> dict[str, list[int]]:
        """
        Returns a dictionary representation of the literals, with keys `"pos"` and `"neg"`.
        """
        return {
            "pos": [i for i in range(MAX_NUM_VARS) if (self.pos & (1 << i)) > 0],
            "neg": [i for i in range(MAX_NUM_VARS) if (self.neg & (1 << i)) > 0],
        }

    def __hash__(self) -> int:
        """
        Returns the hash value of the Literals object.
        """
        return hash((self.pos, self.neg))

    def __eq__(self, other: object) -> bool:
        """
        Checks for equality with another `Literals` object.
        """
        if not isinstance(other, Literals):
            return False
        return self.pos == other.pos and self.neg == other.neg
    
    def __lt__(self, other: object) -> bool:
        """
        Compares with another `Literals` object for sorting.
        """
        if not isinstance(other, Literals):
            raise LongshotError("the argument `other` is not a Literals object.")   
        return (self.pos, self.neg) < (other.pos, other.neg)

    @property
    def is_empty(self) -> bool:
        """
        Check if the literals set is empty (no literals).
        
        Returns:
            bool: True if there are no positive or negative literals, False otherwise.
            
        Example:
            >>> Literals().is_empty  # Empty literals
            True
            >>> Literals(pos=[0]).is_empty  # Has x0
            False
        """
        return super().is_empty
    
    @property
    def is_contradictory(self) -> bool:
        """
        Check if the literals set contains contradictions.
        
        A contradiction occurs when the same variable appears as both positive 
        and negative (e.g., x0 and ¬x0).
        
        Returns:
            bool: True if there are contradictory literals, False otherwise.
            
        Example:
            >>> Literals(pos=[0], neg=[0]).is_contradictory  # x0 and ¬x0
            True
            >>> Literals(pos=[0], neg=[1]).is_contradictory  # x0 and ¬x1
            False
        """
        return super().is_contradictory
    
    @property
    def is_constant(self) -> bool:
        """
        Check if the literals represent a constant (always true or always false).
        
        A literals set is constant if it's either empty or contains contradictions.
        Empty literals evaluate to true in CNF (empty conjunction) or false in DNF (empty disjunction).
        Contradictory literals always evaluate to a constant regardless of variable assignments.
        
        Returns:
            bool: True if the literals represent a constant value, False otherwise.
            
        Example:
            >>> Literals().is_constant  # Empty is constant
            True
            >>> Literals(pos=[0], neg=[0]).is_constant  # Contradiction is constant
            True
            >>> Literals(pos=[0]).is_constant  # x0 is not constant
            False
        """
        return super().is_constant
    
    @property
    def width(self) -> int:
        """
        Get the width (number of literals) in this set.
        
        Width is the total count of distinct literals (both positive and negative).
        Constant literals (empty or contradictory) have width 0.
        
        Returns:
            int: The number of literals in the set, or 0 if constant.
            
        Example:
            >>> Literals(pos=[0, 1], neg=[2]).width  # x0, x1, ¬x2
            3
            >>> Literals().width  # Empty has width 0
            0
            >>> Literals(pos=[0], neg=[0]).width  # Contradiction has width 0
            0
        """
        return super().width
    
    @property
    def pos(self) -> int:
        """
        Get the bitmask representing positive literals.
        
        Each bit position corresponds to a variable index. Bit i is set if variable i 
        appears as a positive literal (xi).
        
        Returns:
            int: Bitmask of positive literals.
            
        Example:
            >>> Literals(pos=[0, 2]).pos  # x0 and x2
            5  # Binary: 0101
        """
        return super().pos
    
    @property
    def neg(self) -> int:
        """
        Get the bitmask representing negative literals.
        
        Each bit position corresponds to a variable index. Bit i is set if variable i 
        appears as a negative literal (¬xi).
        
        Returns:
            int: Bitmask of negative literals.
            
        Example:
            >>> Literals(neg=[1, 3]).neg  # ¬x1 and ¬x3
            10  # Binary: 1010
        """
        return super().neg

class Clause(Literals):
    """
    Inherits from `Literals`. Represents a clause, which is a disjunction (OR) of literals.
    """
    def __str__(self) -> str:
        """
        Overrides the base class method to return a string representation of the clause with literals joined by `∨` (OR symbol).
        """
        return '∨'.join(self._get_literals_str())
    
class Term(Literals):
    """
    Inherits from `Literals`. Represents a term, which is a conjunction (AND) of literals.
    """
    def __str__(self) -> str:
        """
        Overrides the base class method to return a string representation of the term with literals joined by `∧` (AND symbol).
        """
        return '∧'.join(self._get_literals_str())