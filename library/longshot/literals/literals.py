import enum
from collections.abc import Iterable
import numpy as np
from binarytree import Node
from sortedcontainers import SortedSet
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from networkx.algorithms.isomorphism import vf2pp_is_isomorphic

from ..error import LongshotError
from .._core import (
    _Literals,
    _CountingBooleanFunction,
    _CppDecisionTree,
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