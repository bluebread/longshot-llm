import enum
from collections.abc import Iterable
import numpy as np
from numpy.typing import NDArray
from typing import Any
from sortedcontainers import SortedSet

from ..error import LongshotError
from .._core import (
    _Literals,
    _MonotonicBooleanFunction,
    _CountingBooleanFunction,
)

MAX_NUM_VARS = 24

class Literals(_Literals):
    """
    A class representing a set of _literals.
    """
    def __init__(
        self, 
        pos: int | Iterable[int] | None = None,
        neg: int | Iterable[int] | None = None,
        d_literals: dict[int, bool] | None = None,
        ):
        """
        Initializes a Literals object.
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
        Returns the string representation of the Clause object.
        """
        return '.'.join(self._get_literals_str())

    def __hash__(self) -> int:
        """
        Returns the hash value of the Literals object.
        """
        return hash((self.pos, self.neg))

    def __eq__(self, other: object) -> bool:
        """
        Checks if two Literals objects are equal.
        """
        if not isinstance(other, Literals):
            return False
        return self.pos == other.pos and self.neg == other.neg
    
    def __lt__(self, other: object) -> bool:
        """
        Checks if this Literals object is less than another.
        """
        if not isinstance(other, Literals):
            raise LongshotError("the argument `other` is not a Literals object.")   
        return (self.pos, self.neg) < (other.pos, other.neg)

    def vectorize(self, num_vars: int) -> NDArray[np.integer[Any]]:
        """
        Returns a vector representation of the Literals object.
        """
        if not isinstance(num_vars, int):
            raise LongshotError("the argument `num_vars` is not an integer.")
        if num_vars < 0 or num_vars > MAX_NUM_VARS:
            raise LongshotError(f"the argument `num_vars` should be between 0 and {MAX_NUM_VARS}.")
        
        vector = np.array([2] * num_vars, dtype=np.int8)
        
        for i in range(num_vars):
            if (self.pos & (1 << i)) > 0:
                vector[i] = 0
            if (self.neg & (1 << i)) > 0:
                vector[i] = 1
        
        return vector

class Clause(Literals):
    """
    A class representing a clause.
    """
    def __str__(self) -> str:
        """
        Returns the string representation of the Clause object.
        """
        return '∨'.join(self._get_literals_str())
    
class Term(Literals):
    def __str__(self) -> str:
        """
        Returns the string representation of the Term object.
        """
        return '∧'.join(self._get_literals_str())

class FormulaType(enum.IntEnum):
    """
    An enumeration representing the type of formula.
    """
    Conjunctive = 0
    Disjunctive = 1

class NormalFormFormula:
    """
    A class representing a normal form formula.
    """
    def __init__(
        self, 
        num_vars: int, 
        ftype: FormulaType | None = FormulaType.Conjunctive,
        mono: bool = False,
        ):
        if not isinstance(num_vars, int):
            raise LongshotError("the argument `num_vars` is not an integer.")
        if num_vars < 0 or num_vars > MAX_NUM_VARS:
            raise LongshotError(f"the argument `num_vars` should be between 0 and {MAX_NUM_VARS}.")
        if not isinstance(ftype, FormulaType):
            raise LongshotError("the argument `ftype` is not a FormulaType.")
        if not isinstance(mono, bool):
            raise LongshotError("the argument `mono` is not a boolean.")
        
        self._num_vars = num_vars
        self._ftype = ftype
        self._mono = mono
        self._literals = SortedSet()
        
        if self._mono:
            self._bf = _MonotonicBooleanFunction(num_vars)
        else:
            self._bf = _CountingBooleanFunction(num_vars)
        
        if ftype == FormulaType.Conjunctive:
            self._bf.as_cnf()
        if ftype == FormulaType.Disjunctive:
            self._bf.as_dnf()
    
    def copy(self):
        """
        Returns a copy of the formula.
        """
        # TODO: optimize the performance of the copy method
        cpy = NormalFormFormula(self._num_vars, self._ftype, self._mono)
        
        cpy._literals = self._literals.copy()
        
        if self._mono:
            cpy._bf = _MonotonicBooleanFunction(self._bf)
        else:
            cpy._bf = _CountingBooleanFunction(self._bf)
        
        return cpy
    
    def __contains__(self, ls: Literals | dict) -> bool:
        """
        Checks if the formula contains a clause.
        """
        if not isinstance(ls, Literals) and not isinstance(ls, dict):
            raise LongshotError("the argument `clause` is not a Clause or a dictionary.") 
        
        if self._ftype == FormulaType.Conjunctive and isinstance(ls, Term):
            raise LongshotError("the argument `ls` is not a Clause.")
        if self._ftype == FormulaType.Disjunctive and isinstance(ls, Clause):
            raise LongshotError("the argument `ls` is not a Term.")
        
        if isinstance(ls, dict):
            if "pos" not in ls or "neg" not in ls:
                raise LongshotError("the dictionary `clause` should contain 'pos' and 'neg' keys.")
            ls = Literals(d_literals=ls)
        
        return ls in self._literals        
    
    def __iter__(self) -> Iterable[Literals]:
        """
        Returns an iterator over the literals in the formula.
        """
        return iter(self._literals)
    
    def __tuple__(self) -> tuple[Literals, ...]:
        """
        Returns a tuple representation of the formula.
        """
        return tuple(self._literals)
    
    def __list__(self) -> list[Literals]:
        """
        Returns a list representation of the formula.
        """
        return list(self._literals)
    
    def add(self, ls: Literals | dict) -> None:
        """
        Adds a clause to the formula.
        """
        if not isinstance(ls, Literals) and not isinstance(ls, dict):
            raise LongshotError("the argument `clause` is not a Clause or a dictionary.") 
        
        if self._ftype == FormulaType.Conjunctive and isinstance(ls, Term):
            raise LongshotError("the argument `ls` is not a Clause.")
        if self._ftype == FormulaType.Disjunctive and isinstance(ls, Clause):
            raise LongshotError("the argument `ls` is not a Term.")
        
        if isinstance(ls, dict):
            if "pos" not in ls or "neg" not in ls:
                raise LongshotError("the dictionary `clause` should contain 'pos' and 'neg' keys.")
            ls = Literals(d_literals=ls)
                   
        if self._ftype == FormulaType.Conjunctive:
            self._bf.add_clause(ls)
        if self._ftype == FormulaType.Disjunctive:
            self._bf.add_term(ls)
        
        self._literals.add(ls)
    
    def remove(self, ls: Literals | dict) -> None:
        """
        Deletes a clause from the formula.
        """
        if self._mono:
            raise LongshotError("the formula is monotonic.")
        
        if not isinstance(ls, Literals) and not isinstance(ls, dict):
            raise LongshotError("the argument `clause` is not a Clause or a dictionary.") 
        
        if self._ftype == FormulaType.Conjunctive and isinstance(ls, Term):
            raise LongshotError("the argument `ls` is not a Clause.")
        if self._ftype == FormulaType.Disjunctive and isinstance(ls, Clause):
            raise LongshotError("the argument `ls` is not a Term.")
        
        if isinstance(ls, dict):
            if "pos" not in ls or "neg" not in ls:
                raise LongshotError("the dictionary `clause` should contain 'pos' and 'neg' keys.")
            ls = Literals(d_literals=ls)
        
        if self._ftype == FormulaType.Conjunctive:
            self._bf.del_clause(ls)
        if self._ftype == FormulaType.Disjunctive:
            self._bf.del_term(ls)
        
        self._literals.discard(ls)
    
    def eval(self, x: int | tuple[int | bool, ...]) -> bool:
        """
        Evaluates the formula.
        """
        if not isinstance(x, (int, tuple)):
            raise LongshotError("the argument `x` is not an integer.")
        
        if isinstance(x, tuple):
            if len(x) != self._num_vars:
                raise LongshotError(f"the length of the argument `x` should be {self._num_vars}.")
            x = sum((1 << i) if v else 0 for i, v in enumerate(x))
        
        if x < 0 or x >= (1 << self._num_vars):
            raise LongshotError(f"the argument `x` should be between 0 and {1 << self._num_vars - 1}.")
        
        return self._bf.eval(x)
    
    def avgQ(self) -> float:
        """
        Returns the average-case deterministic query complexity of the formula.
        """
        return self._bf.avgQ()
    
    def __str__(self) -> str:
        """
        Returns the string representation of the formula.
        """
        ls_list = [Literals(ls.pos, ls.neg) for ls in self._literals]
        
        if len(ls_list) == 0:
            if self._ftype == FormulaType.Conjunctive:
                return "<True>"
            elif self._ftype == FormulaType.Disjunctive:
                return "<False>"
            else:
                raise LongshotError("the formula type is not valid.")
        
        lop = "∧" if self._ftype == FormulaType.Disjunctive else "∨"
        fop = "∨" if self._ftype == FormulaType.Disjunctive else "∧"
        lstr = [f"({lop.join(ls._get_literals_str())})" for ls in ls_list]
        
        return fop.join(lstr)
        
    @property
    def is_mono(self) -> bool:
        """
        Returns True if the formula is monotonic, False otherwise.
        """
        return self._mono
        
    @property
    def num_vars(self) -> int:
        """
        Returns the number of variables in the formula.
        """
        return self._num_vars
        
    @property
    def ftype(self) -> FormulaType:
        """
        Returns the type of the formula.
        """
        return self._ftype
        
class ConjunctiveNormalFormFormula(NormalFormFormula):
    """
    A class representing a conjunctive normal form formula.
    """
    def __init__(self, num_vars: int, mono: bool = False):
        super().__init__(num_vars, ftype=FormulaType.Conjunctive, mono=mono)

class DisjunctiveNormalFormFormula(NormalFormFormula):
    """
    A class representing a disjunctive normal form formula.
    """
    def __init__(self, num_vars: int, mono: bool = False):
        super().__init__(num_vars, ftype=FormulaType.Disjunctive, mono=mono)