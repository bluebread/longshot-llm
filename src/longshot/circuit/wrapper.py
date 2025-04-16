import enum
from collections.abc import Iterable
import numpy as np
import warnings

from .._core.circuit import (
    _Literals, 
    _NormalFormFormula,
    )
from .._core.circuit import _NormalFormFormulaType as FormulaType
from .._core.circuit import _AC0_Circuit as AC0_Circuit
from ..error import LongshotError

MAX_NUM_VARS = 24

class Literals(_Literals):
    """
    A class representing a set of literals.
    """
    def __init__(self, 
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
    
    def __str__(self, op='.') -> str:
        """
        Returns the string representation of the Clause object.
        """
        return op.join(self._get_literals_str())

class Clause(Literals):
    """
    A class representing a clause.
    """
    def __str__(self) -> str:
        """
        Returns the string representation of the Clause object.
        """
        return super().__str__(op='∨')
    
class Term(Literals):
    def __str__(self) -> str:
        """
        Returns the string representation of the Term object.
        """
        return super().__str__(op='∧')

class NormalFormFormula(_NormalFormFormula):
    """
    A class representing a normal form formula.
    """
    def __init__(self, num_vars: int, ftype: FormulaType | None = FormulaType.Conjunctive):
        if not isinstance(num_vars, int):
            raise LongshotError("the argument `num_vars` is not an integer.")
        if num_vars < 0 or num_vars > MAX_NUM_VARS:
            raise LongshotError(f"the argument `num_vars` should be between 0 and {MAX_NUM_VARS}.")
        if not isinstance(ftype, FormulaType):
            raise LongshotError("the argument `ftype` is not a FormulaType.")
        
        super().__init__(num_vars, ftype)
    
    def add(self, ls: Literals | dict) -> None:
        """
        Adds a clause to the formula.
        """
        if not isinstance(ls, Literals) and not isinstance(ls, dict):
            raise LongshotError("the argument `clause` is not a Clause or a dictionary.") 
        
        if self.ftype == FormulaType.Conjunctive and isinstance(ls, Term):
            raise LongshotError("the argument `ls` is not a Clause.")
        if self.ftype == FormulaType.Disjunctive and isinstance(ls, Clause):
            raise LongshotError("the argument `ls` is not a Term.")
        
        if isinstance(ls, dict):
            if "pos" not in ls or "neg" not in ls:
                raise LongshotError("the dictionary `clause` should contain 'pos' and 'neg' keys.")
            ls = Literals(d_literals=ls)
                   
        super().add(ls)
    
    def eval(self, x: int | tuple[int | bool, ...]) -> bool:
        """
        Evaluates the formula.
        """
        if not isinstance(x, (int, tuple)):
            raise LongshotError("the argument `x` is not an integer.")
        
        if isinstance(x, tuple):
            if len(x) != self.num_vars:
                raise LongshotError(f"the length of the argument `x` should be {self.num_vars}.")
            x = sum((1 << i) if v else 0 for i, v in enumerate(x))
        
        if x < 0 or x >= (1 << self.num_vars):
            raise LongshotError(f"the argument `x` should be between 0 and {1 << self.num_vars - 1}.")
        
        return super().eval(x)
    
    def avgQ(self) -> float:
        """
        Returns the average-case deterministic query complexity of the formula.
        """
        return super().avgQ()
    
    def __str__(self) -> str:
        """
        Returns the string representation of the formula.
        """
        ls_list = [Literals(ls.pos, ls.neg) for ls in self.literals]
        
        if len(ls_list) == 0:
            if self.ftype == FormulaType.Conjunctive:
                return "<True>"
            elif self.ftype == FormulaType.Disjunctive:
                return "<False>"
            else:
                raise LongshotError("the formula type is not valid.")
        
        lop = "∧" if self.ftype == FormulaType.Disjunctive else "∨"
        fop = "∨" if self.ftype == FormulaType.Disjunctive else "∧"
        lstr = [f"({lop.join(ls._get_literals_str())})" for ls in ls_list]
        
        return fop.join(lstr)
        