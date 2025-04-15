import enum
from collections.abc import Iterable
import numpy as np
import warnings

from .._core.circuit import (
    _AC0_Circuit, 
    _Clause, 
    _NormalFormFormula,
    )
from .._core.circuit import _NormalFormFormulaType as FormulaType
from .._core.circuit import _AC0_Circuit as AC0_Circuit
from ..error import LongshotError

MAX_NUM_VARS = 24

# TODO: change the name to `Literals` and create two classes 
# `Clause` and `Term` inheriting from `Literals`
class Clause(_Clause):
    """
    A class representing a clause.
    """
    def __init__(self, 
                 pos_vars: int | Iterable[int] | None = None,
                 neg_vars: int | Iterable[int] | None = None,
                 d_clause: dict[int, bool] | None = None, 
                 ftype: FormulaType | None = None):
        """
        Initializes a Clause object.
        :param pos_vars: Positive variables.
        :param neg_vars: Negative variables.
        :param d_clause: Dictionary of clauses.
        :param ftype: Formula type.
        """
        
        if ftype is not None and not isinstance(ftype, FormulaType):
            raise LongshotError("the argument `ftype` is not a FormulaType.")
        
        self.ftype = ftype
            
        if pos_vars is not None or neg_vars is not None:
            pos_vars = 0 if pos_vars is None else pos_vars
            neg_vars = 0 if neg_vars is None else neg_vars
            
            if isinstance(pos_vars, Iterable):
                if not all(isinstance(i, (int, np.integer)) and i >= 0 and i < MAX_NUM_VARS for i in pos_vars):
                    raise LongshotError("the argument `pos_vars` is not a list of valid integers.")
                pos_vars = sum((1 << i) for i in pos_vars)
            if isinstance(neg_vars, Iterable):
                if not all(isinstance(i, (int, np.integer)) and i >= 0 and i < MAX_NUM_VARS for i in neg_vars):
                    raise LongshotError("the argument `neg_vars` is not a list of valid integers.")
                neg_vars = sum((1 << i) for i in neg_vars)
            
            if isinstance(pos_vars, (int, np.integer)) and isinstance(neg_vars, (int, np.integer)):
                super().__init__(pos_vars, neg_vars)
            else:
                raise LongshotError("the arguments `pos_vars` and `neg_vars` should be integers.")
        elif d_clause is not None:
            if not isinstance(d_clause, dict):
                raise LongshotError("the argument `d_clause` is not a dictionary.")
            
            pos_vars = neg_vars = 0 
            
            for k, v in d_clause.items():
                if not isinstance(k, (int, np.integer)) or k < 0 or k >= MAX_NUM_VARS:
                    raise LongshotError("the key of the dictionary `d_clause` is not a valid integer.")
                if not isinstance(v, bool):
                    raise LongshotError("the value of the dictionary `d_clause` is not a boolean.")
                
                if v:
                    pos_vars |= (1 << k)
                else:
                    neg_vars |= (1 << k)
                
            super().__init__(pos_vars, neg_vars)
        else:
            super().__init__(0, 0)
        
    def __str__(self) -> str:
        """
        Returns the string representation of the Clause object.
        """
        literals = []
        
        for i in range(MAX_NUM_VARS):
            if (self.pos_vars & (1 << i)) > 0:
                literals.append(f"x{i}")
            if (self.neg_vars & (1 << i)) > 0:
                literals.append(f"¬x{i}")
        
        if self.ftype is not None:
            if self.ftype == FormulaType.Conjunctive:
                op = "∨"
            elif self.ftype == FormulaType.Disjunctive:
                op = "∧"
            else:
                raise LongshotError("Invalid formula type.")
        else:
            op = "."
        
        return op.join(literals)
    
    def is_constant(self) -> bool:
        """
        Checks if the clause is a constant.
        """
        return (
            (self.pos_vars | self.neg_vars) == 0
            or (self.pos_vars & self.neg_vars) > 0
        )
    
    def width(self) -> int:
        """
        Returns the width of the clause.
        """
        return self.pos_vars.bit_count() + self.neg_vars.bit_count()
        
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
    
    def add_clause(self, clause: Clause | dict) -> None:
        """
        Adds a clause to the formula.
        """
        if not isinstance(clause, Clause) and not isinstance(clause, dict):
            raise LongshotError("the argument `clause` is not a Clause or a dictionary.") 
        
        if isinstance(clause, dict):
            if "pos_vars" not in clause or "neg_vars" not in clause:
                raise LongshotError("the dictionary `clause` should contain 'pos_vars' and 'neg_vars' keys.")
            clause = Clause(d_clause=clause)
                   
        super().add_clause(clause)
    
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
        clauses_str = [f"({str(Clause(cl.pos_vars, cl.neg_vars, ftype=self.ftype))})" for cl in self.clauses]
        op = "∧" if self.ftype == FormulaType.Conjunctive else "∨"
        
        return op.join(clauses_str)
        