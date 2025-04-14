from typing import Optional
from collections.abc import Iterable

from .._core.circuit import _AC0_Circuit, _Clause, _NormalFormFormulaType, _NormalFormFormula
from ..error import LongshotError

MAX_NUM_VARS = 24

class AC0_Circuit(_AC0_Circuit):
    pass
        
class FormulaType(_NormalFormFormulaType):
    pass
        
class Clause(_Clause):
    """
    A class representing a clause.
    """
    def __init__(self, 
                 pos_vars: int | None = None,
                 neg_vars: int | None = None,
                 d_clause: dict[str, tuple[int, ...]] | None = None, 
                 ftype: FormulaType | None = None):
        """
        Initializes a Clause object.
        :param pos_vars: Positive variables.
        :param neg_vars: Negative variables.
        :param d_clause: Dictionary of clauses.
        :param ftype: Formula type.
        """
        
        if ftype is not None:
            if not isinstance(ftype, FormulaType):
                raise LongshotError("the argument `ftype` is not a FormulaType.")
            self.ftype = ftype
        else:
            self.ftype = None
            
        if pos_vars is not None or neg_vars is not None:
            pos_vars = pos_vars if pos_vars is not None else 0
            neg_vars = neg_vars if neg_vars is not None else 0
            
            if isinstance(pos_vars, int) and isinstance(neg_vars, int):
                super().__init__(pos_vars, neg_vars)
            else:
                raise LongshotError("the arguments `pos_vars` and `neg_vars` should be integers.")
        else:
            super().__init__(0, 0)
        
        if not isinstance(d_clause, dict):
            raise LongshotError("the argument `d_clause` is not a dictionary.")
        
        pos_vars = d_clause.get("pos_vars") if d_clause else None
        neg_vars = d_clause.get("neg_vars") if d_clause else None
        
        if not isinstance(pos_vars, Iterable) and pos_vars is not None:
            raise LongshotError("`d_clause['pos_vars']` should be iterable.")
        if not isinstance(neg_vars, Iterable) and neg_vars is not None:
            raise LongshotError("`d_clause['neg_vars']` should be iterable.")
        
        if pos_vars is not None:
            for i in pos_vars:
                if not isinstance(i, int) or i < 0 or i >= MAX_NUM_VARS:
                    raise LongshotError(f"`d_clause['pos_vars']` should be a list of integers between 0 and {MAX_NUM_VARS - 1}.")
                self.pos_vars |= (1 << i)
        if neg_vars is not None:
            for i in neg_vars:
                if not isinstance(i, int) or i < 0 or i >= MAX_NUM_VARS:
                    raise LongshotError(f"`d_clause['neg_vars']` should be a list of integers between 0 and {MAX_NUM_VARS - 1}.")
                self.neg_vars |= (1 << i)
        
    def __dict__(self) -> dict:
        """
        Returns the dictionary representation of the Clause object.self.
        """
        return {
            "pos_vars": (i for i in range(MAX_NUM_VARS) if (self.pos_vars & (1 << i)) > 0),
            "neg_vars": (i for i in range(MAX_NUM_VARS) if (self.neg_vars & (1 << i)) > 0),
        }
    
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
                op = "AND"
            elif self.ftype == FormulaType.Disjunctive:
                op = "OR"
            else:
                raise LongshotError("Invalid formula type.")
        else:
            op = "?"
        
        return op.join(literals)
        
        
class NormalFormFormula(_NormalFormFormula):
    """
    A class representing a normal form formula.
    """
    def __init__(self, num_vars: int, ftype: Optional[FormulaType] = FormulaType.Conjunctive):
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
    
    def avgQ() -> float:
        """
        Returns the average-case deterministic query complexity of the formula.
        """
        return super().avgQ()
    
    def __str__(self) -> str:
        """
        Returns the string representation of the formula.
        """
        clauses_str = [f"({str(Clause(cl.pos_vars, cl.neg_vars, ftype=self.ftype))})" for cl in self.clauses]
        op = " AND " if self.ftype == FormulaType.Conjunctive else " OR "
        
        return op.join(clauses_str)
        