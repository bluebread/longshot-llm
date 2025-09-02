import enum
from collections.abc import Iterable
import numpy as np
from binarytree import Node
from sortedcontainers import SortedSet

from ..error import LongshotError
from ..literals import Literals, Term, Clause, MAX_NUM_VARS
from .._core import (
    _CountingBooleanFunction,
    _CppDecisionTree,
)


class DecisionTree:
    """
    Represents a binary decision tree for a boolean formula.
    """
    def __init__(self, ctree: _CppDecisionTree | None = None, root: Node | None = None):
        """
        Initializes a `DecisionTree`. It can be built from a C++ decision tree object (`_CppDecisionTree`) or a given root `Node`.
        """
        if not isinstance(ctree, _CppDecisionTree):
            raise LongshotError("the argument `ctree` is not a DecisionTree object.")
        
        if ctree is not None:
            self._root = self._recursive_build(ctree)
            ctree.delete()
        else:
            self._root = root
        
    def _recursive_build(self, ctree: _CppDecisionTree) -> None:
        """
        Recursively builds the decision tree.
        """
        if ctree.is_constant:
            return Node('T' if bool(ctree.var) else 'F')

        node = Node(ctree.var)        
        node.left = self._recursive_build(ctree.lt)
        node.right = self._recursive_build(ctree.rt)
        
        return node
        
    def decide(self, x: Iterable[int | bool]) -> bool:
        """
        Evaluates the decision tree for a given input assignment `x` and returns the boolean result.
        """
        if not isinstance(x, (int, Iterable)):
            raise LongshotError("the argument `x` is neither an integer nor an iterable.")
        if isinstance(x, int):
            x = [bool((x >> i) & 1) for i in range(MAX_NUM_VARS)]
        
        node = self._root
        
        while node.left is not None and node.right is not None:
            node = node.right if x[node.value] else node.left
        
        return node.value == 'T'
    
    @property
    def root(self) -> Node:
        """The root node of this decision tree."""
        return self._root

    @root.setter
    def root(self, new_root: Node) -> None:
        if not isinstance(new_root, Node):
            raise LongshotError("root must be set to a Node instance")
        self._root = new_root
        

class FormulaType(enum.IntEnum):
    """
    An enumeration for the type of a `NormalFormFormula`.
    """
    Conjunctive = 0
    Disjunctive = 1

class NormalFormFormula:
    """
    Represents a boolean formula in either Conjunctive Normal Form (CNF) or Disjunctive Normal Form (DNF).
    """
    def __init__(
        self, 
        num_vars: int, 
        ftype: FormulaType | None = FormulaType.Conjunctive,
        **kwargs
        ):
        """
        Initializes a formula with a given number of variables and a formula type.
        """
        if not isinstance(num_vars, int):
            raise LongshotError("the argument `num_vars` is not an integer.")
        if num_vars < 0 or num_vars > MAX_NUM_VARS:
            raise LongshotError(f"the argument `num_vars` should be between 0 and {MAX_NUM_VARS}.")
        if not isinstance(ftype, FormulaType):
            raise LongshotError("the argument `ftype` is not a FormulaType.")
        
        # Initialize the formula
        self._num_vars = num_vars
        self._ftype = ftype
        
        # Initialize the properties of the formula
        self._literals = SortedSet()
        self._bf = _CountingBooleanFunction(num_vars)
        
        # Convert the boolean function to the specified normal form
        if ftype == FormulaType.Conjunctive:
            self._bf.as_cnf()
        if ftype == FormulaType.Disjunctive:
            self._bf.as_dnf()
    
    def copy(self):
        """
        Creates a deep copy of the formula object.
        :return: A new NormalFormFormula object with the same properties.
        """
        cpy = NormalFormFormula(self._num_vars, self._ftype)
        
        cpy._literals = self._literals.copy()
        cpy._bf = _CountingBooleanFunction(self._bf)
        
        return cpy
    
    def __contains__(self, ls: Literals | dict) -> bool:
        """
        Checks if a given `Literals` object (clause or term) is part of the formula.
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
    
    def toggle(self, ls: Literals | dict) -> None:
        """
        Adds a clause/term to the formula if it's not present, or removes it if it is.
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
        
        if ls.is_constant:
            return  # No need to toggle constants
        
        if ls not in self._literals:
            # Apply the literals to the boolean function
            if self._ftype == FormulaType.Conjunctive:
                self._bf.add_clause(ls)
            if self._ftype == FormulaType.Disjunctive:
                self._bf.add_term(ls)
            
            self._literals.add(ls)
                
        else:
            # Remove the literals from the boolean function
            if self._ftype == FormulaType.Conjunctive:
                self._bf.del_clause(ls)
            if self._ftype == FormulaType.Disjunctive:
                self._bf.del_term(ls)
            
            self._literals.discard(ls)
    
    def eval(self, x: int | tuple[int | bool, ...]) -> bool:
        """
        Evaluates the formula for a given input assignment `x` and returns the boolean result.
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
    
    def avgQ(self, build_tree: bool = False) -> float | tuple[float, DecisionTree]:
        """
        Calculates the average-case deterministic query complexity (AvgQ). If `build_tree` is `True`, it also returns the corresponding `DecisionTree`.
        """
        ctree = _CppDecisionTree() if build_tree else None
        qv = self._bf.avgQ(ctree)
        
        if build_tree:
           return qv, DecisionTree(ctree)
       
        return qv
    
    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the formula.
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
    def num_vars(self) -> int:
        """
        The number of variables in the formula.
        """
        return self._num_vars
        
    @property
    def num_gates(self) -> int:
        """
        The number of gates (clauses or terms) in the formula.
        """
        return len(self._literals)

    @property
    def width(self) -> int:
        """
        The maximum width (number of literals) of any gate in the formula.
        """
        return max([ls.width for ls in self._literals], default=0)

    @property
    def ftype(self) -> FormulaType:
        """
        The `FormulaType` of the formula (CNF or DNF).
        """
        return self._ftype

    @property
    def gates(self) -> SortedSet[Literals]:
        """
        The set of gates (`Literals` objects) in the formula.
        """
        return self._literals.copy()
    
        
class ConjunctiveNormalFormFormula(NormalFormFormula):
    """
    A convenience class that inherits from `NormalFormFormula` and is specialized for CNF formulas.
    """
    def __init__(self, n: int, **config):
        """
        Initializes a `ConjunctiveNormalFormFormula` with the specified number of variables.
        """
        super().__init__(n, ftype=FormulaType.Conjunctive, **config)

class DisjunctiveNormalFormFormula(NormalFormFormula):
    """
    A convenience class that inherits from `NormalFormFormula` and is specialized for DNF formulas.
    """
    def __init__(self, n: int, **config):
        """
        Initializes a DNF formula with `n` variables.
        """
        super().__init__(n, ftype=FormulaType.Disjunctive, **config)