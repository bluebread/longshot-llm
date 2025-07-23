# Longshot API Documentation

This document provides a detailed overview of the classes and methods in `formula.py`.

## Class `Literals`

Represents a set of literals (e.g., `x1`, `¬x2`). It serves as a base class for `Clause` and `Term`.

### Methods

-   `__init__(self, pos=None, neg=None, d_literals=None)`: Initializes a `Literals` object. Can be initialized with positive and negative variable indices or a dictionary.
-   `to_dict(self)`: Returns a dictionary representation of the literals, with keys `"pos"` and `"neg"`.
-   `__str__(self)`: Returns a string representation of the literals, joined by `.` (e.g., `x0.¬x1`).
-   `__repr__(self)`: Returns a developer-friendly string representation.
-   `__int__(self)`: Returns an integer representation of the literals.
-   `__hash__(self)`: Computes the hash of the object.
-   `__eq__(self, other)`: Checks for equality with another `Literals` object.
-   `__lt__(self, other)`: Compares with another `Literals` object for sorting.

## Class `Clause`

Inherits from `Literals`. Represents a clause, which is a disjunction (OR) of literals.

### Methods

-   `__str__(self)`: Overrides the base class method to return a string representation of the clause with literals joined by `∨` (OR symbol).

## Class `Term`

Inherits from `Literals`. Represents a term, which is a conjunction (AND) of literals.

### Methods

-   `__str__(self)`: Overrides the base class method to return a string representation of the term with literals joined by `∧` (AND symbol).

## Class `DecisionTree`

Represents a binary decision tree for a boolean formula.

### Methods

-   `__init__(self, ctree=None, root=None)`: Initializes a `DecisionTree`. It can be built from a C++ decision tree object (`_CppDecisionTree`) or a given root `Node`.
-   `decide(self, x)`: Evaluates the decision tree for a given input assignment `x` and returns the boolean result.

### Properties

-   `root`: The root `Node` of the decision tree.

## Enum `FormulaType`

An enumeration for the type of a `NormalFormFormula`.

### Members

-   `Conjunctive`: Represents a Conjunctive Normal Form (CNF) formula.
-   `Disjunctive`: Represents a Disjunctive Normal Form (DNF) formula.

## Class `NormalFormFormula`

Represents a boolean formula in either Conjunctive Normal Form (CNF) or Disjunctive Normal Form (DNF).

### Methods

-   `__init__(self, num_vars, ftype=FormulaType.Conjunctive, device=None, **kwargs)`: Initializes a formula with a given number of variables and a formula type.
-   `copy(self)`: Creates a deep copy of the formula object.
-   `__contains__(self, ls)`: Checks if a given `Literals` object (clause or term) is part of the formula.
-   `toggle(self, ls)`: Adds a clause/term to the formula if it's not present, or removes it if it is.
-   `eval(self, x)`: Evaluates the formula for a given input assignment `x` and returns the boolean result.
-   `avgQ(self, build_tree=False)`: Calculates the average-case deterministic query complexity (AvgQ). If `build_tree` is `True`, it also returns the corresponding `DecisionTree`.
-   `wl_graph_hash(self, iterations=None)`: Computes the Weisfeiler-Lehman graph hash for the formula's graph representation.
-   `is_isomorphic(cls, F1, F2)`: A class method to check if two formulas `F1` and `F2` are isomorphic (structurally identical).
-   `__str__(self)`: Returns a human-readable string representation of the formula.

### Properties

-   `num_vars`: The number of variables in the formula.
-   `num_gates`: The number of gates (clauses or terms) in the formula.
-   `width`: The maximum width (number of literals) of any gate in the formula.
-   `ftype`: The `FormulaType` of the formula (CNF or DNF).
-   `gates`: The set of gates (`Literals` objects) in the formula.
-   `graph`: A `networkx.Graph` representation of the formula.

## Class `ConjunctiveNormalFormFormula`

A convenience class that inherits from `NormalFormFormula` and is specialized for CNF formulas.

### Methods

-   `__init__(self, n, **config)`: Initializes a CNF formula with `n` variables.

## Class `DisjunctiveNormalFormFormula`

A convenience class that inherits from `NormalFormFormula` and is specialized for DNF formulas.

### Methods

-   `__init__(self, n, **config)`: Initializes a DNF formula with `n` variables.


