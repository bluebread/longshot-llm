import pytest
import sys

from longshot.literals import (
    Literals,
    Clause,
    Term
)
from longshot.formula import (
    NormalFormFormula,
    FormulaType,
    CNF,
    DNF
)

def test_literals():
    c1 = Literals(pos=0b0101, neg=0b1010)
    c2 = Clause(pos=[3], neg=[0, 1])
    c3 = Term(d_literals={0: True, 2: False})
    
    assert str(c1) == "x0.¬x1.x2.¬x3"
    assert str(c2) == "¬x0∨¬x1∨x3"
    assert str(c3) == "x0∧¬x2"

def test_cnf():
    # Test with a conjunctive normal form (CNF)
    cnf = CNF(4)
    cnf.toggle(Clause(pos=0b0101, neg=0b1010))
    cnf.toggle(Clause(pos=[3], neg=[0, 1]))
    
    assert str(cnf) == "(x0∨¬x1∨x2∨¬x3)∧(¬x0∨¬x1∨x3)" or str(cnf) == "(¬x0∨¬x1∨x3)∧(x0∨¬x1∨x2∨¬x3)"
    
    assert cnf.eval(0b0000) == True
    assert cnf.eval(0b1010) == False
    assert cnf.eval(0b0111) == False
    
    assert cnf.avgQ() == 2.125 # this number has been checked manually
    
def test_dnf():
    # Test with a disjunctive normal form (DNF)
    dnf = DNF(5)
    dnf.toggle(Term(pos=0b10101, neg=0b1010))
    dnf.toggle(Term(pos=[3], neg=[0, 1]))
    
    assert str(dnf) == "(x0∧¬x1∧x2∧¬x3∧x4)∨(¬x0∧¬x1∧x3)" or str(dnf) == "(¬x0∧¬x1∧x3)∨(x0∧¬x1∧x2∧¬x3∧x4)"
    
    assert dnf.eval(0b10101) == True
    assert dnf.eval(0b00000) == False
    assert dnf.eval(0b11111) == False
    
    assert dnf.avgQ() == 2.1875 # this number has been checked manually
    
def test_tree():
    dnf = DNF(5)
    dnf.toggle(Term(pos=0b10101, neg=0b01010))
    dnf.toggle(Term(pos=[3], neg=[0, 1]))
    qv, tree = dnf.avgQ(build_tree=True)
    
    assert qv == 2.1875
    assert dnf.eval(0b10101) == tree.decide(0b10101)
    assert dnf.eval(0b00000) == tree.decide(0b00000)
    assert dnf.eval(0b11111) == tree.decide(0b11111)
    
def test_graph():
    f1 = DNF(5)
    f1.toggle(Term(pos=[0,2,4], neg=[1,3]))
    f1.toggle(Term(pos=[3], neg=[0, 1]))

    assert f1.graph.number_of_nodes() == 17  # 5 variables + 10 literals + 2 terms
    assert f1.graph.number_of_edges() == 18  # edges between literals and terms

    f2 = DNF(5)
    f2.toggle(Term(pos=[3,4], neg=[0,1,2]))
    f2.toggle(Term(pos=[1,4], neg=[3]))
    
    assert f2.graph.number_of_nodes() == 17  # 5 variables + 10 literals + 2 terms
    assert f2.graph.number_of_edges() == 18  # edges between literals and terms
    
    assert DNF.is_isomorphic(f1, f2) == True
    assert f1.wl_graph_hash() == 'a6d78fa057c14ea466a3ec99a545197b'
    assert f2.wl_graph_hash() == 'a6d78fa057c14ea466a3ec99a545197b'
    
    
if __name__ == "__main__":
    # pytest.main([__file__])
    # # test_literals()
    # # test_cnf()
    # # test_dnf()
    # # test_tree()
    test_graph()