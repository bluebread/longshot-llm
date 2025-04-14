from longshot.circuit import Clause, NormalFormFormula, FormulaType

def test_clause():
    c1 = Clause(pos_vars=0b0101, neg_vars=0b1010, ftype=FormulaType.Disjunctive)
    c2 = Clause(pos_vars=[3], neg_vars=[0, 1], ftype=FormulaType.Conjunctive)
    c3 = Clause(d_clause={0: True, 2: False})
    
    assert str(c1) == "x0∧¬x1∧x2∧¬x3"
    assert str(c2) == "¬x0∨¬x1∨x3"
    assert str(c3) == "x0.¬x2"

def test_cnf():
    # Test with a conjunctive normal form (CNF)
    cnf = NormalFormFormula(4, ftype=FormulaType.Conjunctive)
    cnf.add_clause(Clause(pos_vars=0b0101, neg_vars=0b1010))
    cnf.add_clause(Clause(pos_vars=[3], neg_vars=[0, 1]))
    
    print(str(cnf))
    assert str(cnf) == "(x0∨¬x1∨x2∨¬x3)∧(¬x0∨¬x1∨x3)"
    
    assert cnf.eval(0b0000) == True
    assert cnf.eval(0b1010) == False
    assert cnf.eval(0b0111) == False
    
    assert cnf.avgQ() == 2.125 # this number has been checked manually
    
def test_dnf():
    # Test with a disjunctive normal form (DNF)
    dnf = NormalFormFormula(5, ftype=FormulaType.Disjunctive)
    dnf.add_clause(Clause(pos_vars=0b10101, neg_vars=0b1010))
    dnf.add_clause(Clause(pos_vars=[3], neg_vars=[0, 1]))
    
    assert str(dnf) == "(x0∧¬x1∧x2∧¬x3∧x4)∨(¬x0∧¬x1∧x3)"
    
    assert dnf.eval(0b10101) == True
    assert dnf.eval(0b00000) == False
    assert dnf.eval(0b11111) == False
    
    assert dnf.avgQ() == 2.1875 # this number has been checked manually
    
if __name__ == "__main__":
    test_clause()
    test_cnf()
    test_dnf()
