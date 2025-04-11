from longshot import Clause, NormalFormFormula

dnf = NormalFormFormula(4)
dnf.add_clause(Clause({0: True, 2: False}))

for i in range(2**4):
    print(f"{format(i, '04b')}: {dnf.eval(i)}")