from longshot._core.circuit import _Clause, _NormalFormFormula

dnf = _NormalFormFormula(4)
dnf.add_clause(_Clause({0: True, 2: False}))

for i in range(2**4):
    print(f"{format(i, '04b')}: {dnf.eval(i)}")