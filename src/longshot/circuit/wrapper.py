from .._core.circuit import (
    _AC0_Circuit,
    _Clause,
    _NormalFormFormulaType,
    _NormalFormFormula,
)

class AC0_Circuit(_AC0_Circuit):
    pass
        
class NormalFormFormulaType(_NormalFormFormulaType):
    pass
        
class Clause(_Clause):
    """
    A class representing a clause.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
class NormalFormFormula(_NormalFormFormula):
    """
    A class representing a normal form formula.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        

# def say_hello():
#     if _Clause is not None:
#         print("Hello from the Circuit module!")
#     else:
#         print("Hello from the Circuit module, but _Clause is not defined!")
        
# import inspect
# import sys

# # 1. grab the current module object
# this_mod = sys.modules[__name__]

# # 2. inspect its globals for functions
# user_funcs = [
#     name for name, obj in vars(this_mod).items()
# ]

# print(user_funcs)
