from ..circuit import NormalFormFormula
from ..error import LongshotError
from ..models import GateToken

# TODO: write a static method to process avgQ sequence into reward sequence. 

class FormulaGame:
    """
    The FormulaGame class implements the RL environment that simulates the process of adding or deleting gates in a normal form formula.
    It calculates the average-case deterministic query complexity, which is the optimization target.
    """
    def __init__(self, formula: NormalFormFormula, **kwargs):
        """
        Initializes the FormulaGame with a given formula and configuration parameters.
        
        Args:
            formula (NormalFormFormula): The formula to be manipulated in the game.
            **kwargs: Additional configuration parameters such as width, size, and penalty.
        """
        self._num_vars = formula.num_vars
        self._width = kwargs.pop('width', self._num_vars)
        self._size = kwargs.pop('size', 2**self._num_vars)
        self._eps = kwargs.pop('eps', 1 / self._num_vars)
        self._kwargs = kwargs
        
        if formula is None or not isinstance(formula, NormalFormFormula):
            raise LongshotError("Formula must be an instance of NormalFormFormula.")
        if formula.num_gates > self._size:
            raise LongshotError(f"Formula has {formula.num_gates} gates greater than {self._size}.")
        if formula.width > self._width:
            raise LongshotError(f"Formula has width {formula.width} greater than {self._width}.")
        
        self._init_f = formula.copy()
        self._cur_f = formula.copy()
        self._cur_avgQ = self._init_avgQ = formula.avgQ()

        
    def step(self, token: GateToken) -> float:
        """
        Simulates a step in the formula game by applying the given token (which indicates adding or deleting a gate) to the formula.
        It returns the reward for this step, which is based on the average-case deterministic query complexity of the resulting formula.
        
        Args:
            token (GateToken): The token representing the gate operation.
        
        Returns:
            float: The reward received after applying the token, based on the average-case deterministic query complexity of the formula.
        """
        n = self._num_vars
        s = self._size
        w = self._width
        eps = self._eps
        ls = token.literals
        
        if ls.is_constant or ls.width > w or self._cur_f.num_gates >= s:
            reward = self._kwargs.get('penalty', -1.0)
        elif ls in self._cur_f and token.type == 'ADD':
            reward = self._kwargs.get('penalty', -1.0)
        elif ls not in self._cur_f and token.type == 'DELETE':
            reward = self._kwargs.get('penalty', -1.0)
        else:
            self._cur_f.toggle(ls)
            self._cur_avgQ = q = self._cur_f.avgQ()
            lmda = 1 / (1 - (q - eps) / n)
            reward = q + lmda
            
        return reward
        
    def reset(self, _ = None) -> None:
        """
        Resets the internal variables of the formula game. This method is called at the beginning of
        each episode to prepare the environment for a new game.
        
        Args:
            _ (optional): Placeholder for compatibility, not used in this implementation.
        """
        
        self._cur_f = self._init_f.copy()
        self._cur_avgQ = self._init_avgQ
        
    @property
    def cur_avgQ(self) -> float:
        """
        Returns the current average-case deterministic query complexity of the formula.
        
        Returns:
            float: The current average-case deterministic query complexity.
        """
        return self._cur_avgQ
