from .formula import NormalFormFormula
from ..error import LongshotError
from ..literals import Literals
from pydantic import BaseModel, ConfigDict, Field


class GateToken(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: str
    literals: Literals = Field(..., description="List of literals involved in the operation")
    
    @classmethod
    def dim_token(cls, num_vars: int) -> int:
        return 2 * num_vars + 3


class FormulaRewardModel:
    """
    The FormulaGame class implements the RL environment that simulates the process of adding or deleting gates in a normal form formula.
    It calculates the average-case deterministic query complexity, which is the optimization target.
    """
    def __init__(self, formula: NormalFormFormula, **kwargs):
        """
        Initializes the FormulaGame with a given formula and configuration parameters.
        
        Args:
            formula (NormalFormFormula): The formula to be manipulated in the game.
            **kwargs: Additional configuration parameters:
                - width: Maximum width constraint (default: num_vars)
                - size: Maximum formula size constraint (default: None, meaning no limit)
                - eps: Small epsilon value for reward calculation (default: 1/num_vars)
                - penalty: Penalty for invalid operations (default: -1.0)
        """
        if formula is None or not isinstance(formula, NormalFormFormula):
            raise LongshotError("Formula must be an instance of NormalFormFormula.")
            
        self._num_vars = formula.num_vars
        self._width = kwargs.pop('width', self._num_vars)
        self._size = kwargs.pop('size', None)  # None means no size constraint
        self._eps = kwargs.pop('eps', 1 / self._num_vars)
        self._kwargs = kwargs
        if self._size is not None and formula.num_gates > self._size:
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
        
        # Check constraints: constant literals, width violation, or size violation (if size limit exists)
        if ls.is_constant or ls.width > w:
            reward = self._kwargs.get('penalty', -1.0)
        elif s is not None and self._cur_f.num_gates >= s:
            reward = self._kwargs.get('penalty', -1.0)
        elif ls in self._cur_f and token.type == 'ADD':
            reward = self._kwargs.get('penalty', -1.0)
        elif ls not in self._cur_f and token.type == 'DEL':
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
    
    @property
    def gates(self) -> set:
        """
        Returns the current set of gates in the formula.
        
        Returns:
            set: Set of gate integers currently in the formula.
        """
        # Convert the formula's gates to a set of integers
        return {int(lit) for lit in self._cur_f.gates}


    