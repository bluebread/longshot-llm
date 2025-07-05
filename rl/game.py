from tensordict import TensorDict, TensorDictBase
from torch import nn
import torch.nn.functional as F
import torch
from torchrl.envs import EnvBase
from torchrl.data import Composite, Binary, Bounded, UnboundedContinuous
from longshot.circuit import NormalFormFormula, Literals, CNF, DNF
from longshot.error import LongshotError

class FormulaGame(EnvBase):
    """
    A reinforcement learning environment for manipulating logical formulas in CNF or DNF form.
    This environment allows agents to interact with a logical formula, modifying it by toggling literals
    and receiving rewards based on the average-case query complexity (avgQ) of the formula.
    """
    metadata = {}
    batch_locked = False
    
    def __init__(self, formula: NormalFormFormula, device=None, *args, **kwargs):
        """
        Initializes the FormulaGame environment with a given logical formula.
        Args:
            formula (NormalFormFormula): The logical formula to manipulate, must be in CNF or DNF form.
            device (torch.device, optional): The device on which the environment will run. Defaults to None, which uses the CPU.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments, including:
                - width (int): Maximum width of the formula.
                - size (int): Maximum number of gates in the formula.
                - eps (float): Small epsilon value for reward calculation.
        Raises:
            LongshotError: If the provided formula is not an instance of NormalFormFormula,
                           or if the formula's number of gates exceeds the specified size,
                           or if the formula's width exceeds the specified width.
        """
        self._num_vars = formula.num_vars
        self._device = device if device is not None else torch.device('cpu')
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
        self._make_spec()
        super().__init__(*args, **kwargs)
    
    def _set_seed(self, seed: int = None) -> None:
        """
        Sets the random seed for reproducibility.
        Args:
            seed (int): The random seed to set. If None, uses the default behavior of torch.
        """
        rng = torch.manual_seed(seed)
        self.rng = rng
        
    def _make_spec(self):
        """        
        Creates the observation, action, state, and reward specifications for the environment.
        The observation includes a sequence of binary values representing the formula,
        the length of the sequence, and the average-case query complexity (avgQ) of the formula.
        The action is a binary vector representing the toggling of literals in the formula.
        The state is a copy of the observation specification, and the reward is an unbounded 
        continuous value representing the reward received after taking an action.
        """
        n = self._num_vars
        d = 2 * n
        s = self._size
        self.observation_spec = Composite(
            sequence=Binary(d, shape=(s, d), dtype=torch.int8, device=self._device),
            length=Bounded(low=0, high=s+1, shape=(1,), dtype=torch.int32, device=self._device),
            avgQ=UnboundedContinuous(shape=(1,), dtype=torch.float32, device=self._device),
            device=self._device,
        )
        self.action_spec = Binary(d, device=self._device)
        self.state_spec = self.observation_spec.clone()
        self.reward_spec = UnboundedContinuous(shape=(1,), dtype=torch.float32, device=self._device)
        
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """        
        Takes a step in the environment based on the provided action.
        This method modifies the current formula by toggling literals based on the action taken,
        calculates the average-case query complexity (avgQ) of the modified formula, and computes the reward.
        If the action exceeds the width or size constraints, a penalty is applied instead of 
        modifying the formula.
        
        Args:
            tensordict (TensorDictBase): The input tensor dictionary containing the action to be taken.
        Returns:
            TensorDictBase: A tensor dictionary containing the next state, reward, and other relevant information.
        """
        n = self._num_vars
        s = self._size
        w = self._width
        eps = self._eps
        pos = [i for i,v in enumerate(tensordict["action"][:n]) if v == 1]
        neg = [i for i,v in enumerate(tensordict["action"][n:]) if v == 1]
        ls = Literals(pos, neg)
        
        if ls.is_constant or ls.width > w or self._cur_f.num_gates >= s:
            reward = self._kwargs.get('penalty', -1.0)
        else:
            self._cur_f.toggle(ls)
            self._cur_avgQ = q = self._cur_f.avgQ()
            lmda = 1 / (1 - (q - eps) / n)
            reward = q + lmda
            
        pd = s - self._cur_f.num_gates
        seq_t = self._cur_f.tensor
        
        if pd > 0:
            seq_t = F.pad(seq_t, (0, 0, 0, pd), mode='constant', value=0)
        
        return TensorDict(
            {
                "avgQ": torch.tensor([self._cur_avgQ], dtype=torch.float32, device=self._device),
                "sequence": seq_t,
                "length": torch.tensor([self._cur_f.num_gates], dtype=torch.int32, device=self._device),
                "reward": torch.tensor([reward], dtype=torch.float32, device=self._device),
                "done": torch.tensor([False], dtype=torch.bool, device=self._device),
            },
            device=self._device,
        )
        
    def _reset(self, _ = None) -> TensorDictBase:
        """        
        Resets the environment to its initial state.
        This method restores the current formula to its initial state and resets the average-case query complexity (avgQ).
        It also prepares the sequence tensor by padding it to the specified size.
        
        Args:
            _ (None): Placeholder for compatibility, not used in this method.
        Returns:
            TensorDictBase: A tensor dictionary containing the initial state of the environment,
                            including the average-case query complexity (avgQ), sequence tensor, 
                            and length of the sequence.
        """
        self._cur_f = self._init_f.copy()
        self._cur_avgQ = self._init_avgQ
        pd = self._size - self._cur_f.num_gates
        seq_t = self._cur_f.tensor
        
        if pd > 0:
            seq_t = F.pad(seq_t, (0, 0, 0, pd), mode='constant', value=0)
            
        return TensorDict(
            {
                "avgQ": torch.tensor([self._cur_avgQ], dtype=torch.float32, device=self._device),
                "sequence": seq_t,
                "length": torch.tensor([self._cur_f.num_gates], dtype=torch.int32, device=self._device),
            },
            device=self._device,
        )
        
if __name__ == "__main__":
    env = FormulaGame(
        formula=CNF(5),
        width=3,
        size=8,
        eps=0.1,
        device=torch.device('cuda')  # Specify the device if needed
    )
    
    print(f"{'='*20} RESET {'='*20}")
    td = env.reset()
    print(td)
    print('- done', td['done'])
    print('- sequence', td['sequence'])
    print('- length', td['length'])
    print('- terminated', td['terminated'])
    print(f"{'='*20} RANDOM ACTION {'='*20}")
    a = env.action_spec.rand()
    print(a)
    td['action'] = a
    
    print(f"{'='*20} STEP {'='*20}")
    td = env.step(td)
    print(td)
    print('- done', td['next']['done'])
    print('- sequence', td['next']['sequence'])
    print('- length', td['next']['length'])
    print('- reward', td['next']['reward'])
    print('- avgQ', td['next']['avgQ'])
    
    print(f"{'='*20} RANDOM STEP 2 {'='*20}")
    td = env.rand_step(td)
    print(td)
    print('- action:', td['action'])
    print('- done:', td['next']['done'])
    print('- sequence:', td['next']['sequence'])
    print('- length:', td['next']['length'])
    print('- reward:', td['next']['reward'])
    print('- avgQ:', td['next']['avgQ'])
    
    print(f"{'='*20} ROLLOUT (5 steps) {'='*20}")
    td = env.rollout(max_steps=5)
    print(td)
    print('- action:', td['action'])
    print('- done:', td['next']['done'])
    print('- sequence:', td['next']['sequence'])
    print('- length:', td['next']['length'])
    print('- reward:', td['next']['reward'])
    print('- avgQ:', td['next']['avgQ'])