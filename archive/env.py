from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn
import torch.nn.functional as F
import torch
from torchrl.envs import EnvBase
from torchrl.data import Composite, Binary, Bounded, UnboundedContinuous
from longshot.circuit import NormalFormFormula, Literals, CNF, DNF
from longshot.error import LongshotError

class FormulaGame(EnvBase):
    metadata = {}
    batch_locked = False
    
    def __init__(self, formula: NormalFormFormula, *args, **kwargs):
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
        self._make_spec()
        super().__init__(*args, **kwargs)
    
    def _set_seed(self, seed: int = None) -> None:
        rng = torch.manual_seed(seed)
        self.rng = rng
        
    def _make_spec(self):
        n = self._num_vars
        d = 2 * n
        s = self._size
        self.observation_spec = Composite(
            sequence=Binary(d, shape=(s, d)),
            length=Bounded(low=0, high=s+1, shape=(1,), dtype=torch.int32),
            avgQ=UnboundedContinuous(shape=(1,), dtype=torch.float32),
        )
        self.action_spec = Binary(d)
        self.state_spec = self.observation_spec.clone()
        self.reward_spec = UnboundedContinuous(shape=(1,), dtype=torch.float32)
        
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
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
                "avgQ": self._cur_avgQ,
                "sequence": seq_t,
                "length": torch.tensor(self._cur_f.num_gates, dtype=torch.int32),
                "reward": torch.tensor(reward, dtype=torch.float32),
                "done": torch.tensor(False, dtype=torch.bool),
            },
        )
        
    def _reset(self, _ = None) -> TensorDictBase:
        self._cur_f = self._init_f.copy()
        self._cur_avgQ = self._init_avgQ
        pd = self._size - self._cur_f.num_gates
        seq_t = self._cur_f.tensor
        
        if pd > 0:
            seq_t = F.pad(seq_t, (0, 0, 0, pd), mode='constant', value=0)
            
        return TensorDict(
            {
                "avgQ": self._cur_avgQ,
                "sequence": seq_t,
                "length": torch.tensor(self._cur_f.num_gates, dtype=torch.int32),
            },
        )
        
if __name__ == "__main__":
    env = FormulaGame(
        formula=CNF(5),
        width=3,
        size=8,
        eps=0.1,
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