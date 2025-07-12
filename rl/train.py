import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import ActorCriticOperator, ActorValueOperator, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import multiprocessing

from longshot.circuit import NormalFormFormula, Literals, CNF, DNF
from .distrib import GumbelTopK
from .model import LongshotModel
from .game import FormulaGame
from .utils import (
    random_formula_permutations,
    permute_tensor,
    inverse_permutation,
)

# Enviroment parameters
num_vars = 4  
width = 4  
size = 16  
eps = 0.1 
num_literal = 2 * num_vars  # number of literals in the formula, i.e. 2 * number of variables

# Hyperparameters
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

# Data collection parameters
frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 10_000

# PPO parameters
sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

# Transformer parameters
d_model = 256
nhead = 8
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 256
dropout = 0.1
layer_norm_eps = 1e-5
bias = True

# Other parameters
statis_freq = 2000  # number of samples to estimate the log probability and entropy

# Define the environment
base_env = FormulaGame(
    formula=DNF(num_vars),
    width=width,
    size=size,
    eps=eps,
    device=device
)
env = TransformedEnv(
    base_env,
    Compose(
        DoubleToFloat(),
        StepCounter(),
    ),
)

# Define the modules
model = LongshotModel(
    num_vars=num_vars,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    layer_norm_eps=layer_norm_eps,
    bias=bias,
    device=device
),
latent_module = TensorDictModule(
    model,
    in_keys=["action"],
    out_keys=["latent"],
)
action_head = ProbabilisticActor(
    module=torch.nn.Sequential(
        torch.nn.Linear(d_model, num_literal),
    ),
    in_keys=["latent"],
    out_keys=["phi"],
    distribution_class=GumbelTopK,
    distribution_kwargs={
        "k": width,  # number of top-k elements to select
        "statis_freq": statis_freq,  # number of samples to estimate the log probability and entropy
    }
)
value_head = ValueOperator(
    module=torch.nn.Sequential(
        torch.nn.Linear(d_model, 1)
    ),
    in_keys=["latent"],
)