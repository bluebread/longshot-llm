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
from torchrl.data.replay_buffers import (
    LazyMemmapStorage, 
    TensorDictReplayBuffer,
    SamplerWithoutReplacement,
)
from torchrl.modules import ActorCriticOperator, ActorValueOperator, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
import tempfile
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

# Replay buffer parameters
buffer_size = 20  # maximum size of the replay buffer
batch_size = 5  # batch size for sampling from the replay buffer

# GAE parameters
gamma_gae = 0.98  # discount factor for GAE
lmbda_gae = 0.95  # lambda for GAE

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

# Define replay buffer
tmpdir = tempfile.TemporaryDirectory()
buffer = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(max_size=buffer_size, scratch_dir=tmpdir), 
    sampler=SamplerWithoutReplacement(),
    batch_size=batch_size,
)
# DO NOT forget to clean up the temporary directory after creating the buffer
# # tmpdir.cleanup()  # Clean up the temporary directory after creating the buffer

# Define the advantage estimation module
gae_module = GAE(
    gamma=gamma_gae,
    lmbda=lmbda_gae,
    value_network=None,
    average_gae=True,
    differentiable=True,
    device=device,
)

# Define the loss module
loss_module = ClipPPOLoss(
    actor_network=action_head,
    critic_network=value_head,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

# Define the optimizer and scheduler
optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)