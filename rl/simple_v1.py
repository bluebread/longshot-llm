import math
import heapq
from tqdm import tqdm
import tempfile
import multiprocessing

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    ParallelEnv
)
from torchrl.data.replay_buffers import (
    LazyMemmapStorage, 
    TensorDictReplayBuffer,
    SamplerWithoutReplacement,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from longshot.circuit import DNF, Literals, NormalFormFormula
from .game import SimpleFixedWidthFormulaGame
from .distrib import GateTokenDistribution


# Hyperparameters and environment parameters

num_vars: int = 5  # number of variables in the formula
width: int = 3  # width of the formula
size: int = 2**5  # size of the formula
eps: float = 1e-6  # small value to avoid division by zero
device: str = "cuda" if torch.cuda.is_available() else "cpu"
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
d_obs = math.comb(num_vars, width) * (2 ** width)  # observation dimension
num_cells = d_obs * 2  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0
num_episodes = 1000  # number of episodes to run
wl_hash_iterations = 10  # number of iterations for the Weisfeiler-Lehman graph hash
num_selected_arms = 32  # maximum number of arms in the UCB algorithm

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

# Replay buffer parameters
buffer_size = 20  # maximum size of the replay buffer
batch_size = 5  # batch size for sampling from the replay buffer

# GAE parameters
gamma_gae = 0.98  # discount factor for GAE
lmbda_gae = 0.95  # lambda for GAE

# Other parameters
statis_freq = 2000  # number of samples to estimate the log probability and entropy

# Define formulas pool and isomorphism hash table

formula_pool: list[NormalFormFormula] = []
iso_lookup: dict[int, list[int]] = {} # map the Weisfeiler-Lehman graph hash to the indices of the formulas in the pool

# UCB algorithm for selecting base formulas

total_steps: int = 0
arms: list[tuple[float, int, float]] = []

def update_arm(idx: int, reward: float):
    global total_steps
    total_steps += 1
    
    if idx < len(arms):
        ucb_v, avg, n = arms[idx]
        avg = (avg * n + reward) / (n + 1)
        ucb_v = avg + math.sqrt(2 * math.log(total_steps) / (n + 1))
        arms[idx] = (ucb_v, n + 1, avg)
    else:
        arms.append((1, reward))

def select_arms(k: int) -> list[int]:
    global arms
    if len(arms) < k:
        return list(range(len(arms)))
    
    topk = heapq.nlargest(k, enumerate(arms), key=lambda x: x[1][0])
    return [idx for idx, _ in topk]

# Define initial environment

base_env = TransformedEnv(
    SimpleFixedWidthFormulaGame(
        DNF(num_vars, device=device),
        width=width,
        size=size,
        eps=eps,
    ),
    Compose(
        DoubleToFloat(),
        StepCounter(),
    ),
)

# TODO: define learning algorithm

td_policy_module = TensorDictModule(
    nn.Sequential(
        nn.Linear(d_obs, num_cells, device=device),
        nn.Tanh(),
        nn.Linear(num_cells, num_cells, device=device),
        nn.Tanh(),
        nn.Linear(num_cells, num_cells, device=device),
        nn.Tanh(),
        nn.Linear(num_cells, 2 * env.action_spec.shape[-1], device=device),
    ), 
    in_keys=["observation"], 
    out_keys=["loc", "scale"] # TODO: modify to match the action space
)

policy_module = ProbabilisticActor(
    module=td_policy_module,
    spec=env.action_spec,
    out_keys=["phi"],
    distribution_class=GateTokenDistribution,
    distribution_kwargs={
        "k": width,  # number of top-k elements to select
        "statis_freq": statis_freq,  # number of samples to estimate the log probability and entropy
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)

value_module = ValueOperator(
    nn.Sequential(
        nn.Linear(d_obs, num_cells, device=device),
        nn.Tanh(),
        nn.Linear(num_cells, num_cells, device=device),
        nn.Tanh(),
        nn.Linear(num_cells, num_cells, device=device),
        nn.Tanh(),
        nn.Linear(num_cells, 1, device=device),
    ),
    in_keys=["observation"],
)

# Define replay buffer
tmpdir = tempfile.TemporaryDirectory()
buffer = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(max_size=buffer_size, scratch_dir=tmpdir), 
    sampler=SamplerWithoutReplacement(),
    batch_size=batch_size,
)
# # DO NOT forget to clean up the temporary directory after creating the buffer
# tmpdir.cleanup()  # Clean up the temporary directory after creating the buffer

# Define the advantage estimation module
advantage_module = GAE(
    gamma=gamma_gae,
    lmbda=lmbda_gae,
    value_network=None,
    average_gae=True,
    differentiable=True,
    device=device,
)

# Define the loss module
loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
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


# TODO: define training loop: 
#   (1) initialize the environment and agent
#   (2) for each episode:
#       (a) pick a random formula from the pool
#       (b) reset the environment with the selected formula
#       (c) for each step in the episode:
#           (i) select an action using the agent's policy
#           (ii) execute the action in the environment
#           (iii) observe the next state and reward
#           (iv) store the transition in the replay buffer
#           (v) sample a batch from the replay buffer
#           (vi) compute the loss and update the agent's policy
#       (d) log the results
#       (e) save the model periodically and evaluate its performance
#       (f) check if the formula has been already solved through the isomorphism hash table

init_formula = DNF(num_vars, device=device)  # initial formula
formula_pool.append(init_formula)  # add the initial formula to the pool
iso_lookup[init_formula.wl_graph_hash(wl_hash_iterations)] = [0]  # initialize the isomorphism lookup table

for i in range(num_episodes):
    # Select a random formula from the pool
    selected_formulas = [formula_pool[i] for i in select_arms(num_selected_arms)]
    envs = ParallelEnv(
        num_selected_arms,
        lambda attr: SimpleFixedWidthFormulaGame(*attr),
        create_env_kwargs=[{ 'formua': f, 'device': device } for f in selected_formulas],
        device=device,
    )
    
    # Define the data collector
    collector = SyncDataCollector(
        envs,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )
    
    for j, td_data in enumerate(collector):
        advantage_module(td_data)
        buffer.extend(td_data)
        
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = buffer.sample(sub_batch_size)
            loss_vals: TensorDictBase = loss_module(subdata.to(device))
            loss_value: torch.Tensor = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()
            
        
        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()