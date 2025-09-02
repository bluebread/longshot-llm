from torchrl.data.replay_buffers import (
    LazyMemmapStorage, 
    TensorDictReplayBuffer,
    SamplerWithoutReplacement,
)
import tempfile
import torch
from tensordict import TensorDict

from env import FormulaGame
from longshot.literals import CNF

env = FormulaGame(formula=CNF(5), width=3, size=8)

with tempfile.TemporaryDirectory() as tempdir:
    buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(max_size=20, scratch_dir=tempdir), 
        sampler=SamplerWithoutReplacement(),
        batch_size=5,
        # `batch_size` is the batch size to be used when sample() is called.
        #  The batch-size can be specified at construction time via the batch_size argument, or 
        # at sampling time. The former should be preferred whenever the batch-size is 
        # consistent across the experiment.
        #  If the batch-size is likely to change, it can be passed to the sample() method. 
        # This option is incompatible with prefetching (since this requires to know the batch-size 
        # in advance) as well as with samplers that have a drop_last argument.
    )
    
    data = TensorDict({"key1": torch.randn(10, 3)}, batch_size=[10])
    print("Data before extend:", data['key1'])
    buffer.extend(data)
    print("Buffer size after extend:", len(buffer))
    sampled_data = buffer.sample()
    print("Sampled data:", sampled_data)
    print("Sampled data shape:", sampled_data['key1'])
    sampled_data = buffer.sample()
    print("Sampled data after first sample:", sampled_data)
    print("Sampled data shape after first sample:", sampled_data['key1'])
    # Note: The data is sampled without replacement, so the same sample will not be returned
    # twice in a row unless the buffer is extended again.
    sampled_data = buffer.sample()
    print("Sampled data after second sample:", sampled_data)
    print("Sampled data shape after second sample:", sampled_data['key1'])
    # When the sampler reaches the end of the list of available indices, a new sample order 
    # will be generated and the resulting indices will be completed with this new draw, 
    # which can lead to duplicated indices, unless the drop_last argument is set to True.
    
with tempfile.TemporaryDirectory() as tempdir:
    print(f"{'-' * 20} Second extend {'-' * 20}")
    # Caution: If the size of the storage changes in between two calls, the samples will 
    # be re-shuffled (as we can’t generally keep track of which samples have been sampled 
    # before and which haven’t).
    data = TensorDict({"key1": torch.randn(10, 3)}, batch_size=[10])
    print("Data before second extend:", data['key1'])
    buffer.extend(data)
    print("Buffer size after second extend:", len(buffer))
    sampled_data = buffer.sample()
    print("Sampled data after second extend:", sampled_data)
    print("Sampled data shape after second extend:", sampled_data['key1'])
    sampled_data = buffer.sample()
    print("Sampled data after second sample:", sampled_data)
    print("Sampled data shape after second sample:", sampled_data['key1'])
    
    print(f"{'-' * 20} ENV TEST {'-' * 20}")
    buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(max_size=20, scratch_dir=tempdir), 
        sampler=SamplerWithoutReplacement(),
        batch_size=5,
        # `batch_size` is the batch size to be used when sample() is called.
        #  The batch-size can be specified at construction time via the batch_size argument, or 
        # at sampling time. The former should be preferred whenever the batch-size is 
        # consistent across the experiment.
        #  If the batch-size is likely to change, it can be passed to the sample() method. 
        # This option is incompatible with prefetching (since this requires to know the batch-size 
        # in advance) as well as with samplers that have a drop_last argument.
    )
    td = env.rollout(max_steps=20, auto_reset=True)
    print("Rollout data:", td)
    print("action: ", td['action'])
    print("avgQ: ", td['avgQ'])
    print("sequence: ", td['sequence'])
    print("length: ", td['length'])
    print("next/sequence: ", td['next']['sequence'])
    print("next/length: ", td['next']['length'])
    print("next/reward: ", td['next']['reward'])
    print("next/avgQ: ", td['next']['avgQ'])
    buffer.extend(td)
    print("Buffer size after env rollout extend:", len(buffer))
    sampled_data = buffer.sample()
    print("Sampled data after env rollout extend:", sampled_data)
    print("sampled 1/reward: ", sampled_data['next']['reward'])
    sampled_data = buffer.sample()
    print("sampled 2/reward: ", sampled_data['next']['reward'])
    sampled_data = buffer.sample()
    print("sampled 3/reward: ", sampled_data['next']['reward'])
    sampled_data = buffer.sample()
    print("sampled 4/reward: ", sampled_data['next']['reward'])
    # Check if the buffer is working correctly with the environment data