import torch
from torchtune.modules import KVCache

# Reference: https://docs.pytorch.org/torchtune/0.3/generated/torchtune.modules.KVCache.html?highlight=kvcache#torchtune.modules.KVCache
cache = KVCache(batch_size=2, max_seq_len=16, num_kv_heads=4, head_dim=32, dtype=torch.float32)
keys, values = torch.ones((2, 4, 8, 32)), torch.zeros((2, 4, 8, 32))
k_out, v_out = cache.update(keys, values)

print("First update:")
print("- Keys' shape:", k_out.shape) 
print("- Values' shape:", v_out.shape)
print("- Cache size:", cache.size)
# Note that k_out.shape[2] would be always 16, as it is the max_seq_len set in KVCache.

keys, values = torch.ones((2, 4, 1, 32)), torch.ones((2, 4, 1, 32))
k_out, v_out = cache.update(keys, values)

print("\nAfter second update:")
print("- Keys' shape:", k_out.shape)
print("- Values' shape:", v_out.shape)
print("- Cache size:", cache.size)

