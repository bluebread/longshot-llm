import torch
import torch.nn as nn
# from torchtune.modules import KVCache
from torch.nn import functional as F

from typing import Tuple

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This is a custom KVCache modified from the original torchtune implementation
class KVCache(nn.Module):
    """
    Standalone ``nn.Module`` containing a kv-cache to cache past key and values during inference.

    Args:
        batch_size (int): batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        num_kv_heads (int): number of key/value heads.
        head_dim (int): per-attention head embedding dimension
        device (str): device to place the cache on (e.g., 'cpu' or 'cuda')
        dtype (torch.dtype): dtype for the caches
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        # (TODO) Comments from ChatGPT:
        # # Using **buffers** for KV‐caches is perfectly reasonable when you’re doing inference (decoder stepping) 
        # # against a frozen encoder—since you don’t want or need to back‐propagate through those cached projections,
        # # you save memory and avoid rebuilding the graph every time.
        # #   However, if you are in a **training** or **fine-tuning** scenario where the encoder (or its embedding layer) is 
        # # still learnable, you generally do not want to have those cached k_val/v_val writes sever the gradient path:
        # #   Any tensor you store into a buffer via an in‐place copy (self._k_cache[...,pos] = k_val) will be detached 
        # # from the autograd graph.
        # #   That means updates to your encoder (or embedding) parameters will not receive gradient contributions from 
        # # downstream losses that depend on the decoder’s use of those cached keys/values.
        super().__init__()
        cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.register_buffer(
            "_k_cache", torch.zeros(cache_shape, device=device, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "_v_cache", torch.zeros(cache_shape, device=device, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "cache_pos", torch.arange(0, cache_shape[2], device=device), persistent=False
        )
        self.batch_size = batch_size

    def reset(self) -> None:
        """Reset the cache to zero."""
        self._k_cache.zero_()
        self._v_cache.zero_()
        self.cache_pos -= self.size

    @property
    def size(self) -> int:
        return self.cache_pos[0].item()

    @property
    def k_cache(self) -> torch.Tensor:
        return self._k_cache[:, :, :self.size, :]
    
    @property
    def v_cache(self) -> torch.Tensor:
        return self._v_cache[:, :, :self.size, :]

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache with the new ``k_val``, ``v_val`` and return the updated cache.

        Note:
            When updating the KV cache, it is assumed that subsequent updates should update key-value
            positions in consecutive sequence positions. If you wish to update cache values which have
            already been filled, use ``.reset()``, which will reset the cache to the zero-th position.

        Example:
            >>> cache = KVCache(batch_size=2, max_seq_len=16, num_kv_heads=4, head_dim=32, dtype=torch.bfloat16)
            >>> keys, values = torch.ones((2, 4, 8, 32)), torch.ones((2, 4, 8, 32))
            >>> cache.update(keys, values)
            >>> # now positions 0 through 7 are filled
            >>> cache.size
            >>> 8
            >>> keys, values = torch.ones((2, 4, 1, 32)), torch.ones((2, 4, 1, 32))
            >>> cache.update(keys, values)
            >>> # this will fill at position 8
            >>> cache.size
            >>> 9

        Args:
            k_val (torch.Tensor): Current key tensor with shape [B, H, S, D]
            v_val (torch.Tensor): Current value tensor with shape [B, H, S, D]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated key and value cache tensors, respectively.

        Raises:
            ValueError: if the batch size of the new key (or value) tensor is greater than the batch size
                used during cache setup.

        Note:
            This function will raise an ``AssertionError`` if the sequence length of ``k_val``
                is longer than the maximum cache sequence length.

        """
        bsz, _, seq_len, _ = k_val.shape
        if bsz > self._k_cache.shape[0]:
            raise ValueError(
                f"The current cache has been setup with a batch size of {self._k_cache.shape[0]}"
                f", but found new key tensors with batch size {k_val.shape[0]}!"
            )

        assert (self.cache_pos[0] + seq_len) <= self._k_cache.shape[2]
        k_out = self._k_cache
        v_out = self._v_cache

        k_out[:, :, self.cache_pos[:seq_len]] = k_val
        v_out[:, :, self.cache_pos[:seq_len]] = v_val

        # forward cache_pos seq_len positions along
        # cache_pos starts at (0, 1, 2, 3, 4, 5, ...)
        # an update of seq_len = 5 tokens brings it to
        # (5, 6, 7, 8, 9, ...)
        # this allows us to track the current position in the cache
        # after the last update in a compile-friendly way without any dynamism
        # e.g. relying on an int size tracker, or re-creating cache_pos every time
        self.cache_pos.add_(seq_len)

        return self.k_cache, self.v_cache


class TransformerDecoderLayerWithCache(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        d_total = d_model * nhead  # Total dimension of all heads
        self.dropout_p = dropout
        self.d_model = d_model
        self.nhead = nhead
        # Projections before Self-attention 
        self.sa_proj_qkv = nn.Linear(d_model, 3 * d_total, bias=bias, device=device, dtype=dtype)
        self.sa_proj_out = nn.Linear(d_total, d_model, bias=bias, device=device, dtype=dtype)
        # Projections before Cross-attention (encoder-decoder)
        self.ca_proj_q = nn.Linear(d_model, d_total, bias=bias, device=device, dtype=dtype)
        self.ca_proj_kv = nn.Linear(d_model, 2 * d_total, bias=bias, device=device, dtype=dtype)
        self.ca_proj_out = nn.Linear(d_total, d_model, bias=bias, device=device, dtype=dtype)
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, device=device, dtype=dtype)
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self,
                in_token: torch.Tensor,
                memory: torch.Tensor,
                caches: dict[str, KVCache]
    ) -> torch.Tensor:
        """
        in_token: (batch, 1, d_model) new decoder input token embeddings
        memory: (batch, seq_len_enc, d_model) encoder outputs
        caches: {
            'self': KVCache for decoder self-attention,
            'cross': KVCache for encoder-decoder attention
        }
        Returns:
            out: (batch, 1, d_model) decoded token output
            updated caches dict
        """
        # ----- Self-Attention with KV-Cache -----
        sa_cache = caches.get('self')
        # Project queries, new keys & values
        qkv = self.sa_proj_qkv(in_token) # (batch, 1, 3 * d_total)
        q, k_new, v_new = qkv.chunk(3, dim=-1)  # Split into q, k, v # (batch, 1, d_total)
        q = q.unflatten(-1, [self.nhead, self.d_model]).transpose(1,2)  # (batch, nhead, 1, d_model)
        k_new = k_new.unflatten(-1, [self.nhead, self.d_model]).transpose(1, 2)
        v_new = v_new.unflatten(-1, [self.nhead, self.d_model]).transpose(1, 2)
        # Retrieve past keys and values
        k, v = sa_cache.update(k_new, v_new) 

        # Compute attention over full sequence
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)
        x = x.transpose(1, 2).flatten(-2)  # (batch, 1, d_model * nhead)
        x = self.sa_proj_out(x) # (batch, 1, d_model)
        x = self.dropout(x)  # Apply dropout
        x = x + in_token  # Add new token to attention output
        x = self.norm1(x)

        # ----- Cross-Attention (encoder-decoder) -----
        cross_cache = caches.get('cross')
        # For memory, caching is optional since memory is static across decoding
        # TODO: Create a new method to handle memory caching, which might be more efficient
        # TODO: Since KVCache holds key/value tensors as buffers, it would **break** autograd
        #       connection back to the encoder outputs. Seperating memory caching is needed.
        if cross_cache and cross_cache.size == 0:
            # initialize cross cache once
            mkv = self.ca_proj_kv(memory) # (batch, seq_len_enc, 2 * d_total)
            mk, mv = mkv.chunk(2, dim=-1)  # Split into k, v
            mk = mk.unflatten(-1, [self.nhead, self.d_model]).transpose(1, 2)
            mv = mv.unflatten(-1, [self.nhead, self.d_model]).transpose(1, 2)
            # Update cross cache with memory keys and values
            cross_cache.update(mk, mv)

        mq = self.ca_proj_q(x)
        mq = mq.unflatten(-1, [self.nhead, self.d_model]).transpose(1, 2)
        mk = cross_cache.k_cache
        mv = cross_cache.v_cache
        y = F.scaled_dot_product_attention(mq, mk, mv, dropout_p=self.dropout_p)
        y = y.transpose(1, 2).flatten(-2)  # (batch, 1, d_model * nhead)
        y = self.ca_proj_out(y)  # (batch, 1, d_model)
        y = self.dropout(y)  # Apply dropout
        y = y + x  # Add self-attention output
        y = self.norm2(y)

        # ----- Feed-Forward -----
        z = F.relu(self.linear1(y))
        z = self.linear2(self.dropout(z))
        z = z + y
        out = self.norm3(z)

        return out


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    d_model = 64
    nhead = 4
    dim_feedforward = 256
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    decoder_layer = TransformerDecoderLayerWithCache(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        device=device,
        dtype=torch.float32
    )

    in_token = torch.randn(batch_size, 1, d_model, device=device, dtype=torch.float32)  # New token input
    memory = torch.randn(batch_size, 10, d_model, device=device, dtype=torch.float32)  # Example encoder output

    caches = {
        'self': KVCache(batch_size=batch_size, max_seq_len=16, num_kv_heads=nhead, head_dim=d_model, device=device, dtype=torch.float32),
        'cross': KVCache(batch_size=batch_size, max_seq_len=16, num_kv_heads=nhead, head_dim=d_model, device=device, dtype=torch.float32)
    }

    out = decoder_layer(in_token, memory, caches)
    print("Output shape:", out.shape)  # Should be (batch_size, 1, d_model)