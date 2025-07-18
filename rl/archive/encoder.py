import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA // Scaled Dot Product Attention
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output
    
class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        multiple_of,
        ffn_dim_multiplier=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False, **factory_kwargs)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False, **factory_kwargs)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, nheads, ffn_dim_multiplier=4, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(
            E_q=dim,
            E_k=dim,
            E_v=dim,
            E_total=dim,
            nheads=nheads,
            dropout=dropout,
        )
        self.ffn = SwiGLUFFN(
            dim=dim,
            hidden_dim=int(dim * ffn_dim_multiplier),
            multiple_of=8,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def test_custom_transformer():
    x = torch.randn(10, 20, 64)  # (N, L_t, dim)
    
    mha = MultiHeadAttention(
        E_q=64,
        E_k=64,
        E_v=64,
        E_total=64,
        nheads=4,
        dropout=0.1,
    )
    ffn = SwiGLUFFN(
        dim=64,
        hidden_dim=128,
        multiple_of=8,
    )
    norm = nn.RMSNorm(64, eps=1e-6)
    
    y = mha(x, x, x)
    y = norm(x + y)  # Residual connection and normalization
    y = ffn(y)
    y = norm(y + x)  # Another residual connection and normalization
    
    print(y.shape)  # Should be (10, 20, 64)
    
def test_pytorch_transformer():
    x = torch.randn(10, 20, 64)  # (N, L_t, dim)
    
    transformer_layer = nn.TransformerEncoderLayer(
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        dropout=0.1,
        activation='relu',
        bias=False,
        batch_first=True,
    )
    transformer = nn.TransformerEncoder(
        transformer_layer,
        num_layers=2,
        norm=None,
        enable_nested_tensor=False,
    )
    transformer_output = transformer(x)
    print(transformer_output.shape)  # Should be (10, 20, 64)
    
def test_performance():
    import time
    x = torch.randn(1000, 50, 512)  # (N, L_t, dim)
    
    start_time = time.time()
    for _ in range(100):
        transformer = TransformerBlock(dim=512, nheads=8, ffn_dim_multiplier=4, dropout=0.1)
        _ = transformer(x)
    print(f"Custom TransformerBlock time: {time.time() - start_time:.4f} seconds")
    
    start_time = time.time()
    for _ in range(100):
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            bias=False,
            batch_first=True,
        )
        transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=1,
            norm=None,
            enable_nested_tensor=False,
        )
        _ = transformer(x)
    print(f"Pytorch Transformer time: {time.time() - start_time:.4f} seconds")
    
if __name__ == "__main__":
    test_custom_transformer()
    test_pytorch_transformer()
    test_performance()
    