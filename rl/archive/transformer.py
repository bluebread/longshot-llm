import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

class CausalSelfAttentionWithCache(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        super().__init__()
        d_total = d_model * nhead  # Total dimension of all heads
        self.dropout_p = dropout
        self.d_model = d_model
        self.nhead = nhead
        self.layer_norm_eps = layer_norm_eps
        # Projections before Self-attention 
        self.sa_proj_qkv = nn.Linear(d_model, 3 * d_total, bias=bias, device=device, dtype=dtype)
        self.sa_proj_out = nn.Linear(d_total, d_model, bias=bias, device=device, dtype=dtype)

    def forward(
        self,
        x: Tensor,  # (batch, 1, d_model) input token embedding
        past_k: Tensor | None = None,  # (batch, nhead, seq_len, d_model) past keys
        past_v: Tensor | None = None   # (batch, nhead, seq_len, d_model) past values
    ) -> Tensor:
        """
        x: (batch, 1, d_model) input token embedding
        past_k: (batch, nhead, seq_len, d_model) past keys
        past_v: (batch, nhead, seq_len, d_model) past values
        Returns:
            out: (batch, seq_len, d_model) 
            k: (batch, nhead, seq_len + 1, d_model)
            v: (batch, nhead, seq_len + 1, d_model)
        """
        # Project queries, new keys & values
        qkv = self.sa_proj_qkv(x)  # (batch, 1, 3 * d_total)
        qkv = qkv.unflatten(-1, [3 * self.nhead, self.d_model]).transpose(1, 2)  # (batch, 3 * nhead, 1, d_model)
        q, k_new, v_new = qkv.chunk(3, dim=1)  # Split into q, k_new and v_new # (batch, nhead, 1, d_model)
        
        # Retrieve past keys and values
        k = torch.cat([past_k, k_new], dim=2) if past_k is not None else k_new  # (batch, nhead, seq_len + 1, d_model)
        v = torch.cat([past_v, v_new], dim=2) if past_v is not None else v_new  # (batch, nhead, seq_len + 1, d_model)
        # Compute attention over full sequence
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)
        y = y.transpose(1, 2).flatten(-2)  # (batch, 1, d_total)
        y = self.sa_proj_out(y)  # (batch, 1, d_model)
        y = F.dropout(y, p=self.dropout_p)  # Apply dropout
        y = y + x  # Add new token to attention output
        out = F.layer_norm(y, (self.d_model,), eps=self.layer_norm_eps)
        
        return out, k, v # Return updated caches

class CrossAttentionWithCache(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        super().__init__()
        d_total = d_model * nhead  # Total dimension of all heads
        self.dropout_p = dropout
        self.d_model = d_model
        self.nhead = nhead
        self.layer_norm_eps = layer_norm_eps
        # Projections before Cross-attention (encoder-decoder)
        self.proj_q = nn.Linear(d_model, d_total, bias=bias, device=device, dtype=dtype)
        self.proj_out = nn.Linear(d_total, d_model, bias=bias, device=device, dtype=dtype)

    def forward(
        self,
        x: Tensor,  # (batch, 1, d_model) decoder input token embeddings
        enc_k: Tensor,     # (batch, nhead, seq_len_enc, d_model) encoder keys
        enc_v: Tensor      # (batch, nhead, seq_len_enc, d_model) encoder values
    ) -> Tensor:
        """
        x: (*, d_model) decoder input token embeddings
        Returns:
            out: (batch, seq_len_dec, d_model) cross-attention output
        """
        # Project queries
        dec_q = self.proj_q(x)  # (batch, 1, d_total)
        dec_q = dec_q.unflatten(-1, [self.nhead, self.d_model]).transpose(1,2)  # (batch, nhead, 1, d_model)
        # Compute attention with encoder keys and values
        y = F.scaled_dot_product_attention(dec_q, enc_k, enc_v, dropout_p=self.dropout_p) # (batch, nhead, 1, d_model)
        y = y.transpose(1,2).flatten(-2)  # (batch, 1, d_total)
        # Apply output projection
        y = self.proj_out(y)  # (batch, 1, d_model)
        y = F.dropout(y, p=self.dropout_p)  # Apply dropout
        y = y + x  # Add self-attention output
        out = F.layer_norm(y, (self.d_model,), eps=self.layer_norm_eps)

        return out

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, device=device, dtype=dtype)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (*, d_model) input tensor
        Returns:
            out: (*, d_model) output tensor after FFN
        """
        y = F.relu(self.linear1(x))  # Apply ReLU activation
        y = F.dropout(y, p=self.dropout)  # Apply dropout
        y = self.linear2(y)  # Output projection
        y = y + x
        out = F.layer_norm(y, (self.d_model,), eps=self.layer_norm_eps)  # Add residual connection and layer norm
        
        return out

class TransformerDecoderLayerWithCache(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        super().__init__()
        self.self_attn = CausalSelfAttentionWithCache(
            d_model=d_model, nhead=nhead, dropout=dropout, layer_norm_eps=layer_norm_eps, bias=bias, device=device, dtype=dtype
        )
        self.cross_attn = CrossAttentionWithCache(
            d_model=d_model, nhead=nhead, dropout=dropout, layer_norm_eps=layer_norm_eps, bias=bias, device=device, dtype=dtype
        )
        self.ffn = FeedForwardNetwork(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout, layer_norm_eps=layer_norm_eps, bias=bias, device=device, dtype=dtype
        )

    def forward(
        self,
        x: Tensor,  # (batch_size, 1, d_model)
        past_k: Tensor,  # (batch, nhead, seq_len, d_model) past keys
        past_v: Tensor,   # (batch, nhead, seq_len, d_model) past values
        enc_k: Tensor,  # (batch_size, nhead, seq_len_enc, d_model)
        enc_v: Tensor   # (batch_size, nhead, seq_len_enc, d_model)
    ) -> Tensor:
        """
        x: (batch_size, 1, d_model) decoder input token embeddings
        past_k: (batch, nhead, seq_len_dec, d_model) past keys for self-attention
        past_v: (batch, nhead, seq_len_dec, d_model) past values for self-attention
        enc_k: (batch_size, nhead, seq_len_enc, d_model) encoder keys
        enc_v: (batch_size, nhead, seq_len_enc, d_model) encoder values
        Returns:
            out: (batch_size, seq_len_dec, d_model) output tensor after decoder layer
        """
        # Self-attention with caching
        y, past_k, past_v = self.self_attn(x, past_k, past_v)
        # Cross-attention with encoder keys and values
        y = self.cross_attn(y, enc_k, enc_v)
        # Feed-forward network
        y = self.ffn(y)
        
        return y, past_k, past_v  # Return updated caches for next layer

class TransformerWithCache(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        d_total = d_model * nhead  # Total dimension of all heads
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias
        # Initialize encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # Use ReLU activation
            layer_norm_eps=layer_norm_eps,
            batch_first=True,  # Use batch_first for input shape (batch_size, seq_len, d_model)
            norm_first=False,  # Use post-layer normalization
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.proj_enc_kv = nn.Linear(d_model, num_decoder_layers * 2 * d_total, bias=bias, device=device, dtype=dtype)
        self.enc_kv = None  # Cache for encoder keys and values
        # Initialize decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayerWithCache(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                bias=bias,
                device=device,
                dtype=dtype
            ) for _ in range(num_decoder_layers)
        ])
        self.past_kv = [(None, None) for _ in range(num_decoder_layers)]  # Cache for past keys and values
        

    def encode(self, x: Tensor) -> None:
        """
        Set the memory for the transformer decoder.
        This is used to cache the memory from the encoder.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len_enc, d_model).
        """
        y = self.encoder(x)
        # Project encoder outputs to keys and values
        enc_kv = self.proj_enc_kv(y)  # (batch_size, seq_len_enc, num_decoder_layers * 2 * d_total)
        enc_kv = enc_kv.unflatten(-1, [self.nhead, 2 * self.num_encoder_layers, self.d_model]).transpose(1, 2)
        # Split into keys and values for each decoder layer
        chunks = enc_kv.chunk(2 * self.num_decoder_layers, dim=-2) # (batch_size, nhead, seq_len_enc, 1, d_model)
        chunks = tuple(map(lambda t: t.squeeze(), chunks))  # (batch_size, nhead, seq_len_enc, d_model)
        self.enc_kv = [(chunks[i], chunks[i + self.num_decoder_layers]) for i in range(self.num_decoder_layers)]
        # Reset past keys and values for decoder layers
        self.past_kv = [(None, None) for _ in range(self.num_decoder_layers)]  # Cache for past keys and values

    def forward(self, x: Tensor) -> Tensor: 
        """
        Forward pass of the transformer model.
        Args:
            x (Tensor): Input token tensor of shape (batch_size, 1, d_model).
        """
        if self.enc_kv is None:
            raise ValueError("Memory not set. Call encode() before forward().")
        
        y = x  # Start with the input token embedding
        
        for i in range(self.num_decoder_layers):
            past_k, past_v = self.past_kv[i]
            enc_k, enc_v = self.enc_kv[i]  # Get encoder keys and values
            # Pass through each decoder layer
            y, past_k, past_v = self.decoder_layers[i](y, past_k, past_v, enc_k, enc_v)
            # Update the cache for the next layer
            self.past_kv[i] = (past_k, past_v)
        
        return y  # Return the final output after all decoder layers
    
if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len_enc = 10
    seq_len_dec = 5
    d_model = 32
    nhead = 4

    # Create a transformer model
    transformer = TransformerWithCache(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.1,
        layer_norm_eps=1e-5,
        bias=True,
        device='cpu',  # Change to 'cuda' if using GPU
        dtype=torch.float32  # Change to torch.float16 if using mixed precision
    )

    # Example encoder input (batch_size, seq_len_enc, d_model)
    enc_input = torch.randn(batch_size, seq_len_enc, d_model)
    
    # Encode the input to set the memory
    transformer.encode(enc_input)

    # Example decoder input (batch_size, 1, d_model)
    dec_input = torch.randn(batch_size, 1, d_model)

    # Forward pass through the transformer
    output = transformer(dec_input)

    print("Output shape:", output.shape)  # Should be (batch_size, 1, d_model)
    
    # Again, you can use the transformer with a different input
    dec_input = torch.randn(batch_size, 1, d_model)
    output = transformer(dec_input)
    print("Output shape after second input:", output.shape)  # Should be (batch_size

# ChatGPT comments: 
# What you’ve stumbled on is the same trade-off every multi-head design faces:

# 1. **Full-embedding per head**

#    * **What it means**: you let each head’s projection live in the full input space of size D\_in (so head\_dim = D\_in, and d\_model = nhead × D\_in).
#    * **Pro**: every head can “see” and re-weigh all D\_in features; you’re not throwing any information away.
#    * **Con**: your Q/K/V projections balloon in size by a factor of nhead.  Compute goes from O(D\_in²) → O(nhead·D\_in²), parameters likewise.  If D\_in≈40 and nhead=8, you’re suddenly projecting from 40→(3×320)=960 dims instead of 40→(3×128)=384 (for a more typical head\_dim=16 approach).

# 2. **Split-embedding across heads** (the usual way)

#    * You pick **d\_model** = head\_dim × nhead, and head\_dim = d\_model/nhead.
#    * **Pro**: total projection size stays at O(d\_model·D\_in), you get nhead “perspectives” but each at a reduced dimension.  More heads doesn’t blow up your compute.
#    * **Con**: if D\_in is tiny (e.g. 30–40) and you insist on 8 heads, you’re forced into head\_dim=4–5, which can be too low for each head to learn anything useful.

# ---

# ## A middle ground: **embed-up then split**

# Because your raw feature size (30–40) is small, you can **learn** a small “expansion” first:

# ```python
# # Suppose D_in = 40, and you’d like head_dim ≈ 16 with nhead = 8
# d_model = 16 * 8  # = 128  
# self.input_proj = nn.Linear(D_in, d_model)
# # …then your attention uses d_model=128, head_dim=16
# ```

# * **Effect**: each head still only sees a 16-dim subspace, but the network can learn how to *pack* those 40 raw features into the 128-dim space in the best way.
# * **Compute**: you pay the one 40→128 projection, then run a standard multi-head with total size O(128·40) for QKV instead of O(320·40).

# ---

# ## When “full embedding per head” can make sense

# * If D\_in is already small *and* your downstream task is extremely sensitive to interactions among *all* inputs, you might tolerate the extra factor-of-nhead compute.
# * But in practice:

#   1. **Fewer heads** (e.g. 2–4) with head\_dim≈D\_in can give you the “every head sees everything” effect without exploding parameter count.
#   2. **Single-query or grouped-query attention** (multi-query, or “shared K/V” across heads) can also reduce cost while sharing information across heads.

# ---

# ### My recommendation for your 30–40-dim problem

# * **Decide on a reasonable head\_dim** (say 8–16).
# * **Compute** d\_model = head\_dim × nhead.
# * **Up-project** your raw features into that space via one small nn.Linear(D\_in, d\_model).
# * **Run standard split-head attention**.

# This gives you:

# * A compact, learnable “feature expansion,”
# * Enough capacity per head,
# * No unnecessary nhead-fold blow-up in compute.

# If you truly want each head to see the *raw* D\_in as head\_dim, then reduce nhead or accept the parameter/computation increase. But almost every Transformer-style model finds the “embed-up then split” design a better sweet-spot of capacity vs. efficiency.
