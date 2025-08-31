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
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        
        super().__init__()
        self.d_head = d_model // nhead  # Dimension of each head
        self.dropout_p = dropout
        self.d_model = d_model
        self.nhead = nhead
        self.layer_norm_eps = layer_norm_eps
        # Projections before Self-attention 
        self.sa_proj_qkv = nn.Linear(d_model, 3 * d_model, bias=bias, device=device, dtype=dtype)
        self.sa_proj_out = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)

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
        qkv = self.sa_proj_qkv(x)  # (batch, 1, 3 * d_model)
        qkv = qkv.unflatten(-1, [3 * self.nhead, self.d_head]).transpose(1, 2)  # (batch, 3 * nhead, 1, d_head)
        q, k_new, v_new = qkv.chunk(3, dim=1)  # Split into q, k_new and v_new # (batch, nhead, 1, d_head)
        
        # Retrieve past keys and values
        k = torch.cat([past_k, k_new], dim=2) if past_k is not None else k_new  # (batch, nhead, seq_len + 1, d_head)
        v = torch.cat([past_v, v_new], dim=2) if past_v is not None else v_new  # (batch, nhead, seq_len + 1, d_head)
        # Compute attention over full sequence
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)
        y = y.transpose(1, 2).flatten(-2)  # (batch, 1, d_model)
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
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        super().__init__()
        self.d_head = d_model // nhead  # Dimension of each head
        self.dropout_p = dropout
        self.d_model = d_model
        self.nhead = nhead
        self.layer_norm_eps = layer_norm_eps
        # Projections before Cross-attention (encoder-decoder)
        self.proj_q = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.proj_out = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)

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
        dec_q = self.proj_q(x)  # (batch, 1, d_model)
        dec_q = dec_q.unflatten(-1, [self.nhead, self.d_head]).transpose(1,2)  # (batch, nhead, 1, d_head)
        # Compute attention with encoder keys and values
        y = F.scaled_dot_product_attention(dec_q, enc_k, enc_v, dropout_p=self.dropout_p) # (batch, nhead, 1, d_head)
        y = y.transpose(1,2).flatten(-2)  # (batch, 1, d_model)
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
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_head = d_model // nhead  # Dimension of each head
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
        self.proj_enc_kv = nn.Linear(d_model, num_decoder_layers * 2 * d_model, bias=bias, device=device, dtype=dtype)
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
        enc_kv = self.proj_enc_kv(y)  # (batch_size, seq_len_enc, num_decoder_layers * 2 * d_model)
        enc_kv = enc_kv.unflatten(-1, [self.nhead, 2 * self.num_encoder_layers, self.d_head]).transpose(1, 2)
        # Split into keys and values for each decoder layer
        chunks = enc_kv.chunk(2 * self.num_decoder_layers, dim=-2) # (batch_size, nhead, seq_len_enc, 1, d_head)
        chunks = tuple(map(lambda t: t.squeeze(), chunks))  # (batch_size, nhead, seq_len_enc, d_head)
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
