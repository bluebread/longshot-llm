import torch
from .transformer import TransformerWithCache

class LongshotModel(torch.nn.Module):
    def __init__(
        self,
        num_literal: int = 10,
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
        if num_literal % 2 != 0:
            raise ValueError("num_literal must be even")
        super(LongshotModel, self).__init__()
        self.num_literal = num_literal
        self.enc_proj = torch.nn.Linear(num_literal, d_model, biase=bias, device=device, dtype=dtype)
        self.dec_proj = torch.nn.Linear(num_literal, d_model, biase=bias, device=device, dtype=dtype)
        self.transformer = TransformerWithCache(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def encode(self, x):
        x = self.enc_proj(x)
        return self.transformer.encode(x)

    def forward(self, x):
        x = self.dec_proj(x)
        return self.transformer(x)