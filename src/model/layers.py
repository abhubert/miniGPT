import torch
import torch.nn as nn
from .attention import CausalSelfAttention

class FeedForward(nn.Module):
    def __init__(self, 
                 d_model:int,
                 d_feedforward:int,
                 dropout:float=0.1,
                 bias:bool=True
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_feedforward, bias=bias)
        self.fc2 = nn.Linear(d_feedforward, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model:int,
                 d_head:int,
                 n_heads:int,
                 d_feedforward:int,
                 max_seq_len:int,
                 dropout:float=0.1,
                 bias:bool=True
    ):
        super().__init__()
        self.l1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
                                        n_heads=n_heads, 
                                        d_head=d_head, 
                                        d_model=d_model, 
                                        bias=bias, 
                                        max_seq_len=max_seq_len, 
                                        dropout=dropout
        )
        self.l2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model,
                              d_feedforward=d_feedforward,
                              dropout=dropout,
                              bias=bias)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #pre-norm
        x = x + self.attn(self.l1(x))
        x = x + self.ff(self.l2(x))

        return x

