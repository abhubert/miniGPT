import torch
import torch.nn as nn
from .attention import CausalSelfAttention

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_models, config.d_feedforward, bias=config.bias)
        self.fc2 = nn.Linear(config.d_feedforward, config.d_models, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.LayerNorm(config.d_models)
        self.attn = CausalSelfAttention(config=config)
        self.l2 = nn.LayerNorm(config.d_models)
        self.ff = FeedForward(config=config)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #pre-norm
        x = x + self.attn(self.l1(x))
        x = x + self.ff(self.l2(x))

        return x

