import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features:int, out_feature:int, bias:float, rank:int, alpha:float|None = None, dropout:float = 0.0):
        super().__init__()

        if not alpha: 
            alpha = rank

        self.weights = nn.Parameter(torch.empty((in_features, out_feature)))
        self.weights.requires_grad = False

        self.scale = alpha/rank

        self.lora_A = nn.Parameter(torch.empty((rank, in_features)))
        self.lora_B = nn.Parameter(torch.empty((out_feature, rank)))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        with torch.no_grad(): 
            nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    
    def forward(self, x):

        x = self.dropout(x)
        x = x @ self.lora_A.T
        x = x @ self.lora_B.T

        return x * self.scale
