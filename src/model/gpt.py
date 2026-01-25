import torch
import torch.nn as nn 
import torch.nn.functional as fn
from .layers import *

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_models)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.d_models)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_models)
        
        self.lm_head = nn.Linear(config.d_models, config.vocab_size, bias=False)

        self.token_embeddings.weight = self.lm_head.weight

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"Sequence Length {T} exceeds max {self.config.max_seq_len}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        token_embedding = self.token_embeddings(idx)
        position_embedding = self.position_embeddings(pos)
        x = self.dropout(token_embedding + position_embedding)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:

            loss = fn.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):

            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = fn.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx