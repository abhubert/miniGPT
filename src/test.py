import torch
from configs.config import GPTConfig
from model.gpt import GPT

if __name__ == "__main__":
    config = GPTConfig(
        vocab_size=1000,
        max_seq_len=128,
        d_models=256,
        n_heads=8,
        n_layers=4,
        d_feedforward=1024,
    )
    
    model = GPT(config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    idx = torch.randint(0, config.vocab_size, (2, 64))  # batch=2, seq=64
    targets = torch.randint(0, config.vocab_size, (2, 64))
    
    logits, loss = model(idx, targets)
    print(f"Logits shape: {logits.shape}")  # (2, 64, 1000)
    print(f"Loss: {loss.item():.4f}")       # ~6.9 (ln(1000)) initially
    
    prompt = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")  # (1, 21)