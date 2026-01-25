# MiniGPT

A minimal GPT implementation from scratch in PyTorch. Built for learning and experimentation with transformer architectures.

## Overview

This is a character-level GPT trained on the Tiny Shakespeare dataset. The implementation includes:

- Causal self-attention with multi-head mechanism
- Transformer blocks with pre-norm architecture
- Position and token embeddings
- Text generation with temperature and top-k sampling

Will expand with different tokenization, pre-processing, architectures, interfaces. 

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

Train on Shakespeare dataset (downloads automatically):

```bash
python src/train.py
```

The script will:
- Download the Tiny Shakespeare dataset to `data/`
- Train for 1000 iterations (~5-10 minutes on GPU)
- Save checkpoints to `checkpoints/`
- Generate a sample at the end

### Testing the Model

```bash
python src/test.py
```

## Project Structure

```
src/
├── configs/          # Model configuration
├── model/            # GPT architecture (attention, layers, main model)
├── processing/       # Tokenizer and data batching
├── train.py          # Training script
└── test.py           # Model testing
```

## Configuration

Modify [`GPTConfig`](src/configs/config.py) to experiment with different architectures:

```python
config = GPTConfig(
    vocab_size=65,        # Character vocabulary size
    max_seq_len=256,      # Maximum sequence length
    d_models=384,         # Model dimension
    n_heads=6,            # Number of attention heads
    n_layers=6,           # Number of transformer blocks
    d_feedforward=1536,   # Feedforward dimension
    dropout=0.2,          # Dropout rate
)
```

## Key Components

- **Model**: [`GPT`](src/model/gpt.py) - Main transformer model with generation
- **Attention**: [`CausalSelfAttention`](src/model/attention.py) - Masked multi-head attention
- **Layers**: [`TransformerBlock`](src/model/layers.py), [`FeedForward`](src/model/layers.py)

## Training Details

- **Optimizer**: AdamW with learning rate 3e-4
- **Scheduler**: Cosine annealing
- **Loss**: Cross-entropy
- **Dataset**: Tiny Shakespeare (~1MB of text)
- **Device**: Automatically uses CUDA, MPS (Apple Silicon), or CPU

## Generation

After training, generate text with:

```python
from model.gpt import GPT
import torch

# Load model
checkpoint = torch.load('checkpoints/best_model.pt')
model = GPT(config)
model.load_state_dict(checkpoint['model_state_dict'])

# Generate
prompt = torch.tensor([[tokenizer.encode("ROMEO:")]], dtype=torch.long)
output = model.generate(prompt, max_new_tokens=200, temperature=0.8)
```

## Notes

Under Construction 

## License

MIT
