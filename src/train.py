"""
Training script for miniGPT on Shakespeare dataset.
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import requests
from pathlib import Path
import time

from configs.config import GPTConfig
from model.gpt import GPT
from processing.tokenizer import SimpleTokenizer
from processing.batch import Batches


def download_shakespeare():
    """Download the Shakespeare dataset."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    data_path = data_dir / "shakespeare.txt"
    
    if not data_path.exists():
        print("Downloading Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        response.raise_for_status()
        
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Downloaded to {data_path}")
    else:
        print(f"Dataset already exists at {data_path}")
    
    return data_path


def prepare_data(data_path, train_split=0.9):
    """Load and tokenize the Shakespeare dataset."""
    print("\nPreparing data")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset length: {len(text):,} characters")
    
    tokenizer = SimpleTokenizer(text)
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Vocabulary: {''.join(tokenizer.vocab[:50])}...")
    
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    n = int(train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Train set: {len(train_data):,} tokens")
    print(f"Validation set: {len(val_data):,} tokens")
    
    return train_data, val_data, tokenizer


def estimate_loss(model, train_data, val_data, batch_generator, eval_iters=50, device='cpu'):
    """Estimate loss on train and validation sets."""
    model.eval()
    losses = {}
    
    for split, data in [('train', train_data), ('val', val_data)]:
        split_losses = []
        for _ in range(eval_iters):
            x, y = batch_generator.get_batch(data)
            x, y = x.to(device), y.to(device)
            
            with torch.no_grad():
                _, loss = model(x, y)
            split_losses.append(loss.item())
        
        losses[split] = sum(split_losses) / len(split_losses)
    
    model.train()
    return losses


def train(
    model,
    train_data,
    val_data,
    batch_generator,
    optimizer,
    scheduler,
    device='cpu',
    max_iters=5000,
    eval_interval=500,
    eval_iters=50,
    checkpoint_dir='checkpoints',
    save_interval=1000
):
    """Training loop."""
    Path(checkpoint_dir).mkdir(exist_ok=True)
    
    model.to(device)
    model.train()
    
    print(f"\n Starting training on {device}")
    print(f"Max iterations: {max_iters}")
    print(f"Batch size: {batch_generator.batch_size}")
    print(f"Block size: {batch_generator.block_size}")
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    for iter_num in range(max_iters):
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, batch_generator, eval_iters, device)
            elapsed = time.time() - start_time
            print(f"Iter {iter_num:5d} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f} | Time: {elapsed:.1f}s")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint_path = Path(checkpoint_dir) / 'best_model.pt'
                torch.save({
                    'iter': iter_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': losses['train'],
                    'val_loss': losses['val'],
                }, checkpoint_path)
                print(f"Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save checkpoint periodically
        if iter_num > 0 and iter_num % save_interval == 0:
            checkpoint_path = Path(checkpoint_dir) / f'checkpoint_iter_{iter_num}.pt'
            torch.save({
                'iter': iter_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint at iteration {iter_num}")
        
        x, y = batch_generator.get_batch(train_data)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
    
    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")


def generate_sample(model, tokenizer, prompt="", max_new_tokens=200, temperature=0.8, device='cpu'):
    """Generate text sample from the model."""
    model.eval()
    
    if prompt:
        # Encode the prompt
        prompt_ids = tokenizer.encode(prompt)
        x = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    else:
        # Start with a newline
        x = torch.tensor([[0]], dtype=torch.long).to(device)
    
    generated = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature)
    generated_ids = generated[0].tolist()
    generated_text = ''.join(tokenizer.decode(generated_ids))
    
    model.train()
    return generated_text


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    data_path = download_shakespeare()
    train_data, val_data, tokenizer = prepare_data(data_path, train_split=0.9)

    config = GPTConfig(
        vocab_size=len(tokenizer.vocab),
        max_seq_len=256,
        d_models=384,
        n_heads=6,
        n_layers=6,
        d_feedforward=1536,
        dropout=0.2,
        bias=False
    )
    
    model = GPT(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n Model parameters: {total_params:,}")
    
    batch_generator = Batches(batch_size=64, block_size=256)
    
    optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-5)
    
    train(
        model=model,
        train_data=train_data,
        val_data=val_data,
        batch_generator=batch_generator,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_iters=1000,
        eval_interval=250,
        eval_iters=50,
        checkpoint_dir='checkpoints',
        save_interval=1000
    )
    
    # Generate sample text
    print("\n" + "="*80)
    print("Sample generation after training:")
    print("="*80)
    sample = generate_sample(model, tokenizer, prompt="ROMEO:", max_new_tokens=200, temperature=0.8, device=device)
    print(sample)
    print("="*80)
    
    # Save Tokenizer
    import pickle
    tokenizer_path = Path('checkpoints') / 'tokenizer.pkl'
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {tokenizer_path}")
