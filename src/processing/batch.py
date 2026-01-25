import numpy as np
import torch

class Batches:

    def __init__(self, batch_size: int = 4, block_size: int = 8):
        self.batch_size = batch_size 
        self.block_size = block_size

    def get_batch(self, data):
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y