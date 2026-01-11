"""Layer normalization component for Transformer."""

import torch
from torch import nn


class LayerNormalization(nn.Module):
    """Custom layer normalization implementation."""

    def __init__(self, embed_dim, epsilon):
        super().__init__()

        # Initializing the scaling and shifting tensors
        self.bias = torch.ones(embed_dim)
        self.alphas = torch.zeros(embed_dim)
        self.epsilon = epsilon

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)
        norm_x = (x - mean) / (torch.sqrt(var) + self.epsilon)
        return norm_x * self.bias + self.alphas
