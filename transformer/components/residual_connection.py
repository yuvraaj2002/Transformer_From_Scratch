import torch
from torch import nn
from .layer_norm import LayerNormalization


class ResidualConnection(nn.Module):

    def __init__(self, embed_dim, epsilon, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(embed_dim, epsilon)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
