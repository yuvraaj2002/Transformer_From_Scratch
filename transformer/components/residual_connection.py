import torch
from torch import nn
from .layer_norm import LayerNormalization

class ResidualConnection(nn.Module):

    def __init__(self,embed_dim,epsilon,dropout,prev_layer):
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.prev_layer = prev_layer
        self.layer_norm = LayerNormalization(embed_dim,epsilon)

    def forward(self,x):
        return x + self.dropout_layer(self.prev_layer(self.layer_norm(x)))