import torch
from torch import nn

class ProjectionLayer(nn.Module):

    def __init__(self,embed_dim,vocab_size):
        super().__init__()
        self.projection_layer = nn.Linear(embed_dim,vocab_size)

    def forward(self,x):
        return self.pro