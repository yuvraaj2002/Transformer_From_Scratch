"""Feedforward network component for Transformer."""

from torch import nn


class FeedForwardBlock(nn.Module):
    """Feedforward block with two linear layers and dropout."""

    def __init__(self, embed_dim_in, embed_dim_out, dropout):
        super().__init__()
        self.linear_layer_1 = nn.Linear(embed_dim_in, embed_dim_out)
        self.dropout_layer = nn.Dropout(dropout)
        self.linear_layer_2 = nn.Linear(embed_dim_out, embed_dim_in)

    def forward(self, x):
        return self.linear_layer_2(self.dropout_layer(self.linear_layer_1(x)))
