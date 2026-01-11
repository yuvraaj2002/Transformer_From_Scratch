"""Positional encoding component for Transformer."""

from torch import nn


class PositionalEncodding(nn.Module):
    """Positional encoding layer that adds positional information to embeddings."""

    def __init__(self, embed_dim, seq_len, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.dropout = dropout
        self.positional_embedding = nn.Embedding(self.seq_len, self.embed_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        return self.dropout_layer(self.positional_embedding(x))
