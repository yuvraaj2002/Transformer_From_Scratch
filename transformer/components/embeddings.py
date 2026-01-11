"""Input embeddings component for Transformer."""

import math
from torch import nn


class InputEmbeddings(nn.Module):
    """Input embedding layer that converts token indices to dense vectors."""

    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embed_dim)

    def forward(self, x):
        return self.embedding_layer(x) * math.sqrt(self.embed_dim)
