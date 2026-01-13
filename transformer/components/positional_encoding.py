import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len, dropout):
        super().__init__()

        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(seq_len, embed_dim)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)  # shape: (1, seq_len, embed_dim)

        # Register as buffer (not trainable, moves with model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, embed_dim)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
