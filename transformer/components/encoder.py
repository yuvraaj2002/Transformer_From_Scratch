import torch
from torch import nn
from .feedforward import FeedForwardBlock
from .multihead_attention import MultiheadAttention
from .residual_connection import ResidualConnection
from .layer_norm import LayerNormalization


class EncoderBlock(nn.Module):
    """Single encoder block with self-attention and feed-forward layers."""

    def __init__(self,embed_dim: int,self_attention_block: MultiheadAttention,feed_forward_block: FeedForwardBlock,config) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.config = config

        # Create residual connections: one for attention, one for feed-forward
        self.residual_connection_attention = ResidualConnection(embed_dim, config.epsilon, config.dropout)
        self.residual_connection_ff = ResidualConnection(embed_dim, config.epsilon, config.dropout)

    def forward(self, x, src_mask=None):
        # 1) Self-attention + Add & Norm
        x = self.residual_connection_attention(x, lambda x: self.self_attention_block(x, True, True))

        # 2) FeedForward + Add & Norm
        x = self.residual_connection_ff(x, lambda x: self.feed_forward_block(x))

        return x


class Encoder(nn.Module):
    """Stack of multiple EncoderBlocks."""

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
