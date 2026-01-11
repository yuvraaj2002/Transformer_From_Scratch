import torch
from torch import nn


class MultiheadAttention(nn.Module):

    def __init__(self,heads,embed_dim,dropout):
        super().__init__()
        self.w_query = nn.Linear(embed_dim,embed_dim,bias=True)
        self.w_key = nn.Linear(embed_dim,embed_dim,bias=True)
        self.w_value = nn.Linear(embed_dim,embed_dim,bias=True)
        self.heads = heads
        self.head_dim = embed_dim//self.heads
        self.dropout_layer = nn.Dropout(dropout)

        # For mixing the data from multi head computation at end
        self.w_out = nn.Linear(embed_dim,embed_dim,bias=True)

    def forward(self,x,mask,dropout):

        # Computing the queries,keys and values [batch,tokens,embed_dim]
        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)

        # Splitting the dimensiona across heads
        queries = queries.view(x.shape[0],x.shape[1],self.heads,self.head_dim)
        keys = keys.view(x.shape[0],x.shape[1],self.heads,self.head_dim)
        values = values.view(x.shape[0],x.shape[1],self.heads,self.head_dim)

        # transposing from multiple head per token to tokens per head
        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        # Computing the attention scores
        attn_scores = (queries @ keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=queries.dtype))

        # Applying mask and then softmax
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Finding the attention weights and adding dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        if dropout:
            attn_weights = self.dropout_layer(attn_weights)

        # Computing context vectors for each head
        context_vectors = attn_weights @ values
        # Shape: [batch, heads, seq_len, head_dim]

        # Transposing and then concatenating the data across heads
        # Transpose to [batch, seq_len, heads, head_dim]
        context_vectors = context_vectors.transpose(1, 2).contiguous()
        # Reshape to concatenate heads: [batch, seq_len, heads * head_dim] = [batch, seq_len, embed_dim]
        context_vectors = context_vectors.view(context_vectors.shape[0], context_vectors.shape[1], -1)

        return self.w_out(context_vectors)
        

