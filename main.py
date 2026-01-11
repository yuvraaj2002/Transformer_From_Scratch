import torch
from rich import print

from transformer.components import InputEmbeddings, LayerNormalization

if __name__ == "__main__":
    x = torch.tensor([
        [
            [0.1, 0.2, 0.3, 0.4],   # token 1
            [1.1, 1.2, 1.3, 1.4],   # token 2
            [2.1, 2.2, 2.3, 2.4],   # token 3
            [3.1, 3.2, 3.3, 3.4],   # token 4
            [4.1, 4.2, 4.3, 4.4],   # token 5
        ]
    ])
    input_embed_layer = InputEmbeddings(x.shape[-1],20)
    layer_norm = LayerNormalization(x.shape[-1],1e-5)
    #print(layer_norm.forward(x))
    print(input_embed_layer.forward(x))