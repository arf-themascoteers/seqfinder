import torch
import torch.nn as nn
import math
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        posdiv = position * div_term
        pe[:, 0, 0::2] = torch.sin(posdiv)
        pe[:, 0, 1::2] = torch.cos(posdiv)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        y = self.pe[:x.size(0)]
        x = x.reshape(-1,1,1) + y
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(5)
    pe = PositionalEncoding(10,100)
    y = pe(x)
    z = x.reshape(-1,1,1) + y
    print(z.shape)