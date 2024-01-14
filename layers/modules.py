import torch
from torch import nn
import math

class GLU(nn.Module):
    def __init__(self, glu_dim, act=nn.Sigmoid()):
        super().__init__()
        self.glu_dim = glu_dim
        self.act = act

        self.linear1 = nn.Linear(in_features=self.glu_dim, out_features=self.glu_dim)
        self.linear2 = nn.Linear(in_features=self.glu_dim, out_features=self.glu_dim)

    def forward(self, x):

        return self.linear1(x) * self.act(self.linear2(x))
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)



class EncoderGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass



class DecoderGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass