import torch
from torch import nn

class GLU(nn.Module):
    def __init__(self, d_model, ff_dim, act=nn.Sigmoid()):
        super().__init__()
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.act = act

        self.linear1 = nn.Linear(in_features=self.d_model, out_features=self.ff_dim)
        self.linear2 = nn.Linear(in_features=self.d_model, out_features=self.ff_dim)

    def forward(self, x):

        return self.linear1(x) * self.act(self.linear2(x))
    


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