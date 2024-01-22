import torch
from torch import nn
import math

class GLU(nn.Module):
    """Module to compute gated linear units activation.
    Replaces regular activation functions.
    """
    def __init__(self, glu_dim, act=nn.Sigmoid()):
        """GLU class constructor

        Args:
            glu_dim (int): in AND out dimension used for linear layers in GLU computation
            act (nn activation function, optional): non-linear activation function to be used in GLU computation. Defaults to nn.Sigmoid().
        """
        super().__init__()
        self.glu_dim = glu_dim
        self.act = act

        self.linear1 = nn.Linear(in_features=self.glu_dim, out_features=self.glu_dim)
        self.linear2 = nn.Linear(in_features=self.glu_dim, out_features=self.glu_dim)

    def forward(self, x):
        """GLU class forward method

        Args:
            x (tensor): 3D tensor; shape=(batch, max_seq_len, model_dim)

        Returns:
            tensor: input after going through GLU layer
        """
        return self.linear1(x) * self.act(self.linear2(x))
    

class PositionalEncoding(nn.Module):
    """Module to compute fixed sine/cosine positional encodings (PEs) for arbitrary input tensor.
    This layer will compute positional encodings, add them to the input tensor, and apply
    a dropout layer.
    """
    def __init__(self, d_model, dropout, max_len):
        """PositionalEncoding class constructor

        Args:
            d_model (int): model dimension / embedding dimension
            dropout (float): percent dropout to apply after adding PEs to input tensor
            max_len (int): maximum length of spectrum; all spectra padded to this value
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """PositionalEncoding class forward method

        Args:
            x (tensor): 3D tensor; shape=(batch, max_seq_len, model_dim)

        Returns:
            tensor: output will be the input plus PEs with dropout applied to the sum
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)



class FFN(nn.Module):
    """Module for feed forward network used in encoder/decoder layers.
    This layer is different from the original transformer implementation
    because it uses GLU activation.
    """
    def __init__(self, d_model, d_ffn, act):
        """FFN class constuctor

        Args:
            d_model (int): model dimension / embedding dimension
            d_ffn (int): feed forward network dimension; usually an integer multiple of d_model
            act (nn activation function): non-linear activation function to be used in GLU computation.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.glu = GLU(glu_dim=d_ffn, act=act)

    def forward(self, x):
        """FFN class forward method

        Args:
            x (tensor): 3D tensor; shape=(batch, max_seq_len, model_dim)

        Returns:
            tensor: input after going through feed foward network
        """
        return self.fc2(self.glu(self.fc1(x)))