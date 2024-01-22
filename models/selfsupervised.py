from layers.modules import PositionalEncoding, EncoderLayerGLU
import torch
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, ffn_factor, dropout, act):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.ffn_factor = ffn_factor
        self.dropout = dropout
        self.act = act


        # ADD POSITIONAL ENCODINGS AFTER CNN / BEFORE ENCODER BLOCK
        self.cnn = nn.Conv1d(in_channels=self.in_dim, out_channels=self.out_dim, padding='same', stride=1, kernel_size=1)
        self.enc = EncoderLayerGLU(d_model=self.out_dim, num_heads=self.num_heads, d_ffn=self.ffn_factor*self.out_dim, act=self.act, dropout=self.dropout)

    def forward(self, x, mask):
        x = torch.permute(x, (0,2,1))
        x = self.cnn(x)
        x = torch.permute(x, (0,2,1))
        x = x * ~mask.unsqueeze(2)
        x = self.enc(x, mask)
        return x



class SpectrumEncoder(nn.Module):
    def __init__(self, num_heads, ffn_factor, dropout, act, hidden_dims):
        super().__init__()
        self.num_heads = num_heads
        self.ffn_factor = ffn_factor
        self.dropout = dropout
        self.act = act

        self.hidden_dims = sorted(hidden_dims)
        self.all_dims = [2] + self.hidden_dims

        self.enc_list = nn.ModuleList()

        for a,b in zip(self.all_dims, self.all_dims[1:]):
            enc_layer = EncoderBlock(in_dim=a, out_dim=b, num_heads=self.num_heads, ffn_factor=self.ffn_factor, dropout=self.dropout, act=self.act)
            self.enc_list.append(enc_layer)


    def generate_mask(self, x):
        mask = x.sum(axis=-1)==0
        return mask

    def forward(self, x):
        mask = self.generate_mask(x)

        for layer in self.enc_list:
            x = layer(x, mask)

        return x, mask


class SpectrumDecoder(nn.Module):
    def __init__(self, num_heads, ffn_factor, dropout, act, hidden_dims):
        super().__init__()
        self.num_heads = num_heads
        self.ffn_factor = ffn_factor
        self.dropout = dropout
        self.act = act

        self.hidden_dims = sorted(hidden_dims, reverse=True)

        self.dec_list = nn.ModuleList()

        for a,b in zip(self.hidden_dims, self.hidden_dims[1:]):
            dec_layer = EncoderBlock(in_dim=a, out_dim=b, num_heads=self.num_heads, ffn_factor=self.ffn_factor, dropout=self.dropout, act=self.act)
            self.dec_list.append(dec_layer)

        self.final_cnn = nn.Conv1d(in_channels=self.hidden_dims[-1], out_channels=2, padding='same', stride=1, kernel_size=1)

    def forward(self, x, mask):
        for layer in self.dec_list:
            x = layer(x, mask)


        x = torch.permute(x, (0,2,1))
        x = nn.ReLU()(self.final_cnn(x))
        x = torch.permute(x, (0,2,1))
        x = x * ~mask.unsqueeze(2)
        return x




class SpectrumSymmetricAE(nn.Module):
    def __init__(self, num_heads, ffn_factor, dropout, act=nn.Sigmoid(), hidden_dims=None):
        super().__init__()
        self.num_heads = num_heads
        self.ffn_factor = ffn_factor
        self.dropout = dropout
        self.act = act
        if hidden_dims is None:
            self.hidden_dims = [self.num_heads*i for i in range(2,8,2)]
        else:
            assert isinstance(hidden_dims, list) and all(isinstance(elem, int) for elem in hidden_dims), "hidden_dims must be a LIST of INTEGERS that are multiples of num_heads"
            self.hidden_dims = hidden_dims

        self.encoder = SpectrumEncoder(num_heads=self.num_heads, ffn_factor=self.ffn_factor, dropout=self.dropout, act=self.act, hidden_dims=self.hidden_dims)
        self.decoder = SpectrumDecoder(num_heads=self.num_heads, ffn_factor=self.ffn_factor, dropout=self.dropout, act=self.act, hidden_dims=self.hidden_dims)

    def forward(self, x):
        x_enc, x_mask = self.encoder(x)
        x_dec = self.decoder(x_enc, x_mask)

        return x_dec, x_enc, x_mask