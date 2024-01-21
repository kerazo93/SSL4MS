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
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)



class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, act):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.glu = GLU(glu_dim=d_ffn, act=act)

    def forward(self, x):
        return self.fc2(self.glu(self.fc1(x)))


class EncoderLayerGLU(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, act, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ffn = d_ffn
        self.act = act
        self.dropout = dropout

        self.self_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.num_heads, batch_first=True)
        self.ffn = FFN(d_model=self.d_model, d_ffn=self.d_ffn, act=self.act)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, padding_mask):
        attn_output, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x



class DecoderLayerGLU(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, act, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ffn = d_ffn
        self.act = act
        self.dropout = dropout

        self.self_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.num_heads, batch_first=True)
        self.ffn = FFN(d_model=self.d_model, d_ffn=self.d_ffn, act=self.act)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output, _ = self.self_attn(query=x, key=enc_output, value=enc_output, key_padding_mask=tgt_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))

        attn_output, _ = self.cross_attn(query=x, key=enc_output, value=enc_output, key_padding_mask=src_mask, need_weights=False)
        x = self.norm2(x + self.dropout(attn_output))

        ff_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x