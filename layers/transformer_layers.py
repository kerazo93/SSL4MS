from torch import nn 
from layers.modules import FFN



class EncoderLayerGLU(nn.Module):
    """Module for transformer-style encoder layer with GLU activation.
    """
    def __init__(self, d_model, num_heads, d_ffn, act, dropout):
        """EncoderLayerGLU class constructor

        Args:
            d_model (int): model dimension / embedding dimension
            num_heads (int): number of attention heads to use for multihead attention
            d_ffn (int): feed forward network dimension; usually an integer multiple of d_model
            act (nn activation function): non-linear activation function to be used in GLU computation.
            dropout (float): percent dropout to apply after adding PEs to input tensor
        """
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
        """EncoderLayerGLU forward method

        Args:
            x (tensor): 3D tensor; shape=(batch, max_seq_len, model_dim)
            padding_mask (tensor): 2D tensor; shape=(batch, max_seq_len)

        Returns:
            tensor: output of single encoder layer computation
        """
        attn_output, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x



class DecoderLayerGLU(nn.Module):
    """Module for transformer-style decoder layer with GLU activation.
    """
    def __init__(self, d_model, num_heads, d_ffn, act, dropout):
        """DecoderLayerGLU class constructor

        Args:
            d_model (int): model dimension / embedding dimension
            num_heads (int): number of attention heads to use for multihead attention
            d_ffn (int): feed forward network dimension; usually an integer multiple of d_model
            act (nn activation function): non-linear activation function to be used in GLU computation.
            dropout (float): percent dropout to apply after adding PEs to input tensor
        """
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
        """DecoderLayerGLU forward method

        Args:
            x (tensor): 3D tensor; shape=(batch, max_seq_len, model_dim)
            enc_output (tensor): output from encoder layer(s)
            src_mask (tensor): 2D tensor; shape=(batch, max_seq_len) for source sequence
            tgt_mask (tensor): 2D tensor; shape=(batch, max_seq_len) for target sequence

        Returns:
            tensor: output of single decoder layer computation
        """
        attn_output, _ = self.self_attn(query=x, key=enc_output, value=enc_output, key_padding_mask=tgt_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))

        attn_output, _ = self.cross_attn(query=x, key=enc_output, value=enc_output, key_padding_mask=src_mask, need_weights=False)
        x = self.norm2(x + self.dropout(attn_output))

        ff_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x