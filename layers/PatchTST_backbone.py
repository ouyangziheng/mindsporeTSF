__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import mindspore
from mindspore import nn, ops, Tensor
import numpy as np

# from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN

# Cell
class PatchTST_backbone(nn.Cell):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int, max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0., act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True, pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True, subtract_last=False,
                 verbose: bool = False, **kwargs):
        super(PatchTST_backbone, self).__init__()

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.Pad(((0, stride),))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

    def construct(self, z):
        # norm
        if self.revin:
            z = ops.Transpose()(z, (0, 2, 1))
            z = self.revin_layer(z, 'norm')
            z = ops.Transpose()(z, (0, 2, 1))

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = ops.Unfold()(z, kernel_size=(1, self.patch_len), stride=(1, self.stride))
        z = ops.Transpose()(z, (0, 1, 3, 2))

        # model
        z = self.backbone(z)
        z = self.head(z)

        # denorm
        if self.revin:
            z = ops.Transpose()(z, (0, 2, 1))
            z = self.revin_layer(z, 'denorm')
            z = ops.Transpose()(z, (0, 2, 1))
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.SequentialCell([nn.Dropout(1.0 - dropout), nn.Conv1d(head_nf, vars, kernel_size=1)])


class Flatten_Head(nn.Cell):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super(Flatten_Head, self).__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.CellList()
            self.dropouts = nn.CellList()
            self.flattens = nn.CellList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten())
                self.linears.append(nn.Dense(nf, target_window))
                self.dropouts.append(nn.Dropout(1.0 - head_dropout))
        else:
            self.flatten = nn.Flatten()
            self.linear = nn.Dense(nf, target_window)
            self.dropout = nn.Dropout(1.0 - head_dropout)

    def construct(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = ops.Stack(axis=1)(x_out)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TSTiEncoder(nn.Cell):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super(TSTiEncoder, self).__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Dense(patch_len, d_model)
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(1.0 - dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

    def construct(self, x) -> Tensor:
        n_vars = x.shape[1]
        # Input encoding
        x = ops.Transpose()(x, (0, 1, 3, 2))
        x = self.W_P(x)

        u = ops.Reshape()(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        u = self.dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(u)
        z = ops.Reshape()(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        z = ops.Transpose()(z, (0, 1, 3, 2))
        return z


class TSTEncoder(nn.Cell):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super(TSTEncoder, self).__init__()

        self.layers = nn.CellList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                   attn_dropout=attn_dropout, dropout=dropout,
                                                   activation=activation, res_attention=res_attention,
                                                   pre_norm=pre_norm, store_attn=store_attn) for _ in range(n_layers)])
        self.res_attention = res_attention

    def construct(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers:
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Cell):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super(TSTEncoderLayer, self).__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(1.0 - dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.SequentialCell([Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)])
        else:
            self.norm_attn = nn.LayerNorm([d_model])

        # Position-wise Feed-Forward
        self.ff = nn.SequentialCell([nn.Dense(d_model, d_ff, has_bias=bias),
                                     get_activation_fn(activation),
                                     nn.Dropout(1.0 - dropout),
                                     nn.Dense(d_ff, d_model, has_bias=bias)])

        # Add & Norm
        self.dropout_ffn = nn.Dropout(1.0 - dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.SequentialCell([Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)])
        else:
            self.norm_ffn = nn.LayerNorm([d_model])

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def construct(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        # Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        # Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        # Position-wise Feed-Forward
        src2 = self.ff(src)
        # Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Cell):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super(_MultiheadAttention, self).__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Dense(d_model, d_k * n_heads, has_bias=qkv_bias)
        self.W_K = nn.Dense(d_model, d_k * n_heads, has_bias=qkv_bias)
        self.W_V = nn.Dense(d_model, d_v * n_heads, has_bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Project output
        self.to_out = nn.SequentialCell([nn.Dense(n_heads * d_v, d_model), nn.Dropout(1.0 - proj_dropout)])

    def construct(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                  key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        bs = Q.shape[0]
        if K is None:
            K = Q
        if V is None:
            V = Q

        # Linear (+ split in multiple heads)
        q_s = ops.Reshape()(self.W_Q(Q), (bs, -1, self.n_heads, self.d_k)).transpose(0, 2, 1, 3)
        k_s = ops.Reshape()(self.W_K(K), (bs, -1, self.n_heads, self.d_k)).transpose(0, 2, 3, 1)
        v_s = ops.Reshape()(self.W_V(V), (bs, -1, self.n_heads, self.d_v)).transpose(0, 2, 1, 3)

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # Back to the original input dimensions
        output = ops.Reshape()(output.transpose(0, 2, 1, 3), (bs, -1, self.n_heads * self.d_v))
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Cell):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super(_ScaledDotProductAttention, self).__init__()
        self.attn_dropout = nn.Dropout(1.0 - attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(Tensor(np.array([head_dim ** -0.5]).astype(np.float32)), requires_grad=lsa)
        self.lsa = lsa

    def construct(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        attn_scores = ops.BatchMatMul()(q, k) * self.scale

        if prev is not None:
            attn_scores = attn_scores + prev

        if attn_mask is not None:
            attn_scores = ops.MaskedFill()(attn_scores, attn_mask, -np.inf)

        if key_padding_mask is not None:
            attn_scores = ops.MaskedFill()(attn_scores, key_padding_mask.expand_dims(1).expand_dims(2), -np.inf)

        attn_weights = ops.Softmax(axis=-1)(attn_scores)
        attn_weights = self.attn_dropout(attn_weights)

        output = ops.BatchMatMul()(attn_weights, v)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights
