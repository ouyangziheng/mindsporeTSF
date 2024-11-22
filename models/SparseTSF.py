__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Cell):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None, norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True, pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):

        super(Model, self).__init__()

        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        assert self.model_type in ['linear', 'mlp']

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, pad_mode='pad', padding=self.period_len // 2, has_bias=False)

        if self.model_type == 'linear':
            self.linear = nn.Dense(self.seg_num_x, self.seg_num_y, has_bias=False)
        elif self.model_type == 'mlp':
            self.mlp = nn.SequentialCell([
                nn.Dense(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Dense(self.d_model, self.seg_num_y)
            ])

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                                 max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                 n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                                 dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                                 attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                                 pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                                                 pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                                 subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                               max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                               n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                               dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                               attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                               pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                                               pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                               subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                           n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                           dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                           attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                           pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                                           pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                           subtract_last=subtract_last, verbose=verbose, **kwargs)

    def construct(self, x):  # x: [Batch, Input length, Channel]
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = ops.ReduceMean(keep_dims=True)(x, 1)
        x = ops.transpose(x - seq_mean, (0, 2, 1))

        # 1D convolution aggregation
        x = self.conv1d(x.view(-1, 1, self.seq_len)).view(-1, self.enc_in, self.seq_len) + x

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = ops.transpose(x.view(-1, self.seg_num_x, self.period_len), (0, 2, 1))

        # sparse forecasting
        if self.model_type == 'linear':
            y = self.linear(x)  # bc,w,m
        elif self.model_type == 'mlp':
            y = self.mlp(x)

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = ops.transpose(y, (0, 2, 1)).view(batch_size, self.enc_in, self.pred_len)

        # permute and denorm
        y = ops.transpose(y, (0, 2, 1)) + seq_mean

        return y
