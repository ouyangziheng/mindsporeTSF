import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor
import math

class AutoCorrelation(nn.Cell):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = ops.ReduceMean(keep_dims=False)(corr, (1, 2))
        index = ops.TopK(sorted=True)(mean_value, top_k)[1]
        weights = ops.Stack(axis=-1)([mean_value[:, index[i]] for i in range(top_k)])
        # update corr
        tmp_corr = ops.Softmax(axis=-1)(weights)
        # aggregation
        delays_agg = ops.ZerosLike()(values)
        for i in range(top_k):
            pattern = ops.Roll(shift=(-int(index[i])), axis=-1)(values)
            delays_agg += pattern * ops.BroadcastTo(values.shape)(tmp_corr[:, i:i+1, None, None])
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = ops.BroadcastTo(values.shape)(ops.arange(length).view(1, 1, 1, -1))
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = ops.ReduceMean(keep_dims=False)(corr, (1, 2))
        weights = ops.TopK(sorted=True)(mean_value, top_k)[0]
        delay = ops.TopK(sorted=True)(mean_value, top_k)[1]
        # update corr
        tmp_corr = ops.Softmax(axis=-1)(weights)
        # aggregation
        tmp_values = ops.Tile()(values, (1, 1, 1, 2))
        delays_agg = ops.ZerosLike()(values)
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i:i+1, None, None]
            pattern = ops.GatherD()(tmp_values, -1, tmp_delay)
            delays_agg += pattern * ops.BroadcastTo(values.shape)(tmp_corr[:, i:i+1, None, None])
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch, head, channel, length = values.shape
        # index init
        init_index = ops.BroadcastTo(values.shape)(ops.arange(length).view(1, 1, 1, -1))
        # find top k
        top_k = int(self.factor * math.log(length))
        weights = ops.TopK(sorted=True)(corr, top_k)[0]
        delay = ops.TopK(sorted=True)(corr, top_k)[1]
        # update corr
        tmp_corr = ops.Softmax(axis=-1)(weights)
        # aggregation
        tmp_values = ops.Tile()(values, (1, 1, 1, 2))
        delays_agg = ops.ZerosLike()(values)
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i:i+1]
            pattern = ops.GatherD()(tmp_values, -1, tmp_delay)
            delays_agg += pattern * tmp_corr[..., i:i+1]
        return delays_agg

    def construct(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = ops.Zeros()((B, L - S, H, D), mindspore.float32)
            values = ops.Concat(axis=1)((values, zeros))
            keys = ops.Concat(axis=1)((keys, zeros))
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = ops.RFFT()(queries.transpose(0, 2, 3, 1), 1)
        k_fft = ops.RFFT()(keys.transpose(0, 2, 3, 1), 1)
        res = q_fft * ops.Conj()(k_fft)
        corr = ops.IRFFT()(res, 1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.transpose(0, 2, 3, 1), corr).transpose(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.transpose(0, 2, 3, 1), corr).transpose(0, 3, 1, 2)

        if self.output_attention:
            return (V, corr.transpose(0, 3, 1, 2))
        else:
            return (V, None)


class AutoCorrelationLayer(nn.Cell):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Dense(d_model, d_keys * n_heads)
        self.key_projection = nn.Dense(d_model, d_keys * n_heads)
        self.value_projection = nn.Dense(d_model, d_values * n_heads)
        self.out_projection = nn.Dense(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def construct(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
