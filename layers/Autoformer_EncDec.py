import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor

class MyLayerNorm(nn.Cell):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(MyLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm([channels])

    def construct(self, x):
        x_hat = self.layernorm(x)
        bias = ops.ReduceMean(keep_dims=True)(x_hat, 1).expand_as(x)
        return x_hat - bias


class MovingAvg(nn.Cell):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, pad_mode="valid")

    def construct(self, x):
        # padding on both ends of time series
        front = x[:, 0:1, :].tile((1, (self.kernel_size - 1) // 2, 1))
        end = x[:, -1:, :].tile((1, (self.kernel_size - 1) // 2, 1))
        x = ops.Concat(1)((front, x, end))
        x = self.avg(x.transpose(0, 2, 1))
        x = x.transpose(0, 2, 1)
        return x


class SeriesDecomp(nn.Cell):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def construct(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class SeriesDecompMulti(nn.Cell):
    """
    Series decomposition block for multiple kernel sizes
    """
    def __init__(self, kernel_size):
        super(SeriesDecompMulti, self).__init__()
        self.moving_avg = nn.CellList([MovingAvg(kernel, stride=1) for kernel in kernel_size])
        self.layer = nn.Dense(1, len(kernel_size))

    def construct(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.expand_dims(-1))
        moving_mean = ops.Concat(-1)(moving_mean)
        weights = nn.Softmax(axis=-1)(self.layer(x.expand_dims(-1)))
        moving_mean = ops.ReduceSum()(moving_mean * weights, -1)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Cell):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, has_bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, has_bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(1.0 - dropout)
        self.activation = ops.ReLU() if activation == "relu" else ops.GeLU()

    def construct(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(0, 2, 1))))
        y = self.dropout(self.conv2(y).transpose(0, 2, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Cell):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.CellList(attn_layers)
        self.conv_layers = nn.CellList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def construct(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Cell):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, has_bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, has_bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(1.0 - dropout)
        self.projection = nn.Conv1d(d_model, c_out, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False)
        self.activation = ops.ReLU() if activation == "relu" else ops.GeLU()

    def construct(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(0, 2, 1))))
        y = self.dropout(self.conv2(y).transpose(0, 2, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.transpose(0, 2, 1)).transpose(0, 2, 1)
        return x, residual_trend


class Decoder(nn.Cell):
    """
    Autoformer decoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.CellList(layers)
        self.norm = norm_layer
        self.projection = projection

    def construct(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
