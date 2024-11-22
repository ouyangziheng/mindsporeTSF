import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np


class moving_avg(nn.Cell):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, pad_mode='valid')

    def construct(self, x):
        # padding on the both ends of time series
        front = ops.tile(x[:, 0:1, :], (1, (self.kernel_size - 1) // 2, 1))
        end = ops.tile(x[:, -1:, :], (1, (self.kernel_size - 1) // 2, 1))
        x = ops.concat((front, x, end), axis=1)
        x = self.avg(ops.transpose(x, (0, 2, 1)))
        x = ops.transpose(x, (0, 2, 1))
        return x


class series_decomp(nn.Cell):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def construct(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Cell):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decomposition Kernel Size
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.enc_in = configs.enc_in
        self.period_len = 24

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        if self.individual:
            self.Linear_Seasonal = nn.CellList()
            self.Linear_Trend = nn.CellList()

            for i in range(self.enc_in):
                self.Linear_Seasonal.append(nn.Dense(self.seq_len, self.pred_len, has_bias=False))
                self.Linear_Trend.append(nn.Dense(self.seq_len, self.pred_len, has_bias=False))
        else:
            self.Linear_Seasonal = nn.Dense(self.seq_len, self.pred_len, has_bias=False)
            self.Linear_Trend = nn.Dense(self.seq_len, self.pred_len, has_bias=False)

    def construct(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = ops.transpose(seasonal_init, (0, 2, 1)), ops.transpose(trend_init, (0, 2, 1))

        if self.individual:
            seasonal_output = ops.zeros((seasonal_init.shape[0], seasonal_init.shape[1], self.pred_len), mindspore.float32)
            trend_output = ops.zeros((trend_init.shape[0], trend_init.shape[1], self.pred_len), mindspore.float32)
            for i in range(self.enc_in):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return ops.transpose(x, (0, 2, 1))  # to [Batch, Output length, Channel]
