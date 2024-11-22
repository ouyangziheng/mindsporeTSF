import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np

class Model(nn.Cell):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Dense(self.seq_len, self.pred_len)
        # Use this line if you want to visualize the weights
        self.Linear.weight.set_data(Tensor((1 / self.seq_len) * np.ones([self.pred_len, self.seq_len]), mindspore.float32))

    def construct(self, x):
        # x: [Batch, Input length, Channel]
        x = self.Linear(ops.transpose(x, (0, 2, 1)))
        x = ops.transpose(x, (0, 2, 1))
        return x  # [Batch, Output length, Channel]
