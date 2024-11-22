import mindspore
import numpy as np
import math
from mindspore import Tensor, ops
import mindspore.common.dtype as mstype

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = (B, 1, L, L)
        self._mask = ops.triu(Tensor(np.ones(mask_shape), mstype.bool_), 1)
        if device != "cpu":
            self._mask = self._mask.to_device(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        mask = np.ones((B, H, L, scores.shape[-1]), dtype=bool)
        mask = np.triu(mask, 1)
        self._mask = Tensor(mask, mstype.bool_)
        if device != "cpu":
            self._mask = self._mask.to_device(device)

    @property
    def mask(self):
        return self._mask
    

class LocalMask():
    def __init__(self, B, L, S, device="cpu"):
        mask_shape = (B, 1, L, S)
        self.len = math.ceil(math.log2(L))
        mask1 = np.triu(np.ones(mask_shape, dtype=bool), k=1)
        mask2 = ~np.triu(np.ones(mask_shape, dtype=bool), k=-self.len)
        self._mask = Tensor(mask1 + mask2, mstype.bool_)
        if device != "cpu":
            self._mask = self._mask.to_device(device)

    @property
    def mask(self):
        return self._mask
