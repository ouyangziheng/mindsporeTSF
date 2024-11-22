import mindspore
# import mindspore.numpy as np
import numpy as original_np
from mindspore import Tensor, ops
import numpy as np

def augmentation(augment_time):
    if augment_time == 'batch':
        return BatchAugmentation()
    elif augment_time == 'dataset':
        return DatasetAugmentation()


class BatchAugmentation():
    def __init__(self):
        pass

    def freq_mask(self, x, y, rate=0.5, dim=1):
        xy = ops.concat((x, y), axis=1)
        xy_f = ops.fft_rfft(xy, dim)
        m = Tensor(np.random.uniform(0, 1, xy_f.shape) < rate, mindspore.float32)
        freal = ops.masked_fill(xy_f.real, m, 0)
        fimag = ops.masked_fill(xy_f.imag, m, 0)
        xy_f = ops.complex(freal, fimag)
        xy = ops.fft_irfft(xy_f, dim)
        return xy

    def freq_mix(self, x, y, rate=0.5, dim=1):
        xy = ops.concat((x, y), axis=dim)
        xy_f = ops.fft_rfft(xy, dim)

        m = Tensor(np.random.uniform(0, 1, xy_f.shape) < rate, mindspore.float32)
        amp = ops.abs(xy_f)
        _, index = ops.sort(amp, axis=dim, descending=True)
        dominant_mask = index > 2
        m = ops.bitwise_and(m, dominant_mask)
        freal = ops.masked_fill(xy_f.real, m, 0)
        fimag = ops.masked_fill(xy_f.imag, m, 0)

        b_idx = original_np.arange(x.shape[0])
        original_np.random.shuffle(b_idx)
        x2, y2 = x[b_idx], y[b_idx]
        xy2 = ops.concat((x2, y2), axis=dim)
        xy2_f = ops.fft_rfft(xy2, dim)

        m = ops.bitwise_not(m)
        freal2 = ops.masked_fill(xy2_f.real, m, 0)
        fimag2 = ops.masked_fill(xy2_f.imag, m, 0)

        freal += freal2
        fimag += fimag2

        xy_f = ops.complex(freal, fimag)

        xy = ops.fft_irfft(xy_f, dim)
        return xy

    def noise(self, x, y, rate=0.05, dim=1):
        xy = ops.concat((x, y), axis=1)
        noise_xy = (Tensor(np.random.uniform(0, 1, xy.shape), mindspore.float32) - 0.5) * 0.1
        xy = xy + noise_xy
        return xy

    def noise_input(self, x, y, rate=0.05, dim=1):
        noise = (Tensor(np.random.uniform(0, 1, x.shape), mindspore.float32) - 0.5) * 0.1
        x = x + noise
        xy = ops.concat((x, y), axis=1)
        return xy

    def vFlip(self, x, y, rate=0.05, dim=1):
        xy = ops.concat((x, y), axis=1)
        xy = -xy
        return xy

    def hFlip(self, x, y, rate=0.05, dim=1):
        xy = ops.concat((x, y), axis=1)
        xy = xy[:, ::-1, ...]  # Reverse along the specified dimension
        return xy

    def time_combination(self, x, y, rate=0.5, dim=1):
        xy = ops.concat((x, y), axis=dim)

        b_idx = original_np.arange(x.shape[0])
        original_np.random.shuffle(b_idx)
        x2, y2 = x[b_idx], y[b_idx]
        xy2 = ops.concat((x2, y2), axis=dim)

        xy = (xy + xy2) / 2
        return xy

    def magnitude_warping(self, x, y, rate=0.5, dim=1):
        pass

    def linear_upsampling(self, x, y, rate=0.5, dim=1):
        xy = ops.concat((x, y), axis=dim)
        original_shape = xy.shape
        start_point = original_np.random.randint(0, original_shape[1] // 2)

        xy = xy[:, start_point:start_point + original_shape[1] // 2, :]

        xy = ops.transpose(xy, (0, 2, 1))
        xy = ops.interpolate(xy, scales=(1.0, 2.0), mode='linear')
        xy = ops.transpose(xy, (0, 2, 1))
        return xy


class DatasetAugmentation():
    def __init__(self):
        pass

    def freq_dropout(self, x, y, dropout_rate=0.2, dim=0, keep_dominant=True):
        x, y = Tensor(x), Tensor(y)

        xy = ops.concat((x, y), axis=0)
        xy_f = ops.fft_rfft(xy, dim)

        m = Tensor(np.random.uniform(0, 1, xy_f.shape) < dropout_rate, mindspore.float32)

        freal = ops.masked_fill(xy_f.real, m, 0)
        fimag = ops.masked_fill(xy_f.imag, m, 0)
        xy_f = ops.complex(freal, fimag)
        xy = ops.fft_irfft(xy_f, dim)

        x, y = xy[:x.shape[0], :], xy[-y.shape[0]:, :]
        return x.asnumpy(), y.asnumpy()

    def freq_mix(self, x, y, x2, y2, dropout_rate=0.2):
        x, y = Tensor(x), Tensor(y)

        xy = ops.concat((x, y), axis=0)
        xy_f = ops.fft_rfft(xy, dim=0)
        m = Tensor(np.random.uniform(0, 1, xy_f.shape) < dropout_rate, mindspore.float32)
        amp = ops.abs(xy_f)
        _, index = ops.sort(amp, axis=0, descending=True)
        dominant_mask = index > 2
        m = ops.bitwise_and(m, dominant_mask)
        freal = ops.masked_fill(xy_f.real, m, 0)
        fimag = ops.masked_fill(xy_f.imag, m, 0)

        x2, y2 = Tensor(x2), Tensor(y2)
        xy2 = ops.concat((x2, y2), axis=0)
        xy2_f = ops.fft_rfft(xy2, dim=0)

        m = ops.bitwise_not(m)
        freal2 = ops.masked_fill(xy2_f.real, m, 0)
        fimag2 = ops.masked_fill(xy2_f.imag, m, 0)

        freal += freal2
        fimag += fimag2

        xy_f = ops.complex(freal, fimag)
        xy = ops.fft_irfft(xy_f, dim=0)
        x, y = xy[:x.shape[0], :], xy[-y.shape[0]:, :]
        return x.asnumpy(), y.asnumpy()
