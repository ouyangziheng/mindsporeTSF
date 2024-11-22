import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
import numpy as np
import math
from utils.masking import TriangularCausalMask, ProbMask

class RevIN(nn.Cell):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def construct(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = Parameter(ops.Ones()(self.num_features, mindspore.float32), name='affine_weight')
        self.affine_bias = Parameter(ops.Zeros()(self.num_features, mindspore.float32), name='affine_bias')

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].expand_dims(1)
        else:
            mean_op = ops.ReduceMean(keep_dims=True)
            self.mean = mean_op(x, dim2reduce).detach()
        var_op = ops.ReduceVar(keep_dims=True, unbiased=False)
        self.stdev = ops.Sqrt()(var_op(x, dim2reduce) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class FullAttention(nn.Cell):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(keep_prob=1.0 - attention_dropout)

    def construct(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = ops.BatchMatMul()(queries, keys.transpose(0, 1, 3, 2))

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores = scores.masked_fill(attn_mask.mask, -np.inf)

        A = self.dropout(ops.Softmax(axis=-1)(scale * scores))
        V = ops.BatchMatMul()(A, values)

        if self.output_attention:
            return (V, A)
        else:
            return (V, None)

class ProbAttention(nn.Cell):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(keep_prob=1.0 - attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = ops.BroadcastTo((B, H, L_Q, L_K, E))(K.expand_dims(-3))
        index_sample = np.random.randint(L_K, size=(L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, np.arange(L_Q).reshape(-1, 1), index_sample, :]
        Q_K_sample = ops.BatchMatMul()(Q.expand_dims(-2), K_sample.swapaxes(-2, -1)).squeeze()

        # find the Top_k query with sparsity measurement
        M = Q_K_sample.max(-1) - Q_K_sample.sum(-1) / L_K
        M_top = ops.TopK(sorted=False)(M, n_top)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[ops.arange(B)[:, None, None], ops.arange(H)[None, :, None], M_top, :]
        Q_K = ops.BatchMatMul()(Q_reduce, K.swapaxes(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = ops.ReduceMean(keep_dims=False)(V, -2)
            context = ops.BroadcastTo((B, H, L_Q, V_sum.shape[-1]))(V_sum.expand_dims(-2)).copy()
        else:
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            context = ops.Cumsum()(V, -2)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores = scores.masked_fill(attn_mask.mask, -np.inf)

        attn = ops.Softmax(axis=-1)(scores)
        context_in[ops.arange(B)[:, None, None], ops.arange(H)[None, :, None], index, :] = ops.BatchMatMul()(attn, V).astype(context_in.dtype)
        if self.output_attention:
            attns = (ops.Ones()((B, H, L_V, L_V), mindspore.float32) / L_V).astype(attn.dtype)
            attns[ops.arange(B)[:, None, None], ops.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def construct(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = ops.Transpose()(queries, (0, 2, 1, 3))
        keys = ops.Transpose()(keys, (0, 2, 1, 3))
        values = ops.Transpose()(values, (0, 2, 1, 3))

        U_part = self.factor * math.ceil(math.log(L_K))  # c*ln(L_k)
        u = self.factor * math.ceil(math.log(L_Q))  # c*ln(L_q)

        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context, attn

class AttentionLayer(nn.Cell):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
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

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
