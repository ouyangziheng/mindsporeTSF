import mindspore
import mindspore.nn as nn
from layers.Embed import PositionalEmbedding
import numpy as np



class Model(nn.Cell):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.d_model = 128
        self.n_heads = 4
        self.e_layers = 2
        self.d_layers = 2
        self.d_ff = 256

        self.transformer_model = nn.Transformer(encoder_layers=self.e_layers, decoder_layers=self.d_layers, d_model=self.d_model, n_head=self.n_heads, dim_feedforward=self.d_ff, batch_first=True)

        self.pe = PositionalEmbedding(self.d_model)

        self.input = nn.Dense(self.enc_in, self.d_model)
        self.output = nn.Dense(self.d_model, self.enc_in)



    def construct(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        batch_size = x_enc.shape[0]

        enc_inp = self.input(x_enc)
        enc_inp = enc_inp + self.pe(enc_inp)
        dec_inp = mindspore.Tensor(np.zeros((batch_size, self.pred_len, self.d_model)), mindspore.float32)
        dec_inp = dec_inp + self.pe(dec_inp)

        out = self.transformer_model(enc_inp, dec_inp)

        y = self.output(out)

        return y



