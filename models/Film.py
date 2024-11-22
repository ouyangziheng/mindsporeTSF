import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.FourierCorrelation import SpectralConv1d, SpectralConvCross1d, SpectralConv1d_local, SpectralConvCross1d_local
from layers.mwt import MWT_CZ1d_cross, mwt_transform
from layers.SelfAttention_Family import FullAttention, ProbAttention
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import numpy as np

class Model(nn.Cell):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        configs.ab = 2

        if configs.ab == 0:
            encoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                              seq_len=self.seq_len, modes1=configs.modes1)
            decoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                              seq_len=self.seq_len // 2 + self.pred_len, modes1=configs.modes1)
            decoder_cross_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                                    seq_len_q=self.seq_len // 2 + self.pred_len,
                                                    seq_len_kv=self.seq_len, modes1=configs.modes1)
        elif configs.ab == 1:
            encoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                              seq_len=self.seq_len)
            decoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                              seq_len=self.seq_len // 2 + self.pred_len)
            decoder_cross_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                                output_attention=False, configs=configs)
        elif configs.ab == 2:
            encoder_self_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            decoder_self_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            decoder_cross_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                                    seq_len_q=self.seq_len // 2 + self.pred_len,
                                                    seq_len_kv=self.seq_len)
        elif configs.ab == 3:
            encoder_self_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention, configs=configs)
            decoder_self_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention, configs=configs)
            decoder_cross_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                                output_attention=False, configs=configs)
        elif configs.ab == 4:
            encoder_self_att = FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                             output_attention=configs.output_attention)
            decoder_self_att = FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                             output_attention=configs.output_attention)
            decoder_cross_att = FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                              output_attention=configs.output_attention)
        elif configs.ab == 8:
            encoder_self_att = ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                             output_attention=configs.output_attention)
            decoder_self_att = ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                             output_attention=configs.output_attention)
            decoder_cross_att = ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                              output_attention=configs.output_attention)
        elif configs.ab == 5:
            encoder_self_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                                   seq_len_q=self.seq_len, seq_len_kv=self.seq_len)
            decoder_self_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                                   seq_len_q=self.seq_len // 2 + self.pred_len,
                                                   seq_len_kv=self.seq_len // 2 + self.pred_len)
            decoder_cross_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                                    seq_len_q=self.seq_len // 2 + self.pred_len,
                                                    seq_len_kv=self.seq_len)
        elif configs.ab == 6:
            encoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                              seq_len=self.seq_len, modes1=configs.modes1)
            decoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                              seq_len=self.seq_len // 2 + self.pred_len, modes1=configs.modes1)
            decoder_cross_att = SpectralConvCross1d_local(in_channels=configs.d_model, out_channels=configs.d_model,
                                                          seq_len_q=self.seq_len // 2 + self.pred_len,
                                                          seq_len_kv=self.seq_len, modes1=configs.modes1)
        elif configs.ab == 7:
            encoder_self_att = mwt_transform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_self_att = mwt_transform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_cross_att = MWT_CZ1d_cross(in_channels=configs.d_model, out_channels=configs.d_model,
                                               seq_len_q=self.seq_len // 2 + self.pred_len, seq_len_kv=self.seq_len,
                                               modes1=configs.modes1, ich=configs.d_model, base=configs.base,
                                               activation=configs.cross_activation)
        elif configs.ab == 8:
            encoder_self_att = SpectralConv1d_local(in_channels=configs.d_model, out_channels=configs.d_model,
                                                    seq_len=self.seq_len, modes1=configs.modes1)
            decoder_self_att = SpectralConv1d_local(in_channels=configs.d_model, out_channels=configs.d_model,
                                                    seq_len=self.seq_len // 2 + self.pred_len, modes1=configs.modes1)
            decoder_cross_att = SpectralConvCross1d_local(in_channels=configs.d_model, out_channels=configs.d_model,
                                                          seq_len_q=self.seq_len // 2 + self.pred_len,
                                                          seq_len_kv=self.seq_len, modes1=configs.modes1)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Dense(configs.d_model, configs.c_out, has_bias=True)
        )

    def construct(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                  enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_mark_enc = ops.zeros((x_enc.shape[0], x_enc.shape[1], 4), mindspore.float32)
        x_dec = ops.zeros((x_enc.shape[0], 48 + 720, x_enc.shape[2]), mindspore.float32)
        x_mark_dec = ops.zeros((x_enc.shape[0], 48 + 720, 4), mindspore.float32)

        # decomp init
        mean = ops.ReduceMean(keep_dims=True)(x_enc, 1).repeat((1, self.pred_len, 1))
        zeros = ops.zeros((x_dec.shape[0], self.pred_len, x_dec.shape[2]), mindspore.float32)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = ops.concat((trend_init[:, -self.label_len:, :], mean), axis=1)
        seasonal_init = ops.concat((seasonal_init[:, -self.label_len:, :], zeros), axis=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
