import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttn, RelLearnableMultiHeadAttn, RelPartialLearnableMultiHeadAttn
from .dropout import VariationalDropout


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # VariationalDropout(dropout),
            nn.Linear(d_inner, d_model),
            # nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        # self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        ##### layer normalization + positionwise feed-forward
        core_out = self.CoreNet(self.layer_norm(inp))

        ##### residual connection
        # output = core_out + inp
        output = core_out

        return output


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout)

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
                                                  **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout)

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, death_rate=0.0,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout)

        self.pos_attn_dropout = nn.Dropout(dropout)
        # self.pos_attn_dropout = VariationalDropout(dropout)
        self.pos_ffn_dropout = nn.Dropout(dropout)
        # self.pos_ffn_dropout = VariationalDropout(dropout)
        self.death_rate = death_rate


    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        coin = True
        residual_input = dec_inp  # before normalization

        if self.training:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        if coin:
            output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                                   attn_mask=dec_attn_mask,
                                   mems=mems)
            # output = self.dec_attn(dec_inp, r,
            #                        attn_mask=dec_attn_mask,
            #                        mems=mems)

            # rescale the output to get the expectation (stochastic layer)
            if self.training:
                output = output / (1 - self.death_rate)

            # residual connection
            output = self.pos_attn_dropout(output) + residual_input
            residual_input = output

            output = self.pos_ff(output)

            # rescale the output to get the expectation (stochastic layer)
            if self.training:
                output = output / (1 - self.death_rate)

            # residual connection
            output = self.pos_ffn_dropout(output) + residual_input
        else:
            output = residual_input

        return output