import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dropout import embedded_dropout
from torch.nn.modules.loss import _Loss
sys.path.append('utils')
sys.path.append('models')

torch.set_printoptions(threshold=10000)

from .decoders import DecoderLayer, RelLearnableDecoderLayer, RelPartialLearnableDecoderLayer


def expected_length(length, death_rate):
    e_length = 0

    for l in range(length):
        survival_rate = 1.0 - (l + 1) / length * death_rate

        e_length += survival_rate

    return e_length


class LabelSmoothedCrossEntropy(_Loss):

    def __init__(self, output_size, label_smoothing):
        super(LabelSmoothedCrossEntropy, self).__init__()
        self.smoothing_value = label_smoothing / (output_size - 2)
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing
        self.padding_idx = -1

    def forward(self, scores, targets):

        gtruth = targets.view(-1)  # batch * time
        scores = scores.view(-1, scores.size(-1))  # batch * time X vocab_size

        lprobs = scores
        non_pad_mask = gtruth.ne(self.padding_idx)
        nll_loss = -lprobs.gather(1, gtruth.unsqueeze(1))[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

        eps_i = self.smoothing_value
        loss = (1. - self.label_smoothing) * nll_loss + eps_i * smooth_loss
        loss_data = nll_loss.data.item()

        return loss, loss_data


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


# The Memory Transformer LM implementation
class MemTransformerLM(nn.Module):

    def __init__(self, vocab, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1, word_dropout=0.0, label_smoothing=0.0,
                 scale_emb=True, death_rate=0.0):
        super(MemTransformerLM, self).__init__()

        n_token = len(vocab)
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.word_dropout = word_dropout
        self.label_smoothing = label_smoothing
        self.scale_emb = scale_emb
        self.word_emb = nn.Embedding(n_token, d_embed, padding_idx=vocab.pad_idx)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()

        if death_rate == 0:

            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, death_rate=death_rate)
                )
        else:

            e_length = expected_length(n_layer, death_rate)

            print("Stochastic Transformer with %.2f expected layers" % e_length)

            for l in range(n_layer):
                # linearly decay the death rate
                death_r = (l + 1.0) / n_layer * death_rate
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, death_rate=death_r)
                )

        self.sample_softmax = sample_softmax
        self.final_norm = nn.LayerNorm(self.d_model)

        # output layer (before softmax)
        self.out_layer = nn.Linear(d_model, n_token)
        if tie_weight:
            self.out_layer.weight = self.word_emb.weight

        self.same_length = same_length
        self.clamp_len = clamp_len

        # self.crit = torch.nn.CrossEntropyLoss(reduction='none')
        self.crit = LabelSmoothedCrossEntropy(n_token, self.label_smoothing)

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)

        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'
        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

            # Important:

        return new_mems

    # forward processing
    def _forward(self, dec_inp, mems=None, prev_input=None):
        # L x B
        qlen, bsz = dec_inp.size()

        # convert word indices to embeddings
        # word_emb = self.word_emb(dec_inp)
        word_emb = embedded_dropout(self.word_emb, dec_inp, dropout=self.word_dropout if self.training else 0)

        if self.scale_emb:
            word_emb.mul_(self.d_model ** 0.5)

        # memory len
        mlen = mems[0].size(0) if mems is not None else 0

        # total length: memory + current input
        klen = mlen + qlen
        # all units having the same attention range
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen)
                             + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None]  # -1
        # otherwise: the furtherest input has large memory
        # the nearest input has access to only the buffered memory (still better than RNN)
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]

        # pad_mask
        if prev_input is not None:
            full_seq = torch.cat([prev_input, dec_inp])
        else:
            full_seq = dec_inp
        pad_mask = full_seq.eq(0).byte()  # L x B

        if pad_mask.long().sum() > 0:
            dec_attn_mask = dec_attn_mask + pad_mask.unsqueeze(0)
            dec_attn_mask = dec_attn_mask.gt(0)

        dec_attn_mask = dec_attn_mask.bool()

        hids = []

        # attention forward

        # positional sequence (arange from n to 0)
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                               dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)

        # positional embedding (R)
        pos_emb = self.pos_emb(pos_seq)

        # apply dropout on both
        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        # the first hiddens are the embeddings
        hids.append(core_out)
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, pos_emb, self.r_w_bias,
                             self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
            # core_out = layer(core_out, pos_emb, dec_attn_mask=dec_attn_mask, mems=mems_i)
            hids.append(core_out)

        # core_out = self.drop(core_out)
        core_out = self.final_norm(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, target_weight, *mems, prev_input=None):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems: mems = self.init_mems()
        # target_weight = None
        hidden, new_mems = self._forward(data, mems=mems, prev_input=prev_input)
        # Return immediately if None
        if target is None:
            if new_mems is None:
                return [hidden]
            else:
                return [hidden] + new_mems

        tgt_len = target.size(0)

        # predictive hidden states (only take the necessary states)
        pred_hid = hidden[-tgt_len:]

        # flatten and take only the states that we need to compute softmax on
        pred_hid = pred_hid.view(-1, pred_hid.size(-1))
        target = target.view(-1)
        if target_weight is not None:
            flattened_mask = target_weight.view(-1)
            non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)

            clean_pred_hid = pred_hid.index_select(0, non_pad_indices)
            clean_target = target.index_select(0, non_pad_indices)

        else:
            clean_pred_hid = pred_hid
            clean_target = target

        logit = self.out_layer(clean_pred_hid).float()

        loss, nll = self.crit.float()(F.log_softmax(logit, dim=-1), clean_target)

        if new_mems is None:
            return [loss, nll]
        else:
            return [loss, nll] + new_mems

    def greedy_step(self, data, *mems):
        # L x B
        if not mems: mems = self.init_mems()
        hidden, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-1:]

        # flatten and take only the states that we need to compute softmax on
        pred_hid = pred_hid.view(-1, pred_hid.size(-1))
        logit = self.out_layer(pred_hid).float()
        prob = F.log_softmax(logit, dim=-1)

        output = torch.argmax(prob, dim=-1, keepdim=True)  # 1 x 1 at this point

        if new_mems is None:
            return [output]
        else:
            return [output] + new_mems

    def step(self, data, decoder_state):
        # L x B
        if not decoder_state:
            # this shouldn't happen (conditional model)
            mems = self.init_mems()
        else:
            mems = decoder_state.mems

        # for mem in mems:
        #     print(mem.size())

        # take the final step of the sequence as current input (no re-calculation)
        input_ = data[:, -1].unsqueeze(1).t()
        # input_ = data.t()
        # print(input_.size())

        # print(data.size())
        # print(mems[0].size())
        hidden, new_mems = self._forward(input_, mems=mems, prev_input=decoder_state.seq)
        # print(new_mems[0].size())

        pred_hid = hidden[-1:]

        # flatten and take only the states that we need to compute softmax on
        pred_hid = pred_hid.view(-1, pred_hid.size(-1))
        logit = self.out_layer(pred_hid).float()
        prob = F.log_softmax(logit, dim=-1)
        decoder_state.update_mems(new_mems, input_)

        if new_mems is None:
            return [prob]
        else:
            return [prob] + new_mems

    def create_decoder_state(self, src, beam_size=1, dec_state=None):

        if dec_state is None or dec_state.mems is None:
            mems = self.init_mems()
        else:
            mems = dec_state.mems  # each memory size should be T x beam_size x H

            # repeat the source
            beam_size = mems[0].size(1)
            src = src.repeat(1, beam_size)

        # forward pass through the source sentence
        ret = self.forward(src, None, None, *mems)
        hiddens, mems = ret[0], ret[1:]

        # if dec_state is None:
        #     dec_state = DecoderState(mems, beam_size=beam_size)
        # else:
        #     dec_state.update_mems(mems, beam_size=beam_size)

        # always return the decoder state
        clone = True if dec_state is None else False
        dec_state = DecoderState(mems, src, src.device, beam_size=beam_size, clone=clone)

        return dec_state


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.
    Modules need to implement this to utilize beam search decoding.
    """
    def __init__(self, mems, src, device, beam_size=1, clone=False):
        self.mems = mems
        self.device = device
        self.beam_size = beam_size
        self.seq = src  # T x B

        # only works with bsz 1 (at the moment, maybe)
        # if clone:
            # bsz = 1
        batch_size = mems[0].size(1)
        new_order = torch.arange(batch_size).view(-1, 1).repeat(1, self.beam_size).view(-1)
        new_order = new_order.to(device)

        self.seq = self.seq.index_select(1, new_order)
        for i, mem in enumerate(self.mems):
            self.mems[i] = mem.index_select(1, new_order)

    def update_mems(self, mems, new_input=None):
        self.mems = mems
        if new_input is not None:
            self.seq = torch.cat([self.seq, new_input], dim=0)

    def _reorder_incremental_state(self, reorder_state):
        # print("Reordering incremental state")
        for i, mem in enumerate(self.mems):
            self.mems[i] = mem.index_select(1, reorder_state)
        self.seq = self.seq.index_select(1, reorder_state)

    def _retain_best_beam(self, best_beam_id):
        for i, mem in enumerate(self.mems):
            self.mems[i] = mem[:, best_beam_id, :].unsqueeze(1)

    def reset_memory(self):
        # for i, mem in enumerate(self.mems):
        #     self.mems[i] = None
        self.mems = None
        return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    #
    # empty = torch.empty(0)
    # print(empty)

    # data = torch.LongTensor(data_len * B).random_(0, args.n_token).to(device)
    # diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)
    #
    # cutoffs = [args.n_token // 2]
    # tie_projs = [False] + [True] * len(cutoffs)
    #
    # for div_val in [1, 2]:
    #     for d_embed in [200, 100]:
    #         model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
    #                                  args.d_model, args.d_head, args.d_inner, args.dropout,
    #                                  dropatt=args.dropout, tie_weight=True,
    #                                  d_embed=d_embed, div_val=div_val,
    #                                  tie_projs=tie_projs, pre_lnorm=True,
    #                                  tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
    #                                  cutoffs=cutoffs, attn_type=0).to(device)
    #
    #         print(sum(p.numel() for p in model.parameters()))
    #
    #         mems = tuple()
    #         for idx, (inp, tgt, seqlen) in enumerate(diter):
    #             print('batch {}'.format(idx))
    #             out = model(inp, tgt, *mems)
    #             mems = out[1:]
