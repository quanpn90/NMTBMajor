import torch
import torch.nn as nn
import torch.nn.functional as F


def _rel_shift(x, zero_triu=False, type=1):
    # zero_pad size: [q_len, 1, bsz, n_head]

    if type == 1:
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)

        x_padded = torch.cat([zero_pad, x], dim=1   )

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        # fills the 'unnecessary' parts with zeros
        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]
    else:
        raise NotImplementedError
    return x


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


# Relative Multihead Attention
class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        # self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        # self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
            .view(qlen, klen, x.size(2), x.size(3))

        return x

    # efficient computation of B and D term using shift
    # x dimension: [q_len, k_len, bsz, n_head]
    def _rel_shift(self, x, zero_triu=False):

        # zero_pad size: [q_len, 1, bsz, n_head]
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)

        x_padded = torch.cat([zero_pad, x], dim=1)

        # x_padded:
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


# Relative Partially Learnable (from Transformer XL)
class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        """

            :param w: input embeddings (E) T x B x H
        :param r: relative encodings (R)
        :param r_w_bias: n_head * d_head
        :param r_r_bias: n_head * d_head (the global relative position bias)
        :param attn_mask:
        :param mems:
        :return:
        """
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            w_heads = self.qkv_net(self.layer_norm(cat))
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            w_heads = self.qkv_net(self.layer_norm(w))
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        # what is relative shift?
        # R is actually the
        # here the B and D are actually B~ and D~ in the paper
        # then shift them to efficiently get B and D
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                        attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score.float(), dim=1)

        # nan will happen ... because of the first positions (aligned right) they will have nothing to attend to
        nan_mask = torch.isnan(attn_prob)
        attn_prob = attn_prob.masked_fill(nan_mask, 0).type_as(attn_score)

        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        # attn_out = self.drop(attn_out)

        # Residual connection
        # output = w + attn_out
        output = attn_out

        return output


# Relative Partially Learnable (from Transformer XL)
class FastRelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(FastRelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        """
        :param w: input embeddings (E) T x B x H
        :param r: relative encodings (R)
        :param r_w_bias: n_head * d_head
        :param r_r_bias: n_head * d_head (the global relative position bias)
        :param attn_mask:
        :param mems:
        :return:
        """
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            w_heads = self.qkv_net(self.layer_norm(cat))
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            w_heads = self.qkv_net(self.layer_norm(w))
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.contiguous().view(qlen, bsz * self.n_head, self.d_head).transpose(0, 1)
        w_head_k = w_head_k.contiguous().view(klen, bsz * self.n_head, self.d_head).transpose(0, 1)
        w_head_v = w_head_v.contiguous().view(klen, bsz * self.n_head, self.d_head).transpose(0, 1)

        w_head_q = w_head_q.view(bsz, self.n_head, qlen, self.d_head)
        w_head_k = w_head_k.view(bsz, self.n_head, klen, self.d_head)

        # r_head_k is the projected positions (not depending on the tensors)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head).transpose(0, 1)  # qlen x n_head x d_head

        #### compute attention score
        # r_w_bias is [n_head, d_head]
        rw_head_q = w_head_q + r_w_bias.unsqueeze(1)  # qlen x bsz x n_head x d_head
        AC = torch.matmul(rw_head_q, w_head_k.transpose(2, 3))
        # AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        # w_head_q = [
        rr_head_q = w_head_q + r_r_bias.unsqueeze(1)
        # [bsz, n_head, q_len, d] > [bsz, n_head, q_len, k_len]
        BD = torch.matmul(rr_head_q, r_head_k.transpose(1, 2))
        # BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        # what is relative shift?
        # R is actually the
        # here the B and D are actually B~ and D~ in the paper
        # then shift them to efficiently get B and D
        # [bsz, n_head, q_len, k_len] to [q_len, k_len, bsz, n_head]
        BD = BD.transpose(0, 2).transpose(1, 3)
        BD = self._rel_shift(BD)
        BD = BD.transpose(0, 2).transpose(1, 3)

        # [bsz x n_head x qlen x klen]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # [qlen x klen x bsz x n_head]
        attn_score = attn_score.transpose(0, 2).transpose(1, 3)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                        attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        # [bsz x n_head x qlen x klen] again
        attn_score = attn_score.transpose(0, 2).transpose(1, 3)

        # [bsz x n_head x qlen x klen] again
        attn_prob = F.softmax(attn_score.float(), dim=-1)

        # nan will happen ... because of the first positions (aligned right) they will have nothing to attend to
        nan_mask = torch.isnan(attn_prob)
        attn_prob = attn_prob.masked_fill(nan_mask, 0).type_as(attn_score)

        attn_prob = self.dropatt(attn_prob)

        attn_prob = attn_prob.view(bsz * self.n_head, qlen, klen)
        # compute attention vector
        # attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = torch.bmm(attn_prob, w_head_v)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.transpose(0, 1).contiguous().view(qlen, bsz, self.d_model)
        # attn_vec = attn_vec.contiguous().view(
        #     attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        # attn_out = self.drop(attn_out)

        # Residual connection
        # output = w + attn_out
        output = attn_out

        return output


# Learnable (no sin/cos position encoding)
class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]  # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]  # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        # attn_out = self.drop(attn_out)

        output = attn_out
        # if self.pre_lnorm:
        #     ##### residual connection
        #     output = w + attn_out
        # else:
        #     ##### residual connection + layer normalization
        #     output = self.layer_norm(w + attn_out)

        return output

if __name__ == '__main__':

    bsz = 1
    n_head = 1

    qlen = 5
    klen = 10

    x = torch.arange(klen - 1, -1, -1.0).unsqueeze(0).repeat(qlen, 1)
    # print(x)
    x = x.unsqueeze(-1).unsqueeze(-1)
    # x = torch.Tensor(qlen, klen, bsz, n_head)
    # x.normal_(0, 1)

    x = _rel_shift(x, zero_triu=True)

    print(x)
    print(x.size())

    # a = torch.arange(5, -5, -1).unsqueeze(0)
    #
    # print(a)
    #
    # b = torch.arange(0, 5).unsqueeze(1)
    #
    # c = (a + b)
    #
    # print(torch.abs(c))