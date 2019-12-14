# coding: utf-8
import time
import math
import sys
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from models.mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir, scale_grad, checkpoint_paths, optimize_model
# from utils.data_parallel import BalancedDataParallel
from utils.scheduler import InvSqrtAnnealingLR
import apex.amp as amp
import apex
from option import create_parser

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

parser = create_parser()

args = parser.parse_args()
args.tied = not args.not_tied

if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.ext_len >= 0, 'extended context length must be non-negative'
assert args.batch_size % args.batch_chunk == 0

# args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
# args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
logging = create_exp_dir(args.work_dir,
                         scripts_to_save=[], debug=args.debug)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)

# Validate `--fp16` option
if args.fp16:
    if not args.cuda:
        print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
        args.fp16 = False
    else:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

device = torch.device('cuda' if args.cuda else 'cpu')

###############################################################################
# Load data
###############################################################################
# corpus = get_lm_corpus(args.data, args.dataset)
corpus = get_lm_corpus(args.data)
ntokens = len(corpus.vocab)
vocab = corpus.vocab
args.n_token = ntokens
eval_batch_size = 1 if (not args.no_order) else args.batch_size
bos_id = corpus.vocab.get_idx('<bos>')
eos_id = corpus.vocab.get_idx('<eos>')
args.bos_id = bos_id
args.eos_id = eos_id

tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len, order=(not args.no_order),
                              device=device, ext_len=args.ext_len,
                              bos_id=bos_id, eos_id=eos_id,
                              switchout=args.switchout)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len, order=(not args.no_order),
                              device=device, ext_len=args.ext_len, bos_id=bos_id, eos_id=eos_id)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len, order=(not args.no_order),
                              device=device, ext_len=args.ext_len, bos_id=bos_id, eos_id=eos_id)

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]


###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)


def init_embed(weight):
    nn.init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('LayerNorm') != -1 or classname.find('FusedLayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)


def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout


def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt

model = MemTransformerLM(vocab, args.n_layer, args.n_head, args.d_model,
                         args.d_head, args.d_inner, args.dropout, args.dropatt,
                         tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
                         tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
                         ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
                         same_length=args.same_length, attn_type=args.attn_type,
                         clamp_len=args.clamp_len, sample_softmax=args.sample_softmax,
                         word_dropout=args.word_dropout, label_smoothing=args.label_smoothing,
                         death_rate=args.layer_drop)

optimize_model(model)
model.apply(weights_init)
model.word_emb.apply(weights_init)  # ensure embedding init is not overridden by out_layer in case of weight sharing
args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

# if args.fp16:
#     model = model.half()

if args.multi_gpu:
    model = model.to(device)
    if args.gpu0_bsz >= 0:
        para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
                                          model, dim=1).to(device)
    else:
        para_model = nn.DataParallel(model, dim=1).to(device)
else:
    para_model = model.to(device)

#### optimizer
if args.optim.lower() == 'sgd':

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.mom)
elif args.optim.lower() == 'adam':

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optim.lower() == 'fused_adam':
    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=args.lr)
elif args.optim.lower() == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

#### Apex
model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level,
                                  keep_batchnorm_fp32=False, loss_scale="dynamic")

#### scheduler
if args.scheduler == 'cosine':
    # here we do not set eta_min to lr_min to be backward compatible
    # because in previous versions eta_min is default to 0
    # rather than the default value of lr_min 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     args.max_step, eta_min=args.eta_min)  # should use eta_min arg

elif args.scheduler == 'inv_sqrt':
    # originally used for Transformer (in Attention is all you need)
    # def lr_lambda(step):
    #     # return a multiplier instead of a learning rate
    #     init_lr = (512 ** (-0.5)) * args.lr
    #
    #     if step == 0 and args.warmup_step == 0:
    #         return 0.0
    #     else:
    #         return init_lr * (step ** (-0.5)) if step > args.warmup_step \
    #             else init_lr * step * (args.warmup_step ** (-1.5))
    scheduler = InvSqrtAnnealingLR(optimizer, args.warmup_step)

    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)

elif args.scheduler == 'constant':
    pass

# if args.cuda and args.fp16:
#     # If args.dynamic_loss_scale is False, static_loss_scale will be used.
#     # If args.dynamic_loss_scale is True, it will take precedence over static_loss_scale.
#     optimizer = FP16_Optimizer(optimizer,
#                                static_loss_scale = args.static_loss_scale,
#                                dynamic_loss_scale = args.dynamic_loss_scale,
#                                dynamic_loss_args = {'init_scale': 2 ** 16})

if len(args.pretrain) > 0:
    print("loading pretrained model from %s" % args.pretrain)
    pretrained_checkpoint = torch.load(args.pretrain)
    model.load_state_dict(pretrained_checkpoint['model'])
    del pretrained_checkpoint

if args.restart:
    # if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
    #     with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
        #         opt_state_dict = torch.load(f)
    #         optimizer.load_state_dict(opt_state_dict)
    # else:
    # if os.path.exists(os.path.join(args.restart_dir, 'checkpoint.pt')):

    if os.path.exists(args.restart_checkpoint):
        with open(args.restart_checkpoint, 'rb') as f:
            checkpoint = torch.load(f)
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['model'])
            amp.load_state_dict(checkpoint['amp'])
    else:
        print('Optimizer was not saved. Start from scratch.')

logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 100)
logging('#params = {}'.format(args.n_all_param))
logging('#non emb params = {}'.format(args.n_nonemb_param))


###############################################################################
# Training code
###############################################################################

def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    eval_iter.reset_order()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model.reset_length(args.eval_tgt_len,
                           args.ext_len + args.tgt_len - args.eval_tgt_len, args.mem_len)
    else:
        model.reset_length(args.eval_tgt_len,
                           args.ext_len, args.mem_len + args.tgt_len - args.eval_tgt_len)

    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = tuple()
        for i, (data, target, seq_len, weight) in enumerate(eval_iter):
            if 0 < args.max_eval_steps <= i:
                break
            ret = model(data, target, weight, *mems)
            loss, nll, mems = ret[0], ret[1], ret[2:]

            if args.no_order:
                mems = tuple()

            total_loss += nll
            total_len += weight.sum().item()

    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_loss / total_len


def train():
    # Turn on training mode which enables dropout.
    global train_step, best_val_loss, eval_start_time, log_start_time
    model.train()

    mems = tuple()

    tr_iter.reset_order()
    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter

    n_accumulated_words = 0
    denom = 8000
    total_words = 0
    train_loss = 0

    for batch, (data, target, seq_len, weight) in enumerate(train_iter):
        model.zero_grad()

        ret = para_model(data, target, weight, *mems)
        loss, nll, mems =  ret[0], ret[1], ret[2:]
        ntarget = weight.float().sum().item()
        loss = loss.float().sum().type_as(loss)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        train_loss += (loss.float().item())
        n_accumulated_words += ntarget
        total_words += ntarget

        # reset the memory if we don't care about order
        if args.no_order:
            mems = tuple()

        if n_accumulated_words >= args.batch_size_update:

            # div gradients to the number of accumulated
            scale_factor = n_accumulated_words
            scale_grad(amp.master_params(optimizer), scale_factor)
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
            n_accumulated_words = 0
            temp_accumulated = 0

            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()

            # step-wise learning rate annealing
            train_step += 1
            if args.scheduler in ['cosine', 'constant', 'dev_perf']:
                # linear warmup stage
                if train_step < args.warmup_step:
                    curr_lr = args.lr * train_step / args.warmup_step
                    optimizer.param_groups[0]['lr'] = curr_lr

                else:
                    if args.scheduler == 'cosine':
                        scheduler.step(train_step)

            elif args.scheduler == 'inv_sqrt':
                # scheduler.step(train_step)
                scheduler.step()

            if train_step % args.log_interval == 0:
                cur_loss = train_loss / total_words
                elapsed = time.time() - log_start_time
                log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                          '| ms/update {:5.2f} | loss {:5.2f}'.format(
                    epoch, train_step, batch + 1, optimizer.param_groups[0]['lr'],
                                       elapsed * 1000 / args.log_interval, cur_loss)
                # if args.dataset in ['enwik8', 'text8']:
                #     log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
                # else:
                log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
                logging(log_str)
                # train_loss = 0
                log_start_time = time.time()

            if train_step % args.eval_interval == 0:
                val_loss = evaluate(va_iter)
                logging('-' * 100)
                log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                          '| valid loss {:5.2f}'.format(
                    train_step // args.eval_interval, train_step,
                    (time.time() - eval_start_time), val_loss)
                # if args.dataset in ['enwik8', 'text8']:
                #     log_str += ' | bpc {:9.5f}'.format(val_loss / math.log(2))
                # else:
                log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
                logging(log_str)
                logging('-' * 100)
                # Save the model (always)
                checkpoint = dict()
                checkpoint['model'] = model.state_dict()
                checkpoint['optimizer'] = optimizer.state_dict()
                checkpoint['amp'] = amp.state_dict()
                checkpoint['args'] = args
                checkpoint['vocab'] = vocab
                val_ppl = math.exp(val_loss)
                checkpoint_name = 'checkpoint_ppl_%.6f_xl.pt' % val_ppl
                # checkpoint['vocab_size'] = args.n_token
                checkpoint_dir = args.work_dir
                with open(os.path.join(args.work_dir, checkpoint_name), 'wb') as f:
                    log_str = "Saving to file %s" % os.path.join(args.work_dir, checkpoint_name)
                    logging(log_str)
                    torch.save(checkpoint, f)
                existed_save_files = checkpoint_paths(checkpoint_dir)
                num_save_files = 5
                for save_file in existed_save_files[num_save_files:]:
                    print(" * Deleting old save file %s ...." % save_file)
                    os.remove(save_file)

                # now we have to delete the worst checkpoint:

                if not best_val_loss or val_loss < best_val_loss:
                    if not args.debug:
                        continue
                    best_val_loss = val_loss

                # dev-performance based learning rate annealing
                if args.scheduler == 'dev_perf':
                    scheduler.step(val_loss)

                eval_start_time = time.time()

            if train_step == args.max_step:
                break


# Loop over epochs.
train_step = 0
train_loss = 0
best_val_loss = None

log_start_time = time.time()
eval_start_time = time.time()

val_loss = evaluate(va_iter)
logging('-' * 100)
log_str = '| Eval at step {:>8d} | time: {:5.2f}s ' \
          '| valid loss {:5.2f}'.format(
    0,
    (time.time() - eval_start_time), val_loss)
# if args.dataset in ['enwik8', 'text8']:
#     log_str += ' | bpc {:9.5f}'.format(val_loss / math.log(2))
# else:
log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
logging(log_str)

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in itertools.count(start=1):
        train()
        if train_step == args.max_step:
            logging('-' * 100)
            logging('End of training')
            break
except KeyboardInterrupt:
    logging('-' * 100)
    logging('Exiting from training early')
#
# # Load the best saved model.
# with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
#     model = torch.load(f)
#     para_model = model.to(device)
#
# # Run on test data.
# test_loss = evaluate(te_iter)
# logging('=' * 100)
# if args.dataset in ['enwik8', 'text8']:
#     logging('| End of training | test loss {:5.2f} | test bpc {:9.5f}'.format(
#         test_loss, test_loss / math.log(2)))
# else:
#     logging('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(
#         test_loss, math.exp(test_loss)))
# logging('=' * 100)
