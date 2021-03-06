# coding: utf-8
import argparse
import time
import math
import os, sys

import torch

from data_utils import get_lm_corpus
from models.    mem_transformer import MemTransformerLM
from utils.exp_utils import get_logger

parser = argparse.ArgumentParser(description='Generate from Transformer XL')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--input_file', type=str, default='./input.txt',
                    help='location of the input source file to translate')
parser.add_argument('--output_file', type=str, default='./output.txt',
                    help='location of the output file for translation')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8', 'bilingual_ted'],
                    help='dataset name')
parser.add_argument('--split', type=str, default='all',
                    choices=['valid', 'test'],
                    help='which split to evaluate')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=5,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--work_dir', type=str, required=True,
                    help='path to the work_dir')
parser.add_argument('--no_log', action='store_true',
                    help='do not log the eval result')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking')
##MODEL CONFIG (for now)
parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=50,
                    help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
parser.add_argument('--d_model', type=int, default=500,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=1000,
                    help='inner dimension in FF')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                         '2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')

def addone(f):
    for line in f:
        yield line
    yield None

args = parser.parse_args()
args.tied = not args.not_tied
if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.ext_len >= 0, 'extended context length must be non-negative'

device = torch.device("cuda" if args.cuda else "cpu")

# Get logger
logging = get_logger(os.path.join(args.work_dir, 'log.txt'),
                     log_=not args.no_log)

# Load dataset
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)

if 'bilingual' in args.dataset:
    bos_id = corpus.vocab.get_idx('<bos>')
    eos_id = corpus.vocab.get_idx('<eos>')
else:
    bos_id = -1
    eos_id = -1
eval_batch_size = args.batch_size
args.eval_tgt_len = args.tgt_len
# tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
#                               device=device, ext_len=args.ext_len, bos_id=bos_id, eos_id=eos_id)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
                              device=device, ext_len=args.ext_len, bos_id=bos_id, eos_id=eos_id)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
                              device=device, ext_len=args.ext_len, bos_id=bos_id, eos_id=eos_id)


# Load the best saved model.
with open(os.path.join(args.work_dir, 'checkpoint.pt'), 'rb') as f:
    # model_state_dict = torch.load(f)
    checkpoint = torch.load(f)
    model_args = checkpoint['args']
    model = MemTransformerLM(ntokens, model_args.n_layer, model_args.n_head, model_args.d_model,
                             model_args.d_head, model_args.d_inner, 0.0, 0.0,
                             tie_weight=model_args.tied, d_embed=model_args.d_embed, div_val=1.0,
                             tie_projs=[False], pre_lnorm=model_args.pre_lnorm, tgt_len=model_args.tgt_len,
                             ext_len=model_args.ext_len, mem_len=model_args.mem_len, cutoffs=[],
                             same_length=model_args.same_length, attn_type=model_args.attn_type,
                             clamp_len=model_args.clamp_len, sample_softmax=False)
    model.load_state_dict(checkpoint['model'])
model.backward_compatible()
model = model.to(device)

logging('Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}'.format(
       args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len))

model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
if args.clamp_len > 0:
    model.clamp_len = args.clamp_len
if args.same_length:
    model.same_length = True


###############################################################################
# Evaluation code
###############################################################################
# def evaluate(eval_iter):
#     # Turn on evaluation mode which disables dropout.
#     model.eval()
#     total_len, total_loss = 0, 0.
#     start_time = time.time()
#     with torch.no_grad():
#         mems = tuple()
#         for idx, (data, target, seq_len, weight) in enumerate(eval_iter):
#             ret = model(data, target, weight, *mems)
#             # print(weight)
#             loss, mems = ret[0], ret[1:]
#             loss = loss.sum()
#             total_loss += loss.float().item()
#             total_len += weight.sum().item()
#         total_time = time.time() - start_time
#     logging('Time : {:.2f}s, {:.2f}ms/segment'.format(
#             total_time, 1000 * total_time / (idx+1)))
#     return total_loss / total_len\

def translate(input_file, output_file):

    inread = open(input_file)
    outf = open(output_file, 'w')
    # with torch.no_grad():

    mems = tuple()
    counter = 0
    for line in addone(inread):

        if line is not None:
            words = corpus.vocab.tokenize(line)  # + ["<bos>"]

            # read in the source sentence
            src = corpus.vocab.convert_to_tensor(words).unsqueeze(1).contiguous().to(device)
            counter += 1

            print("SOURCE %d : %s" % (counter, line.strip()))
            print(src.size())

            # forward into model through the source
            with torch.no_grad():
                ret = model(src, None, None, *mems)
                hiddens, mems = ret[0], ret[1:]
                for mem in mems:
                    print(mem.size())

            # take the last hidden state
            dec_inp = corpus.vocab.convert_to_tensor(["<bos>"]).unsqueeze(1).contiguous().to(device)
            new_sentence = []
            while True:
                ret = model.greedy_step(dec_inp, *mems)

                dec_inp, mems = ret[0], ret[1:]

                dec_word = corpus.vocab.get_sym(dec_inp.squeeze().item())

                if dec_word == "<eos>" or len(new_sentence) >= 100:
                    break
                else:
                    # print(dec_word)
                    new_sentence += [dec_word]
                    continue

            output_sentence = " ".join(new_sentence)
            print("TRANSLATION %d: %s" % (counter, output_sentence))
            outf.write(output_sentence + "\n")
            print("")

    inread.close()

# Run on test data.
translate(args.input_file, args.output_file)
