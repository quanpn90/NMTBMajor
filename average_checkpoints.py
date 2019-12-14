# coding: utf-8
import argparse
import time
import math
import os, sys
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from models.mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir, scale_grad, checkpoint_paths
import apex.amp as amp
import apex


parser = argparse.ArgumentParser(description='average.py')
parser.add_argument('--path', required=True,
                    help='Path to model .pt file')
parser.add_argument('--output', default='model.averaged.pt',
                    help="""Path to output averaged model""")


def build_model(args, vocab):
    cutoffs, tie_projs = [], [False]

    model = MemTransformerLM(vocab, args.n_layer, args.n_head, args.d_model,
                                  args.d_head, args.d_inner, args.dropout, args.dropatt,
                                  tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
                                  tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
                                  ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
                                  same_length=args.same_length, attn_type=args.attn_type,
                                  clamp_len=args.clamp_len, sample_softmax=args.sample_softmax,
                                  word_dropout=args.word_dropout, label_smoothing=args.label_smoothing)

    return model


def main():

    opt = parser.parse_args()

    existed_save_files = checkpoint_paths(opt.path)

    print(existed_save_files)

    models = existed_save_files
    n_models = len(models)

    print("Loading main model from %s ..." % models[0])
    checkpoint = torch.load(models[0], map_location=lambda storage, loc: storage)
    #
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    #
    main_checkpoint = checkpoint
    args = checkpoint['args']
    vocab = checkpoint['vocab']

    main_model = build_model(args, vocab)

    main_model.load_state_dict(checkpoint['model'])

    for i in range(1, len(models)):

        model = models[i]
        print("Loading model from %s ..." % models[i])
        checkpoint = torch.load(model, map_location=lambda storage, loc: storage)

        args = checkpoint['args']

        # delete optim information to save GPU memory
        if 'optimizer' in checkpoint:
            del checkpoint['optimizer']

        current_model = build_model(args, vocab)

        current_model.load_state_dict(checkpoint['model'])

        # Sum the parameter values
        for (main_param, param) in zip(main_model.parameters(), current_model.parameters()):
            main_param.data.add_(param.data)

    # Normalizing
    for main_param in main_model.parameters():
        main_param.data.div_(n_models)

    # Saving
    model_state_dict = main_model.state_dict()

    checkpoint = dict()
    checkpoint['model'] = model_state_dict
    checkpoint['optimizer'] = None
    checkpoint['amp'] = None
    checkpoint['args'] = args
    checkpoint['vocab'] = vocab

    torch.save(checkpoint, opt.output)


if __name__ == "__main__":
    main()

