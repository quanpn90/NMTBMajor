# coding: utf-8
import argparse
from data_utils import get_lm_corpus, create_corpus


parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--outdir', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')

args = parser.parse_args()
# corpus = get_lm_corpus(args.data)

corpus = create_corpus(args.data, args.outdir)

