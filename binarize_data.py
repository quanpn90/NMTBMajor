# coding: utf-8
import argparse
from data_utils import get_lm_corpus, create_corpus


parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')

parser.add_argument('-data_type', default="int64",
                    help="Input type for storing text (int64|int32|int|int16) to reduce memory load")
parser.add_argument('-format', default="mmem",
                    help="Save data format: mmem or raw. Binary should be used to load faster")

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                    help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")

parser.add_argument('-join_vocab', action='store_true', help='Using one dictionary for both source and target')
parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")


parser.add_argument('--outdir', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')

args = parser.parse_args()
# corpus = get_lm_corpus(args.data)

# corpus = create_corpus(args.data, args.outdir)
