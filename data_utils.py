import os, sys
import glob

from collections import Counter, OrderedDict
import numpy as np
import torch
from utils.word_drop import switchout as sw
from utils.vocabulary import Vocab


def collate_tensors(vocab, data, bos_id, eos_id, device, align_right=True, switchout=0.0):
    lengths = [x.size(0) for x in data]
    max_length = max(lengths)

    # tensor size: T x B
    # tensor = data[0].new(max_length, len(data)).fill_(0)
    tensor = data[0].new(len(data), max_length).fill_(0)
    weight = tensor.new(*tensor.size()).fill_(0)

    for i in range(len(data)):
        data_length = data[i].size(0)

        if align_right:
            offset = max_length - data_length
        else:
            offset = 0  # align to the left

        tensor[i].narrow(0, offset, data_length).copy_(data[i])

        if bos_id > 0:
            bos_pos = torch.nonzero(data[i].eq(bos_id)).squeeze().item()
            eos_pos = torch.nonzero(data[i].eq(eos_id)).squeeze().item()
            length = eos_pos - bos_pos + 1
            weight[i].narrow(0, bos_pos + offset, length).fill_(1)
        else:
            weight[i].narrow(0, offset, data_length).fill_(1)

    # Directly switchout in collating. This actually changes the output label as well
    if switchout > 0:
        tensor = sw(tensor, vocab,  tau=switchout, transpose=False, offset=0)

    tensor = tensor.transpose(0, 1).contiguous().to(device)  # BxT to TxB
    weight = weight.transpose(0, 1).contiguous().to(device)
    input_ = tensor[:-1, :]
    target = tensor[1:, :]
    weight = weight[1:, :]

    return input_, target, max_length, weight


class LMOrderedIterator(object):
    def __init__(self, vocab, data, bsz, bptt, device='cpu', ext_len=None, bos_id=-1, eos_id=-1, **kwargs):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.vocab = vocab
        self.bsz = 1
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.data = data  # don't sort the data

        # batch allocation
        self.batches = []
        cur_batch = []  # a list to store the sentence ids
        cur_size = 0
        self.bos_id = bos_id
        self.eos_id = eos_id
        i = 0

        def _oversized(_cur_length, _cur_batch_size, max_length):

            if _cur_batch_size + _cur_length > max_length:
                return True

            return False

        while i < len(self.data):

            current_length = self.data[i].size(0)
            #
            oversized = _oversized(current_length, cur_size, self.bptt)
            #
            if oversized:

                self.batches.append(cur_batch)
                # reset the batch
                cur_batch = []
                cur_size = 0

            cur_batch.append(i)
            cur_size += current_length
            i = i+1

        # catch the last batch
        if len(cur_batch) > 0:
            self.batches.append(cur_batch)

        self.num_batches = len(self.batches)
        self.order = torch.arange(self.num_batches)
        self.cur_index = 0

    def get_batch(self, i):
        """
        :param i: the index of the mini batch
        :return: data_input, data_target, data_length, data_weight
        """

        sent_ids = self.batches[i]
        data = [self.data[i] for i in sent_ids]
        lengths = [x.size(0) for x in data]
        max_length = sum(lengths)

        # tensor size: T x 1
        # tensor = data[0].new(max_length, bsz).fill_(0)
        tensor = data[0].new(1, max_length)
        weight = tensor.new(*tensor.size()).fill_(0)

        # start from position 0
        offset = 0

        for i in range(len(data)):
            data_length = data[i].size(0)
            tensor[0].narrow(0, offset, data_length).copy_(data[i])

            if self.bos_id > 0:
                bos_pos = torch.nonzero(data[i].eq(self.bos_id)).squeeze().item()
                eos_pos = torch.nonzero(data[i].eq(self.eos_id)).squeeze().item()
                length = eos_pos - bos_pos + 1
                weight[0].narrow(0, offset + bos_pos, length).fill_(1)
            else:
                weight[0].narrow(0, offset, data_length).fill_(1)

            # move the offset to the next sentence
            offset = offset + data_length

        tensor = tensor.transpose(0, 1).contiguous().to(self.device)
        weight = weight.transpose(0, 1).contiguous().to(self.device)
        input_ = tensor[:-1, :]
        target = tensor[1:, :]
        weight = weight[1:, :]

        return input_, target, max_length, weight

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def next(self):
        if self.cur_index >= self.num_batches:
            self.cur_index = 0
            self.reset_order()

        batch = self.get_batch(self.cur_index)
        self.cur_index = self.cur_index + 1

        return batch
    #
    # # Variable length iteration
    # def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
    #     max_len = self.bptt + max_deviation * std
    #     i = start
    #     while True:
    #         # 95% of the time long bptt, 5% half
    #         bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
    #
    #         bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
    #         data, target, seq_len, weight = self.get_batch(i, bptt)
    #         i += seq_len
    #         yield data, target, seq_len, weight
    #         if i >= self.data.size(0) - 2:
    #             break

    def __iter__(self):
        # how do we get next batch ...
        for i in range(self.num_batches):
            batch = self.get_batch(i)
            yield batch

    def reset_order(self):
        return


class LMShuffledIterator(object):
    def __init__(self, vocab, data, bsz, bptt, device='cpu', ext_len=None, bos_id=-1, eos_id=-1, switchout=0.0, **kwargs):
        """
            data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.vocab = vocab
        self.data = data
        self.switchout = switchout
        self.align_right = True

        if self.align_right:
            print("[INFO] Batches are aligned to the right")
        else:
            print("[INFO] Batches are aligned to the left")

        # first: sort the data by size
        sorted_data = sorted(data, key=lambda x: x.size(0))
        self.data = sorted_data
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        self.batches = []
        self.multiplier = 8

        self.max_size = bptt  # maximum number of words in this minibatch

        # allocate the sentences into groups
        def _oversized(_cur_length, _n_sentences, _cur_batch_sizes, bsz_length, bsz_tokens):
            if _n_sentences > bsz_length:
                return True

            if _n_sentences == 0:
                return False

            max_size = max(_cur_batch_sizes)
            if (max(max(_cur_batch_sizes), max_size)) * (_n_sentences + 1) > bsz_tokens:
                return True

            return False

        # batch allocation
        cur_batch = []  # a list to store the sentence ids
        cur_batch_sizes = []
        i = 0
        while i < len(self.data):
            current_length = self.data[i].size(0)

            oversized = _oversized(current_length, len(cur_batch), cur_batch_sizes, self.bsz ,self.max_size)

            if oversized:
                # cut-off the current list to fit the multiplier
                current_size = len(cur_batch)
                scaled_size = max(
                    self.multiplier * (current_size // self.multiplier),
                    current_size % self.multiplier)
                batch_ = cur_batch[:scaled_size]
                self.batches.append(batch_)  # add this batch into the batch list

                cur_batch = cur_batch[scaled_size:]  # reset the current batch
                cur_batch_sizes = cur_batch_sizes[scaled_size:]

            cur_batch.append(i)
            cur_batch_sizes.append(current_length)

            i = i + 1

        # catch the last batch
        if len(cur_batch) > 0:
            self.batches.append(cur_batch)

        self.num_batches = len(self.batches)
        self.order = torch.randperm(self.num_batches)
        self.cur_index = 0

    def reset_order(self):
        self.order = torch.randperm(self.num_batches)

    def next(self):
        if self.cur_index >= self.num_batches:
            self.cur_index = 0
            self.reset_order()

        batch = self.get_batch(self.order[self.cur_index])
        self.cur_index = self.cur_index + 1

        return batch

    def get_batch(self, i):
        """
        :param i: index
        :return: a tensor
        """

        sent_ids = self.batches[i]
        data = [self.data[i] for i in sent_ids]
        # lengths = [x.size(0) for x in data]
        # max_length = max(lengths)
        #
        # # tensor size: T x B
        # # tensor = data[0].new(max_length, len(data)).fill_(0)
        # tensor = data[0].new(len(data), max_length).fill_(0)
        # weight = tensor.new(*tensor.size()).fill_(0)
        #
        # for i in range(len(data)):
        #     data_length = data[i].size(0)
        #
        #     if self.align_right:
        #         offset = max_length - data_length
        #     else:
        #         offset = 0  # align to the left
        #
        #     tensor[i].narrow(0, offset, data_length).copy_(data[i])
        #
        #     if self.bos_id > 0:
        #         bos_pos = torch.nonzero(data[i].eq(self.bos_id)).squeeze().item()
        #         eos_pos = torch.nonzero(data[i].eq(self.eos_id)).squeeze().item()
        #         length = eos_pos - bos_pos + 1
        #         weight[i].narrow(0, bos_pos + offset, length).fill_(1)
        #     else:
        #         weight[:, i].narrow(0, offset, data_length).fill_(1)
        #
        # tensor = tensor.transpose(0, 1).contiguous().to(self.device)
        # weight = weight.transpose(0, 1).contiguous().to(self.device)
        # input_ = tensor[:-1, :]
        # target = tensor[1:, :]
        # weight = weight[1:, :]
        input_, target, max_length, weight = collate_tensors(self.vocab, data, self.bos_id, self.eos_id, self.device,
                                                             align_right=self.align_right, switchout=self.switchout)

        # if self.switchout > 0:
        #     input_ = switchout(input_, self.vocab, tau=self.switchout, transpose=True)

        return input_, target, max_length, weight

    def __iter__(self):
        # how do we get next batch ...
        for i in range(self.num_batches):
            batch = self.get_batch(self.order[i])
            yield batch


class Corpus(object):
    def __init__(self, vocab=None, *args, **kwargs):

        if vocab is None:
            self.vocab = Vocab(*args, **kwargs)
        else:
            self.vocab = vocab

        self.train, self.valid = [], []

    def generate_data(self, path, update_vocab=True):
        # self.order = kwargs.get('order', True)

        # if self.dataset in ['ptb', 'wt2', 'enwik8', 'text8', 'bilingual_ted']:
        #     self.vocab.count_file(os.path.join(path, 'train.txt'))
        #     self.vocab.count_file(os.path.join(path, 'valid.txt'))
        #     self.vocab.count_file(os.path.join(path, 'test.txt'))
        # elif self.dataset == 'wt103':
        #     self.vocab.count_file(os.path.join(path, 'train.txt'))
        # elif self.dataset == 'lm1b':
        #     train_path_pattern = os.path.join(
        #         path, '1-billion-word-language-modeling-benchmark-r13output',
        #         'training-monolingual.tokenized.shuffled', 'news.en-*')
        #     train_paths = glob.glob(train_path_pattern)
        #     # the vocab will load from file when build_vocab() is called

        if update_vocab:
            self.vocab.count_file(os.path.join(path, 'train.txt'))
            self.vocab.build_vocab()

        self.train = self.vocab.encode_file(
            os.path.join(path, 'train.txt'))
        self.valid = self.vocab.encode_file(
            os.path.join(path, 'valid.txt'))
        # self.test = self.vocab.encode_file(
        #     os.path.join(path, 'test.txt'))

    def save(self, datadir):

        data = dict()

        data['train'] = self.train
        data['valid'] = self.valid
        data['vocab'] = self.vocab

        fn = os.path.join(datadir, 'cache.pt')
        torch.save(data, fn)

        vn = os.path.join(datadir, 'vocab.txt')
        self.vocab.write_to_file(vn)

    def load(self, datadir):

        fn = os.path.join(datadir, 'cache.pt')
        cache = torch.load(fn)

        self.train = cache['train']
        self.valid = cache['valid']
        self.vocab = cache['vocab']

    def get_iterator(self, split, *args, **kwargs):

        order = kwargs.get('order', True)

        if order:
            if split == 'train':
                data_iter = LMOrderedIterator(self.vocab, self.train, *args, **kwargs)
            elif split in ['valid', 'test']:
                data_iter = LMOrderedIterator(self.vocab, self.valid, *args, **kwargs)
        else:
            if split == 'train':
                data_iter = LMShuffledIterator(self.vocab, self.train, *args, **kwargs)
            elif split in ['valid', 'test']:
                data_iter = LMShuffledIterator(self.vocab, self.valid, *args, **kwargs)

        return data_iter


def get_lm_corpus(datadir):

    # this should be changed into the memory indexed dataset
    corpus = Corpus()
    corpus.load(datadir)
    # fn = os.path.join(datadir, 'cache.pt')
    print('Loading cached dataset...')
    # corpus = torch.load(fn)

    return corpus


def create_corpus(datadir, outdir):

    print('Producing dataset from %s...' % datadir)
    kwargs = dict()
    kwargs['special'] = ["<pad>", '<unk>', '<bos>', '<eos>']
    kwargs['lower_case'] = False

    corpus = Corpus(**kwargs)
    corpus.generate_data(datadir)

    corpus.save(outdir)
    #
    # fn = os.path.join(outdir, 'cache.pt')
    # torch.save(corpus, fn)

    return corpus


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='../data/text8',
                        help='location of the data corpus')

    args = parser.parse_args()

    corpus = get_lm_corpus(args.datadir)
    print('Vocab size : {}'.format(len(corpus.vocab.idx2sym)))
