import torch
import torch.nn.functional as F
import torch.nn as nn


class VariationalDropout(torch.nn.Module):
    def __init__(self, p=0.5, batch_first=False):
        super().__init__()
        self.p = p
        self.batch_first = batch_first

    def forward(self, x):
        if not self.training or not self.p:
            return x

        if self.batch_first:
            m = x.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        else:
            m = x.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.p)

        # scaling to ensure the expected value of the dropped units
        mask = m / (1 - self.p)

        return mask * x
