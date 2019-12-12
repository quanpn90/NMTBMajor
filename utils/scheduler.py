from torch.optim.lr_scheduler import _LRScheduler


class InvSqrtAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):

        lr_start = optimizer.param_groups[0]['lr']
        self.init_lr = 512 ** (-0.5) * lr_start
        self.warmup_steps = warmup_steps
        self._step = 0

        super(InvSqrtAnnealingLR, self).__init__(optimizer, last_epoch)

    def step(self, last_epoch=-1):
        self._step += 1

        lr = self.get_lr()

        self.optimizer.param_groups[0]['lr'] = lr

    def get_lr(self):

        if self._step <= self.warmup_steps:
            lr = self.init_lr*self._step*self.warmup_steps**(-1.5)
        else:
            lr = self.init_lr * self._step ** (-0.5)

        return lr
