import functools
import os, shutil
import numpy as np
import re
import torch


# this function is borrowed from Facebook
# avoid jumping into the middle of a character
def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')


def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return get_logger(log_path=os.path.join(dir_path, 'log.txt'))


def save_checkpoint(model, optimizer, path, epoch):
    torch.save(model, os.path.join(path, 'model_{}.pt'.format(epoch)))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_{}.pt'.format(epoch)))


def scale_grad(parameters, denominator):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    for p in parameters:
        p.grad.data.div_(denominator)
        # print(p.grad.data)
    # if norm_type == inf:
    #     total_norm = max(p.grad.data.abs().max() for p in parameters)
    # else:
    #     total_norm = 0
    #     for p in parameters:
    #         param_norm = p.grad.data.norm(norm_type)
    #         total_norm += param_norm.item() ** norm_type
    #     total_norm = total_norm ** (1. / norm_type)
    # clip_coef = max_norm / (total_norm + 1e-6)
    # if clip_coef < 1:
    #     for p in parameters:
    #         p.grad.data.mul_(clip_coef)
    # return total_norm
    return


def checkpoint_paths(path, pattern=r'checkpoint_ppl_(\d+).(\d+)_xl.pt'):
    """Retrieves all checkpoints found in `path` directory.
    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)

    files = list()

    for fname in os.listdir(path):
        cur_path = os.path.join(path, fname)
        if os.path.isdir(cur_path):
            continue
        elif "ppl" in fname:
            files.append(fname)

    # sort py perplexity (ascending)
    files = sorted(files, key=lambda s: float(s.split("_")[2]))

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = int(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    # return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]
    return [os.path.join(path, x[1]) for x in entries]


def optimize_model(model):

    def replace_layer_norm(m, name):

        replacable = True
        try:
            from apex.normalization.fused_layer_norm import FusedLayerNorm
        except ImportError:
            replacable = False

        if replacable:
            for attr_str in dir(m):
                target_attr = getattr(m, attr_str)
                if type(target_attr) == torch.nn.LayerNorm:
                    setattr(m, attr_str, FusedLayerNorm(target_attr.normalized_shape,
                                                        eps=target_attr.eps,
                                                        elementwise_affine=target_attr.elementwise_affine))
                    # setattr(m, attr_str,
                    #         SynchronizedBatchNorm2d(target_attr.num_features, target_attr.eps, target_attr.momentum,
                    #                                 target_attr.affine))
            for n, ch in m.named_children():
                replace_layer_norm(ch, n)

    replace_layer_norm(model, "Transformer")