import torch
import numpy as np
from numpy import pi
import timeit
import random
import sys
try:
    import resource
except ImportError:
    resource = None

from contextlib import contextmanager


def _check_param_device(param, old_param_device):
    """This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.

    Arguments:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device


def parameter_grads_to_vector(parameters):
    """Convert parameters to one vector

    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        if param.grad is not None:
            vec.append(param.view(-1))
        else:
            zrs = torch.zeros_like(param)
            vec.append(zrs.view(-1))
    return torch.cat(vec)


def vector_to_parameter_grads(vec, parameters):
    """Convert one vector to the parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'.format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # ensure that param requires grad
        assert param.requires_grad

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old grad of the parameter
        param.grad.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


def require_and_zero_grads(vs):
    for v in vs:
        v.requires_grad_(True)
        try:
            v.grad.zero_()
        except AttributeError:
            pass


@contextmanager
def temp_require_grad(vs):
    prev_grad_status = [v.requires_grad for v in vs]
    require_and_zero_grads(vs)
    yield
    for v, status in zip(vs, prev_grad_status):
        v.requires_grad_(status)


def bdot(x, y):
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze(1).squeeze(1)


def bfill_lowertriangle(A, vec):
    ii, jj = np.tril_indices(A.size(-2), k=-1, m=A.size(-1))
    A[..., ii, jj] = vec
    return A


def bfill_diagonal(A, vec):
    ii, jj = np.diag_indices(min(A.size(-2), A.size(-1)))
    A[..., ii, jj] = vec
    return A


def qk_to_q_qdot(qks, dt):

    return 0.5 * (qks[1:] + qks[:-1]), (1. / dt) * (qks[1:] - qks[:-1])


def peak_memory_mb() -> float:
    """
    Get peak memory usage for this process, as measured by
    max-resident-set size:
    https://unix.stackexchange.com/questions/30940/getrusage-system-call-what-is-maximum-resident-set-size
    Only works on OSX and Linux, returns 0.0 otherwise.
    """
    if resource is None or sys.platform not in ('linux', 'darwin'):
        return 0.0

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore

    if sys.platform == 'darwin':
        # On OSX the result is in bytes.
        return peak / 1_000_000

    else:
        # On Linux the result is in kilobytes.
        return peak / 1_000


def set_rng_seed(rng_seed: int) -> None:
    random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)


def time_series_norm(xs, dt, p=2):
    """
    Computes the Lp norm for a time series

    See Lee & Verleysen, Generalization of the Lp norm for time series and its
    application to Self-Organizing Maps
    https://pdfs.semanticscholar.org/713c/2c5546e34ae25d808d375fc071551681c7ec.pdf
    """
    assert xs.ndim == 1
    sumd = 0.0
    for i in range(xs.shape[0]):
        if i == 0:
            prevx = 0.0
        else:
            prevx = xs[i - 1]

        if i == xs.shape[0] - 1:
            nextx = 0.0
        else:
            nextx = xs[i + 1]
        if xs[i] * prevx <= 0:
            Am1 = dt / 2 * np.abs(xs[i])
        else:
            Am1 = dt / 2 * xs[i]**2 / (np.abs(xs[i]) + np.abs(prevx))

        if xs[i] * nextx <= 0.0:
            Ap1 = dt / 2 * np.abs(xs[i])
        else:
            Ap1 = dt / 2 * xs[i]**2 / (np.abs(xs[i]) + np.abs(nextx))

        sumd += (Am1 + Ap1)**p

    return sumd**(1 / p)


class Timer(object):

    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start


def wrap_to_pi2(inp, mask=None):
    """
    takes a tensor and wraps its values to
    [-pi, pi)
    Assumes inp.shape = (N, qdim)
    Assumes mask.shape = (qdim,)
    """

    if mask is None:
        mask = torch.ones(1, inp.shape[1])
    elif mask.dim() == 1:
        mask = mask.unsqueeze(0)

    mask = mask.type(inp.dtype)

    while True:
        geq = ((inp > pi).type(inp.dtype) * mask).detach()
        leq = ((inp < -pi).type(inp.dtype) * mask).detach()

        if geq.sum() + leq.sum() > 0:
            inp = inp + 2 * pi * leq - 2 * pi * geq
        else:
            break

    return inp


def wrap_to_pi(inp, mask=None):
    """Wraps to [-pi, pi)"""
    if mask is None:
        mask = torch.ones(1, inp.size(1))

    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    mask = mask.to(dtype=inp.dtype)
    val = torch.fmod((inp + pi) * mask, 2 * pi)
    neg_mask = (val * mask) < 0
    val = val + 2 * pi * neg_mask.to(val.dtype)
    val = (val - pi)
    inp = (1 - mask) * inp + mask * val
    return inp


def diffangles2(inp1, inp2, **kwargs):
    """
    computes the difference between two
    angles [in rad] accounting for the
    branch cut at pi
    """

    return wrap_to_pi(inp1 - inp2, **kwargs)


def diffangles(inp1, inp2, **kwargs):
    return wrap_to_pi2(inp1 - inp2, **kwargs)


def read_tb(path):
    """
    path : a tensorboard file OR a directory, where we will find all TB files
           of the form events.*
    """
    import pandas
    import numpy as np
    from glob import glob
    from collections import defaultdict
    import tensorflow as tf
    import os.path as osp
    if osp.isdir(path):
        fnames = glob(osp.join(path, "events.*"))
    elif osp.basename(path).startswith("events."):
        fnames = [path]
    else:
        raise NotImplementedError("Expected tensorboard file or directory containing them. Got %s" %
                                  path)
    tag2pairs = defaultdict(list)
    maxstep = 0
    for fname in fnames:
        for summary in tf.train.summary_iterator(fname):
            if summary.step > 0:
                for v in summary.summary.value:
                    pair = (summary.step, v.simple_value)
                    tag2pairs[v.tag].append(pair)

                maxstep = max(summary.step, maxstep)

    data = np.empty((maxstep, len(tag2pairs)))
    data[:] = np.nan
    tags = sorted(tag2pairs.keys())
    for (colidx, tag) in enumerate(tags):
        pairs = tag2pairs[tag]
        for (step, value) in pairs:
            data[step - 1, colidx] = value
    return pandas.DataFrame(data, columns=tags)


class objectview(object):

    def __init__(self, d):
        self.__dict__ = d
