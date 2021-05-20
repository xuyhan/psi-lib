import math
from typing import List, Dict
import numpy as np
from real.batched_real import HEReal
from tqdm.notebook import tqdm
from utils.utils import init_ops, merge_ops


def _rotate(v: HEReal, n):
    return v.rot(n)


def _mult(m: HEReal, s):
    return m.multiply_raw(list(s))


def _mult_const(m, c):
    return m.multiply_raw(c)


def _zeros(ref: HEReal):
    return ref.zeros()


def _add_ip(m1: HEReal, m2: HEReal):
    m1.add_in_place(m2)


def shift_sum(v: HEReal, n: int):
    """
    @param v: ciphertext
    @param n: how many items to sum
    @return: the sum
    """
    n = 2 ** int(math.ceil(math.log2(n)))
    result = v
    while n > 1:
        n = n // 2
        _add_ip(result, _rotate(v, -n))
    return result


def rot_sum(v: HEReal, n: int, m: int):
    """
    @param v: ciphertext
    @param n: output size
    @param m: input size, require that m is n * 2^k for some k
    @return: the sum
    """
    ops = init_ops()

    t = m

    result = v
    while t > n:
        t = t // 2
        _add_ip(result, _rotate(v, -t))
        ops['addCC'] += 1
        ops['rots'] += 1

    return result, ops


def permute(v, p):
    assert (len(v) == len(p))
    result = [0 for _ in range(len(v))]
    for i, w in enumerate(v):
        result[p[i]] = w
    return result


def concat(messages: List[HEReal], n: int) -> (HEReal, Dict):
    ops = init_ops()
    m0 = messages[0]

    for i in range(1, len(messages)):
        t = _rotate(messages[i], i * n)
        _add_ip(m0, t)
        ops['rots'] += 1
        ops['addCC'] += 1
    return m0, ops


def unconcat(message: HEReal, length: int, n: int) -> (List[HEReal], Dict):
    if length % n != 0:
        raise Exception('Cannot unconcat %s to %s' % (length, n))
    ops = init_ops()

    messages = []

    for i in range(length // n):
        messages.append(_rotate(message, - i * n))
        ops['rots'] += 1
    return messages, ops


def convolve3D(conv: List[HEReal], d_out: int, filter) -> (HEReal, Dict):
    """
    @param conv: list of regions, expect each to be flattened in order of channel, d, d
    @param filter: set of kernels, expect to be of (channel, k, k)
    """
    if len(filter.shape) != 3:
        raise Exception('Expected 3D filter.')
    ops = init_ops()

    result = None
    kernel_flat = filter.flatten()

    for i in range(kernel_flat.shape[0]):
        c = [kernel_flat[i] for _ in range(d_out ** 2)]
        p = _mult_const(conv[i], c)
        ops['mulPC'] += 1
        if result is None:
            result = p
        else:
            _add_ip(result, p)
            ops['addCC'] += 1

    return result, ops


def dense_to_sparse(message: HEReal, W: np.ndarray) -> List[HEReal]:
    """
    @param message: ciphertext
    @param W: weight matrix of dimension (d_out, d_in)
    """
    result = []
    for i in tqdm(range(W.shape[0])):
        t = _mult(message, W[i])
        t = shift_sum(t, W.shape[1])
        result.append(t)
    return result


def sparse_to_dense(messages: List[HEReal], i, o) -> HEReal:
    """
    @param messages: list of ciphertexts
    @param i: original length
    @param o: output length
    """
    result = _zeros(messages[0])


def dense_to_dense(message: HEReal, W: np.ndarray) -> (HEReal, Dict):
    """
    @param message: ciphertext
    @param W: weight matrix of dimension (d_out, d_in)
    @param n_in: number of nodes in previous layer
    """

    n_out = W.shape[0]
    n_in = W.shape[1]
    n_in_ = n_out * (2 ** math.ceil(math.log2((n_in + n_out - 1) / n_out)))

    if n_in > message.slot_count():
        raise Exception('input size too large')
    if n_in_ > message.slot_count():
        raise Exception('cannot find good dimensions for: ' + str(n_out) + ', ' + str(n_in))

    result = None
    ops = init_ops()

    for i in tqdm(range(n_out)):
        row = [W[(i + j) % n_out, j] for j in range(n_in)]
        right_shift = i
        row = [0 for _ in range(right_shift)] + row
        t = _mult(_rotate(message, right_shift), row)
        ops['rots'] += 1
        ops['mulPC'] += 1

        if result is None:
            result = t
        else:
            _add_ip(result, t)
            ops['addCC'] += 1

    r, ops_ = rot_sum(result, n_out, n_in_)
    ops = merge_ops(ops, ops_)

    return r, ops


def find_groups(fm, k, s):
    """
    @param fm: feature map
    @param k: kernel size
    @param s: stride size
    """
    d_out = int(math.ceil((fm.shape[0] - k + 1) / s))

    return [
        [fm[i // k + (j // d_out) * s][i % k + (j % d_out) * s] for j in range(d_out ** 2)] for i in range(k * k)
    ]


def conv_weights(d_in, kernel, s):
    '''
    obtain weights from kernel, equivalent to Dense(d_in ** 2, d_out ** 2)
    '''
    d_out = int(math.ceil((d_in - kernel.shape[0] + 1) / s))
    print(d_out)
    W = np.zeros((d_out ** 2, d_in ** 2))
    k = kernel.shape[0]
    kernel_flat = kernel.flatten()

    for j in range(d_out * d_out):
        for i in range(k * k):
            W[j][(j // d_out * s + i // k) * d_in + j % d_out * s + i % k] = kernel_flat[i]

    return W


def stacked_to_il(message, W, n, d):
    assert (2 ** d >= n)

    # stack mult
    gap = 2 ** d - W.shape[1]
    num_mults = int(math.ceil(W.shape[0] / n))
    W = np.column_stack((W, np.zeros((W.shape[0], gap))))
    ts = []

    for i in range(num_mults):
        w = list(W[i:i + n].flatten())
        t = _mult(message, w)
        for j in range(d):
            _add_ip(t, _rotate(t, 2 ** j))
        ts.append(t)

    # combine
    result = _zeros()
    for i, t in enumerate(ts):
        _add_ip(result, _rotate(t, i))

    # perm
    perm = [(i % n) * d + (i // n) for i in range(W.shape[0])]

    return result, perm


def il_to_sparse(message, W, n, perm):
    assert (n == W.shape[1])

    result = []
    for i in range(W.shape[0]):
        t = _mult(message, permute(W[i], perm))
        t = shift_sum(t, n)
        result.append(t)
    return result
