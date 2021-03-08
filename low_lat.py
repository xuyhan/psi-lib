import math
from typing import List

import numpy as np

from batched_real import HEReal

def _rotate(v: HEReal, n):
    return v.rot(n)
    #return v[-n:] + v[:-n]

def _mult(m: HEReal, s):
    return m.multiply_raw(list(s))
    # s = self._enc(list(s))
    # return [m[i] * s[i] for i in range(len(m))]

def _mult_const(m, c):
    return m.multiply_raw(c)
    # s = self._enc([c for _ in range(8192)])
    # return [m[i] * s[i] for i in range(len(m))]

def _zeros(ref: HEReal):
    return ref.zeros()
   # return [0 for _ in range(8192)]

def _add_ip(m1: HEReal, m2: HEReal):
    m1.add_in_place(m2)
    # assert(len(m1) == len(m2))
    # for i in range(len(m2)):
    #     m1[i] += m2[i]

# def _enc(self, l):
#     assert(len(l) <= 8192)
#     return l + [0 for _ in range(8192 - len(l))]

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

def permute(v, p):
    assert(len(v) == len(p))
    result = [0 for _ in range(len(v))]
    for i, w in enumerate(v):
        result[p[i]] = w
    return result

def concat(messages: List[HEReal], n: int) -> HEReal:
    m0 = messages[0]
    for i in range(1, len(messages)):
        t = _rotate(messages[i], i * n)
        _add_ip(m0, t)
    return m0

def convolve(conv: List[HEReal], kernel) -> HEReal:
    """
    @param conv: list of regions
    @param kernel: kernel, as 2D numpy array
    """
    result = None
    kernel_flat = kernel.flatten()
    for i in range(kernel.shape[0] * kernel.shape[1]):
        p = _mult_const(conv[i], kernel_flat[i])
        if result is None:
            result = p
        else:
            _add_ip(result, p)
    return result

def dense_to_sparse(message: HEReal, W: np.ndarray) -> List[HEReal]:
    """
    @param message: ciphertext
    @param W: weight matrix of dimension (d_out, d_in)
    """
    result = []
    for i in range(W.shape[0]):
        t = _mult(message, W[i])
        t = shift_sum(t, W.shape[1])
        result.append(t)
    return result

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
    W = np.zeros((d_out ** 2, d_in ** 2))
    k = kernel.shape[0]
    kernel_flat = kernel.flatten()

    for j in range(d_out * d_out):
        for i in range(k * k):
            W[j][(j // d_out * s + i // k) * d_in + j % d_out * s + i % k] = kernel_flat[i]

    return W


# if __name__ == '__main__':
#     '''
#         test if fast conv and dense conv give same result
#     '''
#     helper = LowLatOps(None)
#
#     s = 2
#     k = 3
#     d_in = 32
#
#     fm = np.arange(d_in ** 2).reshape(d_in, d_in)
#     v = helper._enc(list(fm.flatten()))
#     kernel = np.arange(k * k).reshape(k, k)
#     W = helper.conv_weights(d_in, kernel, s)
#     print(list(np.matmul(W, fm.flatten())[:30]))
#
#     gs = [helper._enc(g) for g in LowLatOps.find_groups(fm, kernel.shape[0], s)]
#     print(helper.convolve(gs, kernel)[:30])
#
#     print(W.shape)


'''
def sparse_to_dense(self, ts):
    result = self._zeros()
    for i, t in enumerate(ts):
        self._add_ip(result, self._rotate(t, i))
    return result
        
def stacked_to_il(message, W, n, d):
    assert(2 ** d >= n)

    # stack mult
    gap = 2 ** d - W.shape[1]
    num_mults = int(math.ceil(W.shape[0] / n))
    W = np.column_stack((W, np.zeros((W.shape[0], gap))))
    ts = []

    for i in range(num_mults):
        w = list(W[i:i + n].flatten())  #+ [0 for _ in range(gap)]
        t = mult(message, w)
        for j in range(d):
            add(t, rotate(t, 2 ** j))
        print(t[:100])
        exit(0)
        ts.append(t)

    # combine
    result = zeros()
    for i, t in enumerate(ts):
        add(result, rotate(t, i))

    # perm
    perm = [(i % n) * d + (i // n) for i in range(W.shape[0])]

    return result, perm

def il_to_sparse(message, W, n, perm):
    assert(n == W.shape[1])

    result = []
    for i in range(W.shape[0]):
        t = mult(message, permute(W[i], perm))
        t = shift_sum(t, n)
        result.append(t)
    return result
'''