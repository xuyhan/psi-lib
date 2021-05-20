from typing import List
from sympy import Poly
from sympy.abc import x


def special_mod(a, modulus):
    a = a % modulus
    if a > modulus // 2:
        a = a - modulus
    return a


def fast_rem(coefs: List[int], n: int):
    k = len(coefs) - n
    quo = [0 for _ in range(len(coefs))]
    rem = coefs[k:]
    for i in range(k):
        rem[len(rem) - 1 - i] -= coefs[k - 1 - i]
        quo[len(quo) - 1 - i] = coefs[k - 1 - i]
    return quo, rem


class Polynomial:
    def __init__(self, poly: Poly, n: int):
        self.p = poly
        self.n = n

    def __str__(self):
        return self.p.__str__()

    @staticmethod
    def from_coef(coef, n: int):
        return Polynomial(Poly.from_list(coef, gens=x), n)

    def coefs(self):
        all_coeffs = self.p.all_coeffs()
        all_coeffs = [int(c) for c in all_coeffs]
        all_coeffs = [0 for _ in range((1 << self.n) - len(all_coeffs))] + all_coeffs
        return all_coeffs

    def apply(self, object):
        return self.p(object)

    def add(self, poly):
        if isinstance(poly, Polynomial):
            return Polynomial(self.p + poly.p, self.n)
        return Polynomial(self.p + poly, self.n)

    def mul(self, poly):
        if isinstance(poly, Polynomial):
            return Polynomial(self.p * poly.p, self.n)
            # NB: the commented version uses an implementation of negaconvolution fft from sagelib
            # which can be found at https://github.com/sagemath/sagelib/blob/master/sage/rings/polynomial/convolution.py
            # with some slight adjustments to their code, we can achieve great speedups!
            # l1 = self.coefs()[::-1]
            # l2 = poly.coefs()[::-1]
            # l3 = _negaconvolution_fft(l1, l2, self.n, True)
            # return Polynomial.from_coef(l3[::-1], self.n)

        return Polynomial(self.p * poly, self.n)

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.mul(other)

    def __neg__(self):
        return Polynomial(self.p.__neg__(), self.n)

    def __mod__(self, poly):
        return Polynomial(self.p.__mod__(poly.p), self.n)

    def rem(self, base):
        return self

    def wrap_coefs(self, modulus):
        cs = self.p.all_coeffs()
        new_coef = [special_mod(int(c), modulus) for c in cs]
        return Polynomial.from_coef(new_coef, self.n)

    def fix_coefs(self, modulus):
        def f(c):
            if c > modulus // 2:
                return -(modulus - c)
            return c

        return Polynomial.from_coef([f(c) for c in self.p.all_coeffs()], self.n)

    def round_coefs_nearest(self):
        coef_old = self.p.all_coeffs()
        coef_new = [int(round(c)) for c in coef_old]
        return Polynomial.from_coef(coef_new, self.n)

    def right_shift(self, k):
        coef_old = self.p.all_coeffs()
        coef_new = [int(c) >> k for c in coef_old]
        return Polynomial.from_coef(coef_new, self.n)

    def debug(self):
        cs = self.p.all_coeffs()
        for i in range(len(cs)):
            print('Power: ' + str(i))
            print('Coefficient: ' + str(int(cs[i])))
        print()
