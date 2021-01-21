import numpy as np
from numpy.polynomial import Polynomial
import random
import math
from sympy import poly, Poly, div
from sympy.abc import x

# First we set the parameters


class Params:
    def __init__(self, N: int, p: int, p_0: int, L: int):
        self.N = N
        self.p = p
        self.p_0 = p_0
        self.L = L
        self.delta = 2 ** p
        self.q_0 = 2 ** p_0
        self.q_L = self.q_0 * ((self.delta) ** L)
        self.poly_mod = polymod(N)
        self.P = 2 ** 11

        print("Summary====")
        print("delta: " + str(self.delta))
        print("q_L: " + str(self.q_L))

    def q_l(self, l: int):
        return self.q_0 * ((self.delta) ** l)

    def delta_l(self, l: int):
        return (self.delta) ** l


class Encoder:
    def __init__(self, params: Params):
        """
            M specifies the cyclotomic polynomial to use.
        """
        self.M = params.N * 2
        self.N = params.N
        self.params = params
        self.root = np.exp(2 * np.pi * 1j / self.M)
        self.basis = np.array(self.vandermonde(self.root, self.M)).T

    @staticmethod
    def vandermonde(e: np.complex128, M: int) -> np.array:
        N = M // 2
        matrix = []

        for i in range(N):
            root = e ** (2 * i + 1)
            matrix.append([root ** j for j in range(N)])

        return matrix

    def encode(self, v: np.array) -> Poly:
        if v.shape != (self.N // 2,):
            raise Exception('Cannot encode vector of incorrect shape.')
        v = self.mirror(v) * self.params.delta_l(1)
        v = self.project_to_basis(v)
        v = self.random_round(v)
        v = np.matmul(self.basis.T, v)

        return self.sigma_inverse(v)

        #return Poly.from_list(np.round(np.real(poly.all_coeffs())).astype(int), gens=x)

    def decode(self, p: Poly, real=True) -> np.array:
        p = (1/self.params.delta_l(1)) * p

        v = self.sigma(p)
        v = self.mirror_inv(v)
        if real:
            v = np.real(v)
        return v

    def sigma_inverse(self, b: np.array) -> Poly:
        A = Encoder.vandermonde(self.root, self.M)
        cs = np.linalg.solve(A, b)[::-1]
        cs = [np.round(np.real(c)) for c in cs]

        return Poly.from_list(cs, gens=x)

    def sigma(self, p: Poly) -> np.array:
        outputs = []
        p_np = Polynomial(np.array(p.all_coeffs()[::-1], dtype=float))

        for i in range(self.N):
            root = self.root ** (2 * i + 1)

            output = p_np(root)
            outputs.append(output)

        return np.array(outputs)

    def mirror(self, v: np.array) -> np.array:
        mirrored = [np.conjugate(element) for element in v[::-1]]
        return np.concatenate([v, mirrored])

    def mirror_inv(self, v: np.array) -> np.array:
        return v[:self.N // 2]

    def project_to_basis(self, v: np.array) -> np.array:
        return np.array([np.real(np.vdot(v, b) / np.vdot(b, b)) for b in self.basis])

    def random_round(self, v):
        residue = v - np.floor(v)

        choices = []
        for r in residue:
            choices.append(np.random.choice([r, r - 1], 1, p=[1 - r, r]))
        choices = np.array(choices).flatten()

        return [int(x) for x in (v - choices)]



def polymod(N: int) -> Poly:
    coef = [1] + [0 for _ in range(N - 1)] + [1]
    coef = np.array(coef, dtype=object)
    return Poly.from_list(coef, gens=x)

def debug_poly(poly: Poly, Q_l):
    cs = poly.coef
    for i in range(len(cs)):
        print('Power: ' + str(i))
        print('Coefficient: ' + str(int(cs[i])))
    print()

def special_mod(a, modulus):
    if a >= 0:
        return a % modulus
    if -a < modulus // 2:
        return a
    return a % modulus

def round_poly(poly: Poly, poly_mod: Poly, Q_l: int, debug=False):
    if poly.get_domain().__str__() != 'ZZ':
        raise Exception('Non-integer polynomial')
    if max(poly.all_coeffs()) > 0 and math.log2(max(poly.all_coeffs())) > 64:
        print('Warning: coefficient has exceeded 64 bits!')

    poly = div(poly, poly_mod)[1]
    #return poly

    cs = poly.all_coeffs()

    new_coef = []
    for c in cs:
        new_coef.append(special_mod(int(c), Q_l))

    return Poly.from_list(new_coef, gens=x)

def round_nearest(poly: Poly) -> Poly:
    coef_old = poly.all_coeffs()

    coef_new = []
    for c in coef_old:
        coef_new.append(int(round(c)))

    return Poly.from_list(coef_new, gens=x)

def fix_coef(coef: int, modulus: int):
    if coef > modulus // 2:
        return -(modulus - coef)
    return coef


class Gen:
    @staticmethod
    def gen_key(n: int) -> Poly:
        coeffs = np.random.choice([-1, 0, 1], n + 1)

        return Poly.from_list(coeffs, gens=x)

    @staticmethod
    def gen_error(n: int) -> Poly:
        #return Poly.from_list([ for _ in range(n+1)], gens=x)

        return Gen.gen_key(n)

    @staticmethod
    def gen_enc(n: int) -> Poly:
        return Poly.from_list([0 for _ in range(n+1)], gens=x)
        #return Gen.gen_key(n)

    @staticmethod
    def gen_a(n: int, Q_l: int) -> Poly:
        coeffs = [0] * (n + 1)

        for i in range(n + 1):
            coeffs[i] = random.randint(0, Q_l)

        return Poly.from_list(coeffs, gens=x)


class KeyGen:
    @staticmethod
    def gen(params: Params):
        N = params.N

        s = Gen.gen_key(N)
        e = Gen.gen_error(N)
        a = Gen.gen_a(N, params.q_L)
        b = (-a * s + e) % params.poly_mod # TODO

        a_prime = Gen.gen_a(N, params.q_L * params.P)
        e_prime = Gen.gen_error(N)

        b_prime = -a_prime * s + e_prime + params.P * s * s
        b_prime = round_nearest(b_prime)
        b_prime = round_poly(b_prime, params.poly_mod, params.q_L * params.P)

        pk = (b, a)
        sk = (1, s)
        ek = (b_prime, a_prime)


        return sk, pk, ek


class Ciphertext:
    def __init__(self, c0: Poly, c1: Poly, l: int):
        self.c0 = c0
        self.c1 = c1
        self.l = l

class CipherTemp:
    def __init__(self, c0: Poly, c1: Poly, c2: Poly, l: int):
        self.c0 = c0
        self.c1 = c1
        self.l = l
        self.c2 = c2


class Encryptor:
    def __init__(self, pk, params: Params):
        self.pk = pk
        self.params = params

    def encrypt(self, plaintext: Poly):
        v = Gen.gen_enc(self.params.N)

        e_0 = Gen.gen_error(self.params.N)
        e_1 = Gen.gen_error(self.params.N)

        c0 = plaintext + e_0 + v * self.pk[0]
        c1 = e_1 + v * self.pk[1]

        c0 = round_nearest(c0)
        c1 = round_nearest(c1)

        c0 = round_poly(c0, self.params.poly_mod, self.params.q_L)
        c1 = round_poly(c1, self.params.poly_mod, self.params.q_L)

        return Ciphertext(c0, c1, self.params.L)


class Decryptor:
    def __init__(self, sk, params: Params):
        self.sk = sk
        self.params = params

    def decrypt(self, ciphertext: Ciphertext):
        q_l = self.params.q_l(ciphertext.l)
        poly = round_poly((ciphertext.c0 + ciphertext.c1 * self.sk[1]),
                          self.params.poly_mod, q_l)

        poly_coefs = [fix_coef(c, q_l) for c in poly.all_coeffs()]
        return Poly.from_list(poly_coefs, gens=x)

    def decrypt_temp(self, ciphertext):
        q_l = self.params.q_l(ciphertext.l)
        poly = round_poly((ciphertext.c0 + ciphertext.c1 * self.sk[1] + ciphertext.c2 *
                           self.sk[1] * self.sk[1]),
                          self.params.poly_mod, q_l)
        poly_coefs = [fix_coef(np.real(c), q_l) for c in poly.coef]
        return Poly(poly_coefs)



class Evaluator:
    def __init__(self, ek, params: Params):
        self.ek = ek
        self.params = params
        self.encoder = Encoder(params)

    def add_raw(self, cipher: Ciphertext, raw: np.array):
        encoded = self.encoder.encode(raw, cipher.l)
        return Ciphertext(cipher.c0 + encoded, cipher.c1, cipher.l)

    def add(self, cipher1: Ciphertext, cipher2: Ciphertext) -> Ciphertext:
        if cipher1.l != cipher2.l:
            raise Exception("Level mismatch: %s != %s" % (cipher1.l, cipher2.l))

        q_l = self.params.q_l(cipher1.l)

        c0 = cipher1.c0 + cipher2.c0
        c1 = cipher1.c1 + cipher2.c1

        c0 = round_poly(c0, self.params.poly_mod, q_l)
        c1 = round_poly(c1, self.params.poly_mod, q_l)

        return Ciphertext(c0, c1, cipher1.l)

    def mul(self, cipher1: Ciphertext, cipher2: Ciphertext, debug=False) -> Ciphertext:
        if cipher1.l != cipher2.l:
            raise Exception("Level mismatch: %s != %s" % (cipher1.l, cipher2.l))

        q_l = self.params.q_l(cipher1.l)

        d0 = cipher1.c0 * cipher2.c0
        d1 = cipher1.c0 * cipher2.c1 + cipher1.c1 * cipher2.c0
        d2 = cipher1.c1 * cipher2.c1

        d0 = round_poly(d0, self.params.poly_mod, q_l)
        d1 = round_poly(d1, self.params.poly_mod, q_l)
        d2 = round_poly(d2, self.params.poly_mod, q_l)

        f0 = round_nearest((1.0 / self.params.P) * self.ek[0] * d2)
        f1 = round_nearest((1.0 / self.params.P) * self.ek[1] * d2)

        c0 = d0 + f0
        c1 = d1 + f1

        c0 = round_poly(c0, self.params.poly_mod, q_l, debug)
        c1 = round_poly(c1, self.params.poly_mod, q_l)

        return Ciphertext(c0, c1, cipher1.l)

    def square(self, cipher: Ciphertext):
        return self.mul(cipher, cipher)

    def rescale(self, cipher: Ciphertext, diff=-1):
        q_l = self.params.q_l(cipher.l + diff)
        scale = self.params.delta ** (diff)

        cipher.c0 = cipher.c0 * scale
        cipher.c1 = cipher.c1 * scale
        cipher.c0 = round_nearest(cipher.c0)
        cipher.c1 = round_nearest(cipher.c1)

        cipher.c0 = round_poly(cipher.c0, self.params.poly_mod, q_l)
        cipher.c1 = round_poly(cipher.c1, self.params.poly_mod, q_l)
        cipher.l += diff


params = Params(N=8, p=11, p_0=25, L=3)

encoder = Encoder(params)


sk, pk, ek = KeyGen.gen(params)

encryptor = Encryptor(pk, params)
decryptor = Decryptor(sk, params)
evaluator = Evaluator(ek, params)

p = encoder.encode(np.array([1, 2,3,.4]))
q = encoder.encode(np.array([0,-2,3,4]))

p_enc = encryptor.encrypt(p)
q_enc = encryptor.encrypt(q)

r_enc = evaluator.square(p_enc)
evaluator.rescale(r_enc)

r_dec = decryptor.decrypt(r_enc)
res = encoder.decode(r_dec)
print(res)

r_enc = evaluator.square(r_enc)

evaluator.rescale(r_enc)

r_dec = decryptor.decrypt(r_enc)
res = encoder.decode(r_dec)
print(res)
# #




