import numpy as np
import random
import math
from Polynomial import Polynomial
from numpy.polynomial.polynomial import Polynomial as Polynomial_np
# First we set the parameters
from Logger import debug as d
from Logger import debug_colours
from Polynomial import special_mod


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
        self.logP = 400
        self.P = 2 ** self.logP

        print("========Summary========")
        print("delta: " + str(self.delta))
        for i in reversed(range(self.L + 1)):
            print("q_" + str(i) + ": " + str(self.q_l(i)))
        print("========================")

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
        print('Building basis')
        self.basis = np.array(self.vandermonde(self.root, self.M))
        print('Transposing basis')
        self.basis = self.basis.T

    @staticmethod
    def vandermonde(e: np.complex128, M: int) -> np.array:
        N = M // 2
        matrix = []

        for i in range(N):
            root = e ** (2 * i + 1)
            matrix.append([root ** j for j in range(N)])

        return matrix

    def encode(self, v: np.array) -> Polynomial:
        if v.shape != (self.N // 2,):
            raise Exception('Cannot encode vector of incorrect shape.')
        v = self.mirror(v) * self.params.delta_l(1)
        v = self.project_to_basis(v)
        v = self.random_round(v)
        v = np.matmul(self.basis.T, v)

        return self.sigma_inverse(v)

    def decode(self, p: Polynomial, real=True) -> np.array:
        # d('decode', str(p), debug_colours.GREEN)
        # p =  p * (1.0 / self.params.delta_l(1))
        # d('decode', str(p), debug_colours.GREEN)
        # # d('decode', str(p), debug_colours.GREEN)
        # # p = p.right_shift(self.params.p)
        d('decode', str(p), debug_colours.GREEN)

        v = self.sigma(p)
        v = self.mirror_inv(v)

        if real:
            v = np.real(v)

        for i in range(len(v)):
            v[i] = v[i] / self.params.delta_l(1)

        return v

    def sigma_inverse(self, b: np.array) -> Polynomial:
        A = Encoder.vandermonde(self.root, self.M)
        cs = np.linalg.solve(A, b)[::-1]
        cs = [np.round(np.real(c)) for c in cs]

        return Polynomial.from_coef(cs).round_coefs_nearest()

    def sigma(self, p: Polynomial) -> np.array:
        outputs = []
        p_np = Polynomial_np(np.array(p.p.all_coeffs()[::-1], dtype=float))

        for i in range(self.N):
            root = self.root ** (2 * i + 1)

            output = p.p(root)
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

def polymod(N: int) -> Polynomial:
    coef = [1] + [0 for _ in range(N - 1)] + [1]
    coef = np.array(coef, dtype=object)
    return Polynomial.from_coef(coef)

def debug_poly(poly: Polynomial, Q_l):
    cs = poly.coef
    for i in range(len(cs)):
        print('Power: ' + str(i))
        print('Coefficient: ' + str(int(cs[i])))
    print()

def round_poly(poly: Polynomial, poly_mod: Polynomial, Q_l: int, debug=False):
    if poly.p.get_domain().__str__() != 'ZZ':
        raise Exception('Non-integer polynomial')
    # if max(poly.p.all_coeffs()) > 0 and math.log2(max(poly.p.all_coeffs())) > 64:
    #     print('Warning: coefficient has exceeded 64 bits!')

    poly = poly.rem(poly_mod)
    poly = poly.wrap_coefs(Q_l)
    return poly

class Gen:
    @staticmethod
    def gen_key(n: int, h: int) -> Polynomial:
        coeffs = [0 for _ in range(n)]
        inds = np.random.choice(range(n), size=h, replace=False)
        balls = np.random.choice([-1, 1], h)

        for i in range(h):
            coeffs[inds[i]] = balls[i]

        print(coeffs)

        return Polynomial.from_coef(coeffs)

    @staticmethod
    def gen_error(n: int) -> Polynomial:
        coefs = np.random.normal(scale=1.5, size=n)
        coefs = [special_mod(int(coefs[i]), n) for i in range(n)]
        d('gen_error', str(coefs))
        return Polynomial.from_coef(coefs)

    @staticmethod
    def gen_enc(n: int) -> Polynomial:
        coefs = np.random.choice([-1, 0, 1], size=n, p=[0.25, 0.5, 0.25])
        return Polynomial.from_coef(coefs)

    @staticmethod
    def gen_a(n: int, Q_l: int) -> Polynomial:
        coeffs = [0] * (n)

        for i in range(n):
            coeffs[i] = special_mod(random.randint(0, Q_l), Q_l)

        return Polynomial.from_coef(coeffs)


class KeyGen:
    @staticmethod
    def gen(params: Params):
        N = params.N

        # secret key
        s = Gen.gen_key(N, h = N // 2)
        sk = (1, s)

        # public key
        e = Gen.gen_error(N)
        a = Gen.gen_a(N, params.q_L)

        b = -a * s + e
        b = b.rem(params.poly_mod)

        pk = (b, a)

        # relin key
        a_prime = Gen.gen_a(N, params.q_L * params.P)
        e_prime = Gen.gen_error(N)

        b_prime = -a_prime * s + e_prime + s * s * params.P
        b_prime = b_prime.round_coefs_nearest()
        b_prime = b_prime.rem(params.poly_mod)
        b_prime = b_prime.wrap_coefs(params.q_L * params.P)

        ek = (b_prime, a_prime)

        return sk, pk, ek


class Ciphertext:
    def __init__(self, c0: Polynomial, c1: Polynomial, l: int):
        self.c0 = c0
        self.c1 = c1
        self.l = l

class CipherTemp:
    def __init__(self, c0: Polynomial, c1: Polynomial, c2: Polynomial, l: int):
        self.c0 = c0
        self.c1 = c1
        self.l = l
        self.c2 = c2


class Encryptor:
    def __init__(self, pk, params: Params):
        self.pk = pk
        self.params = params

    def encrypt(self, plaintext: Polynomial):
        d('encrypt', 'poly: ' + str(plaintext.p))
        v = Gen.gen_enc(self.params.N)

        e_0 = Gen.gen_error(self.params.N)
        e_1 = Gen.gen_error(self.params.N)

        c0 = plaintext + e_0 + v * self.pk[0]
        c1 = e_1 + v * self.pk[1]

        c0 = c0.round_coefs_nearest()
        c1 = c1.round_coefs_nearest()

        d('encrypt', 'c0: ' + str(c0.p))
        d('encrypt', 'c1: ' + str(c1.p))

        c0 = c0.rem(self.params.poly_mod)
        c1 = c1.rem(self.params.poly_mod)

        #c0 = c0.wrap_coefs(self.params.q_L)
        #c1 = c1.wrap_coefs(self.params.q_L)

        d('encrypt', 'c0_mod: ' + str(c0.p))
        d('encrypt', 'c1_mod: ' + str(c1.p))

        return Ciphertext(c0, c1, self.params.L)


class Decryptor:
    def __init__(self, sk, params: Params):
        self.sk = sk
        self.params = params

    def decrypt(self, ciphertext: Ciphertext):
        q_l = self.params.q_l(ciphertext.l)

        d('decrypt', 'modulus: ' + str(q_l))

        poly = ciphertext.c0 + ciphertext.c1 * self.sk[1]
        d('decrypt', 'plaintext_: ' + str(poly))

        from sympy import div

        d('decrypt', 'plaintext//base: ' + str(div(poly.p, self.params.poly_mod.p)[0]))

        poly = poly.rem(self.params.poly_mod)
        d('decrypt', 'plaintext rem base: ' + str(poly))

        poly = poly.wrap_coefs(q_l)
        d('decrypt', 'plaintext wrap: ' + str(poly))

        # if poly.check_overflow(q_l):
        #     raise Exception("Cannot decrypt ciphertext!")

        return poly.fix_coefs(q_l)


class Evaluator:
    def __init__(self, ek, params: Params, encoder: Encoder):
        self.ek = ek
        self.params = params
        self.encoder = encoder

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

        d('mul', 'levels: ' + str(cipher1.l) + ', ' + str(cipher2.l))
        d('mul', 'moduli: ' + str(self.params.q_l(cipher1.l)) + ', ' + str(self.params.q_l(cipher2.l)))
        d('mul', 'cipher1 c0: ' + str(cipher1.c0))
        d('mul', 'cipher1 c1: ' + str(cipher1.c1))
        d('mul', 'cipher2 c0: ' + str(cipher2.c0))
        d('mul', 'cipher2 c1: ' + str(cipher2.c1))

        q_l = self.params.q_l(cipher1.l)

        d0 = cipher1.c0 * cipher2.c0
        d1 = cipher1.c0 * cipher2.c1 + cipher1.c1 * cipher2.c0
        d2 = cipher1.c1 * cipher2.c1

        d0 = d0.rem(self.params.poly_mod)
        d1 = d1.rem(self.params.poly_mod)
        d2 = d2.rem(self.params.poly_mod)

        d0 = d0.wrap_coefs(q_l)
        d1 = d1.wrap_coefs(q_l)
        d2 = d2.wrap_coefs(q_l)

        # if True:
        #     f0 = (self.ek[0] * (1.0 / self.params.P) * d2).round_coefs_nearest()
        #     f1 = (self.ek[1] * (1.0 / self.params.P) * d2).round_coefs_nearest()
        #
        #     c0 = d0 + f0
        #     c1 = d1 + f1
        #
        #     c0 = c0.rem(self.params.poly_mod)
        #     c1 = c1.rem(self.params.poly_mod)
        #
        #     c0 = c0.wrap_coefs(q_l)
        #     c1 = c1.wrap_coefs(q_l)
        #
        #     d('!!!', str(c0), debug_colours.FAIL)
        #     d('!!!', str(c1), debug_colours.FAIL)

        f0 = (self.ek[0] * d2).right_shift(self.params.logP)
        f1 = (self.ek[1] * d2).right_shift(self.params.logP)

        c0 = d0 + f0
        c1 = d1 + f1

        c0 = c0.rem(self.params.poly_mod)
        c1 = c1.rem(self.params.poly_mod)

        c0 = c0.wrap_coefs(q_l)
        c1 = c1.wrap_coefs(q_l)

        return Ciphertext(c0, c1, cipher1.l)

    def square(self, cipher: Ciphertext):
        return self.mul(cipher, cipher)

    def rescale(self, cipher: Ciphertext, diff=-1):
        d('rescale', str(diff))
        q_l = self.params.q_l(cipher.l + diff)

        d('rescale', 'upper modulus: ' + str(self.params.q_l(cipher.l)))
        d('rescale', 'lower modulus: ' + str(q_l))

        shift_amount = self.params.p * (-diff)

        d('rescale', 'c0 before: ' + str(cipher.c0))
        d('rescale', 'c1 before: ' + str(cipher.c1))

        cipher.c0 = cipher.c0.right_shift(shift_amount)
        cipher.c1 = cipher.c1.right_shift(shift_amount)

        d('rescale', 'c0 after: ' + str(cipher.c0))
        d('rescale', 'c1 after: ' + str(cipher.c1))

        # cipher.c0 = cipher.c0.wrap_coefs(q_l)
        # cipher.c1 = cipher.c1.wrap_coefs(q_l)

        cipher.l += diff


N = 16
params = Params(N, p=30, p_0=130, L=5)

print('Initialising scheme')
encoder = Encoder(params)
print('Generating keys')

sk, pk, ek = KeyGen.gen(params)

encryptor = Encryptor(pk, params)
decryptor = Decryptor(sk, params)
evaluator = Evaluator(ek, params, encoder)

p_raw = np.array([0.2, 0.5, .7, 0.4, 0.5, 0.4, 0.3, 0.99])
d('main', 'Performing encryption of array: ' + str(p_raw), debug_colours.CYAN)
p = encoder.encode(p_raw)
p_enc = encryptor.encrypt(p)

r_dec = decryptor.decrypt(p_enc)
d('main', str(encoder.decode(r_dec)), debug_colours.CYAN)

q_raw = np.array([1 * i for i in range(N // 2)])
d('main', 'Performing encryption of array: ' + str(q_raw), debug_colours.CYAN)
q = encoder.encode(q_raw)
q_enc = encryptor.encrypt(q)

r_dec = decryptor.decrypt(q_enc)
d('main', str(encoder.decode(r_dec)), debug_colours.CYAN)

for _ in range(5):
    d('main', 'Performing multiplication at level: ' + str(p_enc.l), debug_colours.CYAN)
    d('main', 'Modulus: ' + str(params.q_l(p_enc.l)), debug_colours.CYAN)
    r_enc = evaluator.square(p_enc)
    evaluator.rescale(r_enc)
    r_dec = decryptor.decrypt(r_enc)
    d('main', str(encoder.decode(r_dec)), debug_colours.CYAN)
    d('main', str(r_dec), debug_colours.BOLD)
    p_enc = r_enc


# q_enc = encryptor.encrypt(q)
#
# r_enc = evaluator.square(p_enc)
# evaluator.rescale(r_enc)
#
# r_dec = decryptor.decrypt(r_enc)
# print(encoder.decode(r_dec))
#
# r_enc = evaluator.square(r_enc)
# evaluator.rescale(r_enc)
#
# r_dec = decryptor.decrypt(r_enc)
# print(encoder.decode(r_dec))
#
#
#
#
