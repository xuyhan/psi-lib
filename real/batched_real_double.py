from real.batched_real import HEReal
from crypto.schemes import *


class HERealDouble(HEReal):
    def __init__(self, v, scheme: SchemeCKKS):
        self.scheme = scheme
        self.length = scheme.slot_count()

        if isinstance(v, Ciphertext):
            self.ciphertext = v
        else:
            self.ciphertext = scheme.encrypt(v)

        self.num_slots = scheme.slot_count()

    def add(self, he_real):
        cipher_new = self.scheme.add(self.ciphertext, he_real.ciphertext)
        return HERealDouble(cipher_new, self.scheme)

    def add_raw(self, d: float):
        cipher_new = self.scheme.add_raw(self.ciphertext, np.array([d] * self.length))
        return HERealDouble(cipher_new, self.scheme)

    def add_in_place(self, he_real):
        self.scheme.add_in_place(self.ciphertext, he_real.ciphertext)

    def sub_in_place(self, he_real):
        self.scheme.sub_in_place(self.ciphertext, he_real.ciphertext)

    def add_raw_in_place(self, d):
        if isinstance(d, List):
            vals = np.array(d + [0 for _ in range(self.length - len(d))])
        else:
            vals = np.array([d] * self.length)

        self.scheme.add_raw_in_place(self.ciphertext, vals)

    def multiply(self, he_real):
        cipher_new = self.scheme.multiply(self.ciphertext, he_real.ciphertext)
        return HERealDouble(cipher_new, self.scheme)

    def multiply_raw(self, d):
        if isinstance(d, List):
            vals = np.array(d + [0 for _ in range(self.length - len(d))])
        else:
            vals = np.array([d] * self.length)
        cipher_new = self.scheme.multiply_raw(self.ciphertext, vals)
        return HERealDouble(cipher_new, self.scheme)

    def multiply_in_place(self, he_real):
        self.scheme.multiply_in_place(self.ciphertext, he_real.ciphertext)

    def multiply_raw_in_place(self, d):
        if isinstance(d, List):
            vals = np.array(d + [0 for _ in range(self.length - len(d))])
        else:
            vals = np.array([d] * self.length)

        self.scheme.multiply_raw_in_place(self.ciphertext, vals)

    def square(self):
        cipher_new = self.scheme.square(self.ciphertext)
        return HERealDouble(cipher_new, self.scheme)

    def square_in_place(self, flag=False):
        self.scheme.square_in_place(self.ciphertext, flag)

    def sum(self, n):
        total = self.scheme.encrypt(np.array([0]))

        for i in range(n):
            self.scheme.add_in_place(total, self.ciphertext)
            self.scheme.rotate_in_place(self.ciphertext, 1)

        self.scheme.rotate_in_place(self.ciphertext, -n)

        return HERealDouble(total, self.scheme)

    def relinearise(self):
        self.scheme.relinearise(self.ciphertext)

    def debug(self, length):
        plaintext = self.scheme.decrypt(self.ciphertext)
        raw = self.scheme._batch_decode(plaintext)

        decoded = []

        for j in range(length):
            decoded.append(raw[j])

        return decoded

    def noise(self):
        self.scheme.evaluate_ciphertext(self.ciphertext)

    def slot_count(self):
        return self.scheme.slot_count()

    def zeros(self):
        v = [0 for _ in range(self.slot_count())]
        return HERealDouble(v, self.scheme)

    def rot(self, n: int):
        return HERealDouble(self.scheme.rotate(self.ciphertext, -n), self.scheme)

    def perm(self, elt: int):
        return HERealDouble(self.scheme.perm(self.ciphertext, elt), self.scheme)
