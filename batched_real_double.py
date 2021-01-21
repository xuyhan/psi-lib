from batched_real import BatchedReal
from schemes import *

class BatchedRealDouble(BatchedReal):
    def __init__(self, v, scheme: SchemeCKKS):
        self.scheme = scheme
        self.length = scheme.slot_count()

        if isinstance(v, Ciphertext):
            self.ciphertext = v
        else:
            self.ciphertext = scheme.encrypt(v)

        self.num_slots = scheme.slot_count()

    def add(self, batched_real):
        cipher_new = self.scheme.add(self.ciphertext, batched_real.ciphertext)
        return BatchedRealDouble(cipher_new, self.scheme)

    def add_in_place(self, batched_real):
        self.scheme.add_in_place(self.ciphertext, batched_real.ciphertext)

    def add_raw_in_place(self, d):
        self.scheme.add_raw_in_place(self.ciphertext, np.array([d] * self.length))

    def multiply(self, batched_real):
        cipher_new = self.scheme.multiply(self.ciphertext, batched_real.ciphertext)
        return BatchedRealDouble(cipher_new, self.scheme)

    def multiply_raw(self, d):
        cipher_new = self.scheme.multiply_raw(self.ciphertext, np.array([d] * self.length))
        return BatchedRealDouble(cipher_new, self.scheme)

    def multiply_in_place(self, batched_real):
        self.scheme.multiply_in_place(self.ciphertext, batched_real.ciphertext)

    def square(self):
        cipher_new = self.scheme.square(self.ciphertext)
        return BatchedRealDouble(cipher_new, self.scheme)

    def square_in_place(self):
        self.scheme.square_in_place(self.ciphertext)

    def sum(self):
        total = self.scheme.encrypt(np.array([0]))

        for i in range(self.length):
            self.scheme.rotate_in_place(total, 1)
            self.scheme.add_in_place(total, self.ciphertext)

        self.scheme.rotate_in_place(self.ciphertext, self.scheme.slot_count() // 2 - 2 * self.length)

        for i in range(self.length):
            self.scheme.rotate_in_place(self.ciphertext, 1)
            self.scheme.add_in_place(total, self.ciphertext)

        return BatchedRealDouble(total, self.scheme)

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
