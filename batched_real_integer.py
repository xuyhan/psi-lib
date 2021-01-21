from schemes import *
from batched_real import BatchedReal

class BatchedInteger:
    def __init__(self, v, scheme: CRTScheme):
        self.scheme = scheme
        self.length = 5000

        if isinstance(v, CiphertextCRT):
            self.ciphertext = v
        else:
            self.ciphertext = scheme.encrypt(v)

        self.num_slots = scheme.schemes[0].slot_count()

    def add(self, batched_int):
        cipher_new = self.scheme.add(self.ciphertext, batched_int.ciphertext)
        return BatchedInteger(cipher_new, self.scheme)

    def add_in_place(self, batched_int):
        self.scheme.add_in_place(self.ciphertext, batched_int.ciphertext)

    def add_int_in_place(self, n: int):
        self.scheme.add_int_in_place(self.ciphertext, np.array([n] * self.length))

    def multiply(self, batched_int):
        cipher_new = self.scheme.multiply(self.ciphertext, batched_int.ciphertext)
        return BatchedInteger(cipher_new, self.scheme)

    def multiply_int(self, n: int):
        cipher_new = self.scheme.multiply_int(self.ciphertext, np.array([n] * self.length))
        return BatchedInteger(cipher_new, self.scheme)

    def multiply_in_place(self, batched_int):
        self.scheme.multiply_in_place(self.ciphertext, batched_int.ciphertext)

    def multiply_int_in_place(self, n: int):
        self.scheme.multiply_int_in_place(self.ciphertext, np.array([n] * self.length))

    def square(self):
        cipher_new = self.scheme.square(self.ciphertext)
        return BatchedInteger(cipher_new, self.scheme)

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

        return BatchedInteger(total, self.scheme)

    def relinearise(self):
        self.scheme.relinearise(self.ciphertext)

    def debug(self, length):
        result = []

        for i in range(self.scheme.n):
            scheme = self.scheme.schemes[i]
            plaintext = scheme.decrypt(self.ciphertext.get(i))
            raw = scheme._batch_decode(plaintext)

            decoded = []

            for j in range(length):
                decoded.append(raw[j])

            result.append(decoded)

        result_crt = []

        for j in range(length):
            moduli = [result[i][j] for i in range(self.scheme.n)]

            bases = [self.scheme.schemes[i].plaintext_modulus for i in range(self.scheme.n)]
            result_crt.append(crt(p=moduli, q=bases))

        t = self.scheme.get_plaintext_modulus()

        for i in range(len(result_crt)):
            result_crt[i] = -(t - result_crt[i]) if result_crt[i] > (t // 2) else result_crt[i]

        return result_crt


class BatchedRealInteger(BatchedReal):
    def __init__(self, v, scale, scheme: CRTScheme):
        self.scale = scale
        self.scheme = scheme

        arg = v
        if isinstance(arg, np.ndarray):
            arg = (arg * scale).astype(int)
        self.batched_int = BatchedInteger(arg, scheme)

    def add(self, batched_real):
        if self.scale != batched_real.scale:
            raise Exception('scale mismatch')

        return BatchedRealInteger(self.batched_int.add(batched_real.batched_int).ciphertext, self.scale, self.batched_int.scheme)

    def add_raw_in_place(self, raw: float):
        raw_int = int(raw * self.scale)
        self.batched_int.add_int_in_place(n=raw_int)

    def add_in_place(self, batched_real):
        if self.scale != batched_real.scale:
            raise Exception('Scale mismatch')
        self.batched_int.add_in_place(batched_real.batched_int)

    def multiply(self, batched_real):
        return BatchedRealInteger(self.batched_int.multiply(batched_real.batched_int).ciphertext,
                                  self.scale * batched_real.scale,
                                  self.batched_int.scheme)

    def multiply_raw(self, real: float):
        real = int(real * self.scheme.default_weight_scale)
        return BatchedRealInteger(self.batched_int.multiply_int(real).ciphertext,
                           self.scale * self.scheme.default_weight_scale,
                           self.batched_int.scheme)

    def multiply_in_place(self, batched_real):
        self.batched_int.multiply_in_place(batched_real.batched_int)
        self.scale *= batched_real.scale

    def square(self):
        return BatchedRealInteger(self.batched_int.square().ciphertext,
                           self.scale ** 2,
                           self.batched_int.scheme)

    def square_in_place(self):
        self.batched_int.square_in_place()
        self.scale = self.scale ** 2

    def sum(self):
        return BatchedRealInteger(self.batched_int.sum().ciphertext,
                           self.scale,
                           self.batched_int.scheme)

    def debug(self, length: int):
        result = self.batched_int.debug(length)
        for i in range(len(result)):
            result[i] /= self.scale
        return result

    def relinearise(self):
        self.batched_int.relinearise()

    def noise(self):
        self.scheme.evaluate_ciphertext(self.batched_int.ciphertext)