from typing import List

from seal import *
from seal_helper import *
import numpy as np
from crt import crt

class Scheme:
    def summary(self):
        pass

    def _batch_encode(self, v: np.ndarray, debug=False) -> Plaintext:
        pass

    def _batch_decode(self, plain: Plaintext) -> uIntVector:
        pass

    def encrypt(self, v: np.ndarray) -> Ciphertext:
        pass

    def decrypt(self, cipher: Ciphertext) -> Plaintext:
        pass

    def zero(self):
        pass

    def add(self, cipher_a: Ciphertext, cipher_b: Ciphertext) -> Ciphertext:
        pass

    def add_int(self, cipher_a: Ciphertext, v: np.ndarray) -> Ciphertext:
        pass

    def add_in_place(self, cipher_a: Ciphertext, cipher_b: Ciphertext):
        pass

    def add_int_in_place(self, cipher_a: Ciphertext, v: np.ndarray):
        pass

    def multiply(self, cipher_a: Ciphertext, cipher_b: Ciphertext) -> Ciphertext:
        pass

    def multiply_int(self, cipher_a: Ciphertext, v: np.ndarray) -> Ciphertext:
        pass

    def multiply_in_place(self, cipher_a: Ciphertext, cipher_b: Ciphertext):
        pass

    def multiply_int_in_place(self, cipher_a: Ciphertext, v: np.ndarray):
        pass

    def square(self, cipher_a: Ciphertext) -> Ciphertext:
        pass

    def square_in_place(self, cipher_a: Ciphertext):
        pass

    def slot_count(self):
        pass

    def rotate_in_place(self, cipher: Ciphertext, n: int):
        pass

    def relinearise(self, cipher: Ciphertext):
        pass

    def evaluate_ciphertext(self, ciphertext):
        pass


class CiphertextCRT:
    def __init__(self, ciphers: List[Ciphertext]):
        self.ciphers = ciphers

    def get(self, i):
        return self.ciphers[i]

class CRTScheme:
    def summary(self):
        pass

    def get_plaintext_modulus(self) -> int:
        pass

    def encrypt(self, v: np.ndarray) -> CiphertextCRT:
        pass

    def encrypt_special(self, v: np.ndarray) -> CiphertextCRT:
        pass

    def zero(self) -> CiphertextCRT:
        pass

    def add(self, cipher_a: CiphertextCRT, cipher_b: CiphertextCRT) -> CiphertextCRT:
        pass

    def add_int(self, cipher_a: CiphertextCRT, v: np.ndarray) -> CiphertextCRT:
        pass

    def add_in_place(self, cipher_a: CiphertextCRT, cipher_b: CiphertextCRT):
        pass

    def add_int_in_place(self, cipher_a: CiphertextCRT, v: np.ndarray):
        pass

    def multiply(self, cipher_a: CiphertextCRT, cipher_b: CiphertextCRT) -> CiphertextCRT:
        pass

    def multiply_int(self, cipher_a: CiphertextCRT, v: np.ndarray) -> CiphertextCRT:
        pass

    def multiply_in_place(self, cipher_a: CiphertextCRT, cipher_b: CiphertextCRT):
        pass

    def multiply_int_in_place(self, cipher_a: CiphertextCRT, v: np.ndarray):
        pass

    def square(self, cipher_a: CiphertextCRT) -> CiphertextCRT:
        pass

    def square_in_place(self, cipher_a: CiphertextCRT):
        pass

    def slot_count(self):
        pass

    def rotate_in_place(self, cipher: CiphertextCRT, n: int):
        pass

    def relinearise(self, cipher: CiphertextCRT):
        pass

    def evaluate_ciphertext(self, ciphertext: CiphertextCRT):
        pass

class SchemeBFV(Scheme):
    def __init__(self, poly_modulus_degree, plain_modulus_bits):
        params = EncryptionParameters(scheme_type.BFV)

        params.set_poly_modulus_degree(poly_modulus_degree)
        params.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
        params.set_plain_modulus(PlainModulus.Batching(poly_modulus_degree, plain_modulus_bits))

        context = SEALContext.Create(params)
        context_data = context.key_context_data()
        plain_modulus = context_data.parms().plain_modulus().value()
        qualifiers = context.first_context_data().qualifiers()

        keygen = KeyGenerator(context)

        self.plaintext_modulus = plain_modulus
        self.encryptor = Encryptor(context, keygen.public_key())
        self.decryptor = Decryptor(context, keygen.secret_key())
        self.encoder = BatchEncoder(context)
        self.evaluator = Evaluator(context)
        self.relin_keys = keygen.relin_keys()
        self.gal_keys = keygen.galois_keys()

    def summary(self):
        print('Plaintext modulus: %s' % self.plaintext_modulus)
        print('Slot count: %s' % self.slot_count())

    def _batch_encode(self, v: np.ndarray, debug=False) -> Plaintext:
        matrix = [0] * self.encoder.slot_count()
        matrix[:len(v)] = v % self.plaintext_modulus
        matrix = uIntVector(matrix)

        plaintext = Plaintext()
        self.encoder.encode(matrix, plaintext)

        return plaintext

    def _batch_decode(self, plain: Plaintext) -> uIntVector:
        raw = uIntVector()
        self.encoder.decode(plain, raw)
        return raw

    def encrypt(self, v: np.ndarray) -> Ciphertext:
        ciphertext = Ciphertext()
        self.encryptor.encrypt(self._batch_encode(v), ciphertext)
        return ciphertext

    def decrypt(self, cipher: Ciphertext) -> Plaintext:
        plaintext = Plaintext()
        self.decryptor.decrypt(cipher, plaintext)
        return plaintext

    def zero(self):
        return self.encrypt(np.array([0]))

    def add(self, cipher_a: Ciphertext, cipher_b: Ciphertext) -> Ciphertext:
        cipher_new = Ciphertext()
        self.evaluator.add(cipher_a, cipher_b, cipher_new)
        return cipher_new

    def add_int(self, cipher_a: Ciphertext, v: np.ndarray) -> Ciphertext:
        cipher_new = Ciphertext()
        self.evaluator.add(cipher_a, self._batch_encode(v), cipher_new)
        return cipher_new

    def add_in_place(self, cipher_a: Ciphertext, cipher_b: Ciphertext):
        self.evaluator.add_inplace(cipher_a, cipher_b)

    def add_int_in_place(self, cipher_a: Ciphertext, v: np.ndarray):
        self.evaluator.add_plain_inplace(cipher_a, self._batch_encode(v))

    def multiply(self, cipher_a: Ciphertext, cipher_b: Ciphertext) -> Ciphertext:
        cipher_new = Ciphertext()
        self.evaluator.multiply(cipher_a, cipher_b, cipher_new)
        return cipher_new

    def multiply_int(self, cipher_a: Ciphertext, v: np.ndarray) -> Ciphertext:
        if np.sum(v) == 0:
            return self.zero()

        cipher_new = Ciphertext()
        self.evaluator.multiply_plain(cipher_a, self._batch_encode(v, debug=True), cipher_new)
        return cipher_new

    def multiply_in_place(self, cipher_a: Ciphertext, cipher_b: Ciphertext):
        self.evaluator.multiply_inplace(cipher_a, cipher_b)

    def multiply_int_in_place(self, cipher_a: Ciphertext, v: np.ndarray):
        self.evaluator.multiply_plain_inplace(cipher_a, self._batch_encode(v))

    def square(self, cipher_a: Ciphertext) -> Ciphertext:
        cipher_new = Ciphertext()
        self.evaluator.square(cipher_a, cipher_new)
        return cipher_new

    def square_in_place(self, cipher_a: Ciphertext):
        self.evaluator.square_inplace(cipher_a)

    def slot_count(self):
        return self.encoder.slot_count()

    def rotate_in_place(self, cipher: Ciphertext, n: int):
        self.evaluator.rotate_rows_inplace(cipher, n, self.gal_keys)

    def relinearise(self, cipher: Ciphertext):
        self.evaluator.relinearize_inplace(cipher, self.relin_keys)

    def evaluate_ciphertext(self, ciphertext):
        budget = self.decryptor.invariant_noise_budget(ciphertext)
        print("size: %s, budget: %s" %(
            str(ciphertext.size()),
            str(budget) + " bits"
            )
        )

        return budget


class SchemeBFV_CRT(CRTScheme):
    def __init__(self, poly_modulus_degree, qs):
        self.bfv_schemes = []
        for i in range(len(qs)):
            scheme = SchemeBFV(poly_modulus_degree, qs[i])
            self.bfv_schemes.append(scheme)
        self.n = len(self.bfv_schemes)

    def get_plaintext_modulus(self) -> int:
        n = self.bfv_schemes[0].plaintext_modulus
        for i in range(1, self.n):
            n *= self.bfv_schemes[i].plaintext_modulus
        return n

    def summary(self):
        print('CRT Scheme uses %s ciphertexts' % self.n)
        print('Plaintext modulus: %s ' % self.get_plaintext_modulus())

    def encrypt(self, v: np.ndarray) -> CiphertextCRT:
        ciphers_new = [self.bfv_schemes[i].encrypt(v) for i in range(self.n)]
        return CiphertextCRT(ciphers_new)

    def zero(self) -> CiphertextCRT:
        ciphers_new = [self.bfv_schemes[i].zero() for i in range(self.n)]
        return CiphertextCRT(ciphers_new)

    def add(self, cipher_a: CiphertextCRT, cipher_b: CiphertextCRT) -> CiphertextCRT:
        ciphers_new = [self.bfv_schemes[i].add(cipher_a.get(i), cipher_b.get(i)) for i in range(self.n)]
        return CiphertextCRT(ciphers_new)

    def add_int(self, cipher_a: CiphertextCRT, v: np.ndarray) -> CiphertextCRT:
        ciphers_new = [self.bfv_schemes[i].add_int(cipher_a.get(i), v) for i in range(self.n)]
        return CiphertextCRT(ciphers_new)

    def add_in_place(self, cipher_a: CiphertextCRT, cipher_b: CiphertextCRT):
        for i in range(self.n):
            self.bfv_schemes[i].add_in_place(cipher_a.get(i), cipher_b.get(i))

    def add_int_in_place(self, cipher_a: CiphertextCRT, v: np.ndarray):
        for i in range(self.n):
            self.bfv_schemes[i].add_int_in_place(cipher_a.get(i), v)

    def multiply(self, cipher_a: CiphertextCRT, cipher_b: CiphertextCRT) -> CiphertextCRT:
        ciphers_new = [self.bfv_schemes[i].multiply(cipher_a.get(i), cipher_b.get(i)) for i in range(self.n)]
        return CiphertextCRT(ciphers_new)

    def multiply_int(self, cipher_a: CiphertextCRT, v: np.ndarray) -> CiphertextCRT:
        ciphers_new = [self.bfv_schemes[i].multiply_int(cipher_a.get(i), v) for i in range(self.n)]
        return CiphertextCRT(ciphers_new)

    def multiply_in_place(self, cipher_a: CiphertextCRT, cipher_b: CiphertextCRT):
        for i in range(self.n):
            self.bfv_schemes[i].multiply_in_place(cipher_a.get(i), cipher_b.get(i))

    def multiply_int_in_place(self, cipher_a: CiphertextCRT, v: np.ndarray):
        for i in range(self.n):
            self.bfv_schemes[i].multiply_int_in_place(cipher_a.get(i), v)

    def square(self, cipher_a: CiphertextCRT) -> CiphertextCRT:
        ciphers_new = [self.bfv_schemes[i].square(cipher_a.get(i)) for i in range(self.n)]
        return CiphertextCRT(ciphers_new)

    def square_in_place(self, cipher_a: CiphertextCRT):
        for i in range(self.n):
            self.bfv_schemes[i].square_in_place(cipher_a.get(i))

    def slot_count(self):
        return self.bfv_schemes[0].slot_count()

    def rotate_in_place(self, cipher: CiphertextCRT, n: int):
        for i in range(self.n):
            self.bfv_schemes[i].rotate_in_place(cipher.get(i), n)

    def relinearise(self, cipher: CiphertextCRT):
        for i in range(self.n):
            self.bfv_schemes[i].relinearise(cipher.get(i))

    def evaluate_ciphertext(self, ciphertext: CiphertextCRT):
        for i in range(self.n):
            self.bfv_schemes[i].evaluate_ciphertext(ciphertext.get(i))


class BatchedInteger:
    def __init__(self, v, scheme: SchemeBFV_CRT):
        self.scheme = scheme
        self.length = 5000

        if isinstance(v, CiphertextCRT):
            self.ciphertext = v
        else:
            self.ciphertext = scheme.encrypt(v)

        self.num_slots = scheme.bfv_schemes[0].slot_count()

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
            scheme = self.scheme.bfv_schemes[i]
            plaintext = scheme.decrypt(self.ciphertext.get(i))
            raw = scheme._batch_decode(plaintext)

            decoded = []

            for j in range(length):
                decoded.append(raw[j])

            result.append(decoded)

        result_crt = []

        for j in range(length):
            moduli = [result[i][j] for i in range(self.scheme.n)]
            #print(moduli)

            bases = [self.scheme.bfv_schemes[i].plaintext_modulus for i in range(self.scheme.n)]
            result_crt.append(crt(p=moduli, q=bases))

        #print(result_crt)

        t = self.scheme.get_plaintext_modulus()

        for i in range(len(result_crt)):
            result_crt[i] = -(t - result_crt[i]) if result_crt[i] > (t // 2) else result_crt[i]
            #print(result_crt[i])

        return result_crt

class BatchedReal(BatchedInteger):
    def __init__(self, v, scale: float, scheme: SchemeBFV_CRT):
        self.scale = scale
        arg = v
        if isinstance(arg, np.ndarray):
            arg = (arg * scale).astype(int)
        super().__init__(arg, scheme)

    @staticmethod
    def zero(scale: float, scheme: SchemeBFV_CRT):
        return BatchedReal(np.array([0]), scale, scheme)

    def add(self, batched_real):
        if self.scale != batched_real.scale:
            raise Exception('scale mismatch')

        return BatchedReal(super().add(batched_real).ciphertext,
                           self.scale,
                           self.scheme)

    def add_raw_in_place(self, raw: float):
        raw_int = int(raw * self.scale)
        super().add_int_in_place(n=raw_int)

    def add_in_place(self, batched_real):
        if self.scale != batched_real.scale:
            raise Exception('Scale mismatch')
        super().add_in_place(batched_real)

    def multiply(self, batched_real):
        return BatchedReal(super().multiply(batched_real).ciphertext,
                           self.scale * batched_real.scale,
                           self.scheme)

    def multiply_plain(self, real: np.ndarray):
        real = (real * self.scale).astype(int)
        return BatchedReal(super().multiply_int(real).ciphertext,
                           self.scale ** 2,
                           self.scheme)

    def multiply_in_place(self, batched_real):
        super().multiply_in_place(batched_real)
        self.scale *= batched_real.scale

    def square(self):
        return BatchedReal(super().square().ciphertext,
                           self.scale ** 2,
                           self.scheme)

    def square_in_place(self):
        super().square_in_place()
        self.scale = self.scale ** 2

    def sum(self):
        return BatchedReal(super().sum().ciphertext,
                           self.scale,
                           self.scheme)

    def debug(self, length: int):
        result = super().debug(length)
        for i in range(len(result)):
            result[i] /= self.scale
        return result


class BatchedRealVec:
    def __init__(self, data: List[BatchedReal], scale: float, scheme: SchemeBFV_CRT):
        self.data = data
        self.scale = scale
        self.scheme = scheme

    def len(self):
        return len(self.data)

    @staticmethod
    def zero(length: int, scale: float, scheme: SchemeBFV_CRT):
        return BatchedRealVec([BatchedReal.zero(scale, scheme) for _ in range(length)],
                              scale,
                              scheme)

    def relinearise(self):
        for batched_real in self.data:
            batched_real.relinearise()

    def set_element(self, i, batched_real: BatchedReal):
        self.data[i] = batched_real

    def element(self, i: int) -> BatchedReal:
        return self.data[i]

    def subregion(self, i, j):
        return BatchedRealVec(self.data[i: j], self.scale, self.scheme)

    def append(self, batched_real: BatchedReal):
        self.data.append(batched_real)

    def prepend(self, batched_real: BatchedReal):
        self.data = [batched_real] + self.data

    def extend(self, batched_real_vec):
        self.data.extend(batched_real_vec.data)

    def add(self, batched_real_vec):
        self.__vec_enc_check(batched_real_vec)

        new_data = []
        for i in range(len(self.data)):
            new_data.append(self.data[i].add(batched_real_vec.data[i]))

        return BatchedRealVec(np.array(new_data), self.scale, self.scheme)

    def add_raw_in_place(self, vec_raw: np.ndarray):
        self.__vec_raw_check(vec_raw)

        for i in range(len(self.data)):
            self.data[i].add_raw_in_place(vec_raw[i])

    def add_in_place(self, batched_real_vec):
        self.__vec_enc_check(batched_real_vec)

        for i in range(len(self.data)):
            self.data[i].add_in_place(batched_real_vec.data[i])

    def multiply_element_wise(self, batched_real_vec):
        self.__vec_enc_check(batched_real_vec)

        new_data = [self.data[i].multiply(batched_real_vec.data[i]) for i in range(len(self.data))]

        return BatchedRealVec(new_data, self.scale * batched_real_vec.scale, self.scheme)

    def multiply_element_wise_plain(self, vec_raw):
        self.__vec_raw_check(vec_raw)
        new_data = [self.data[i].multiply_plain(vec_raw[i]) for i in range(len(self.data))]
        return BatchedRealVec(new_data, self.scale ** 2, self.scheme)

    def square(self):
        new_data = np.array([self.data[i].square() for i in range(len(self.data))])
        return BatchedRealVec(new_data, self.scale ** 2, self.scheme)

    def square_in_place(self):
        self.scale = self.scale ** 2
        for batched_real in self.data:
            batched_real.square_in_place()

    def get_sum(self):
        if self.len() == 1:
            return self.data[0]

        total = self.data[0].add(self.data[1])
        for i in range(2, self.len()):
            total.add_in_place(self.data[i])

        return total

    def dot(self, batched_real_vec):
        self.__vec_enc_check(batched_real_vec)

        return self.multiply_element_wise(batched_real_vec).get_sum()

    def dot_plain(self, vec_raw):
        self.__vec_raw_check(vec_raw)

        return self.multiply_element_wise_plain(vec_raw).get_sum()

    def __vec_enc_check(self, batched_real_vec):
        if self.scale != batched_real_vec.scale:
            raise Exception("Scale mismatch: %s != %s" % (self.scale, batched_real_vec.scale))
        if self.len() != batched_real_vec.len():
            raise Exception("Length mismatch: %s != %s" % (self.data.len(), batched_real_vec.data.len()))

    def __vec_raw_check(self, vec_raw):
        if self.len() != vec_raw.shape[0]:
            raise Exception("Length mismatch: %s != %s" % (len(self.data), len(vec_raw)))


class BatchedRealMat:
    def __init__(self, data: List[BatchedRealVec], scale: float, scheme: SchemeBFV_CRT):
        self.data = data
        self.scale = scale
        self.scheme = scheme

    @staticmethod
    def __init_np__(data: np.ndarray, scale: float, scheme: SchemeBFV_CRT):
        data_updated = []
        for row in data:
            data_updated.append(BatchedRealVec(row, scale, scheme))
        return BatchedRealMat(data_updated, scale, scheme)

    def relinearise(self):
        for row in self.data:
            row.relinearise()

    @staticmethod
    def zeros(num_rows, num_cols, scale, scheme):
        data = []
        for _ in range(num_rows):
            data.append(BatchedRealVec.zero(num_cols, scale, scheme))
        return BatchedRealMat(data, scale, scheme)

    def shape(self):
        return len(self.data), self.data[0].len()

    def set_element(self, i: int, j: int, batched_real: BatchedReal):
        self.data[i].set_element(j, batched_real)

    def element(self, i: int, j: int) -> BatchedReal:
        return self.data[i].element(j)

    def row(self, i: int) -> BatchedRealVec:
        return self.data[i]

    def col(self, i: int) -> BatchedRealVec:
        vec_data = np.array([self.element(j, i) for j in range(self.shape()[0])])
        return BatchedRealVec(vec_data, self.scale, self.scheme)

    def subregion(self, r_start, r_end, c_start, c_end):
        data_subregion = []
        for vec in self.data[r_start: r_end]:
            data_subregion.append(vec.subregion(c_start, c_end))
        return BatchedRealMat(data_subregion, self.scale, self.scheme)

    def add(self, real_mat):
        if self.shape() != real_mat.shape():
            raise Exception("Error: cannot add matrices with different shapes.")

        new_data = []
        for i in range(self.shape()[0]):
            new_data.append(self.data[i].add(real_mat.data[i]))

        return BatchedRealMat(new_data, self.scale, self.scheme)

    def add_in_place(self, real_mat):
        if self.shape() != real_mat.shape():
            raise Exception("Error: cannot add matrices with different shapes.")

        for i in range(self.shape()[0]):
            self.data[i].add_in_place(real_mat.data[i])

    def add_raw_in_place(self, raw_mat: np.ndarray, debug=False):
        if debug:
            print('Adding biases')

        if self.shape() != raw_mat.shape:
            raise Exception('Shape mismatch in matrix add: %s != %s' % (self.shape(), raw_mat.shape))

        for i in range(self.shape()[0]):
            self.row(i).add_raw_in_place(raw_mat[i])

    def flatten(self):
        result = []
        for i in range(self.shape()[0]):
            result.append(self.row(i).data)
        result = np.array(result)
        result = result.reshape(1, -1)[0]
        vec = BatchedRealVec(list(result), self.scale, self.scheme)
        return BatchedRealMat([vec], self.scale, self.scheme)

    def square_in_place(self):
        self.scale = self.scale ** 2
        for vec in self.data:
            vec.square_in_place()

    def mult(self, real_mat):
        if self.shape()[1] != real_mat.shape()[0]:
            raise Exception("Error cannot multiply %s matrix with %s matrix" %(self.shape, real_mat.shape))
        data_new = []
        scale_new = None
        for i in range(self.shape()[0]):
            row_new = [None for _ in range(real_mat.shape()[1])]
            for j in range(real_mat.shape()[1]):
                row_a = self.row(i)
                col_b = real_mat.col(j)
                row_new[j] = row_a.dot(col_b)
                if scale_new is not None and scale_new != row_new[j].scale:
                    raise Exception("Scale error in matrix multiply")
                scale_new = row_new[j].scale
            data_new.append(BatchedRealVec(row_new, scale_new, self.scheme))
        return BatchedRealMat(data_new, scale_new, self.scheme)

    def mult_plain(self, raw_mat: np.ndarray, debug=False):
        if self.shape()[1] != raw_mat.shape[0]:
            raise Exception("Error cannot multiply %s matrix with %s matrix" %(self.shape(), raw_mat.shape))
        data_new = []
        shape_a = self.shape()
        shape_b = raw_mat.shape

        if debug:
            print('Starting matrix multiplication...')

            total_steps = shape_a[0] * shape_b[1]
            progress_step = total_steps // 10

        for i in range(shape_a[0]):
            row_new = [None for _ in range(shape_b[1])]
            for j in range(shape_b[1]):
                row_a = self.row(i)
                col_b = raw_mat[:, j]
                row_new[j] = row_a.dot_plain(col_b)

                progress = i * shape_a[0] + j
                if debug and progress % progress_step == 0:
                    print('Progress: %.2f' % (progress / total_steps))

            data_new.append(BatchedRealVec(row_new, self.scale ** 2, self.scheme))

        return BatchedRealMat(data_new, self.scale ** 2, self.scheme)

    def left_mult_plain(self, raw_mat: np.ndarray):
        if self.shape()[0] != raw_mat.shape[1]:
            raise Exception("Error cannot multiply %s matrix with %s matrix" %(raw_mat.shape, self.shape()))
        data_new = []
        shape_a = raw_mat.shape
        shape_b = self.shape()

        for i in range(shape_a[0]):
            row_new = [None for _ in range(shape_b[1])]
            for j in range(shape_b[1]):
                row_a = raw_mat[i]
                col_b = self.col(j)
                row_new[j] = col_b.dot_plain(row_a)

            data_new.append(BatchedRealVec(row_new, self.scale ** 2, self.scheme))
        return BatchedRealMat(data_new, self.scale ** 2, self.scheme)

    def pad_in_place(self):
        old_h, old_w = self.shape()
        for vec in self.data:
            vec.prepend(BatchedReal.zero(scale=self.scale, scheme=self.scheme))
            vec.append(BatchedReal.zero(scale=self.scale, scheme=self.scheme))
        self.data = [BatchedRealVec.zero(old_w + 2, self.scale, scheme=self.scheme)] + self.data
        self.data.append(BatchedRealVec.zero(old_w + 2, self.scale, scheme=self.scheme))

    def transpose(self):
        data = []
        for j in range(self.shape()[1]):
            row_data = []
            for i in range(self.shape()[0]):
                row_data.append(self.row(i).element(j))
            data.append(BatchedRealVec(data=row_data, scale=self.scale, scheme=self.scheme))
        return BatchedRealMat(data, self.scale, scheme=self.scheme)

    def column_concat(self, batched_real_mat):
        if self.shape()[0] != batched_real_mat.shape()[0]:
            raise Exception('Row mismatch: %s != %s' % (self.shape()[0], batched_real_mat.shape()[0]))

        for i in range(self.shape()[0]):
            self.row(i).extend(batched_real_mat.row(i))

    def noise(self):
        self.scheme.evaluate_ciphertext(self.element(0, 0).ciphertext)


class Creator:
    def __init__(self, scheme, scale=4096):
        self.scheme = scheme
        self.scale = scale

    def encrypt(self, mat: np.ndarray) -> BatchedRealMat:
        if len(mat.shape) != 3:
            raise Exception('Expected input to have three dimensions. Received: %s' % len(mat.shape))

        mat_data = []

        for i in range(mat.shape[1]):
            row_data = []
            for j in range(mat.shape[2]):
                pixel_data = mat[:, i, j].reshape(1, -1)[0]
                batched_real = BatchedReal(np.array(pixel_data), self.scale, self.scheme)
                row_data.append(batched_real)
            mat_data.append(BatchedRealVec(row_data, self.scale, self.scheme))

        return BatchedRealMat(mat_data, self.scale, self.scheme)

    def debug(self, batched_real_mat: BatchedRealMat, length: int) -> np.ndarray:
        num_rows, num_cols = batched_real_mat.shape()

        result = []

        for i in range(num_rows):
            row = []
            for j in range(num_cols):
                row.append(batched_real_mat.element(i, j).debug(length))
            result.append(row)

        return np.array(result)


def get_scheme(poly_mods, plaintext_mods):
    scheme = SchemeBFV_CRT(
        poly_mods, plaintext_mods
    )

    return scheme

if __name__ == '__main__':
    creator = Creator(get_scheme())

    a = np.random.uniform(-1, 1, (5, 3))
    b = np.random.uniform(-1, 1, (3, 6))
    c = np.matmul(a, b)

    a_enc = creator.encrypt(a)
    c_enc = a_enc.mult_plain(b)

    c_dec = creator.debug(c_enc)
    print(c)
    print(c_dec)



'''
class BatchedVec(BatchedReal):
    def __init__(self, v, scale):
        super().__init__(v, scale, len(v))

    def dot(self, batched_vec):
        v = self.multiply(batched_vec)
        return v.sum()

    # def dot_plain(self, plain_vec):
    #     v = self.multiply_plain(plain_vec)
    #     return v.sum()

class BatchedMat(BatchedReal):
    class Perm():
        def __init__(self, func, diag_inds_func, n):
            self.func = func
            self.diag_inds_func = diag_inds_func
            self.n = n

        def diag_inds(self, k):
            inds = set()
            for ind in self.diag_inds_func(k):
                inds.add(ind % self.n)
            return list(inds)

        def __call__(self, r, c, j, k):
            return self.func(r, c, j, k)

    def __init__(self, mat, scale, shape):
        self.shape = shape

        if not isinstance(mat, Ciphertext):
            vec = np.zeros(self.shape[0] * self.shape[1])

            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    vec[self.index_map(i, j)] = mat[i][j]

            super().__init__(vec, scale, self.shape[0] * self.shape[1])
        else:
            super().__init__(mat, scale, self.shape[0] * self.shape[1])

        def pa(r, c, j, k):
            return int(r * self.shape[1] + ((r + c) % self.shape[1]) == j)

        def pb(r, c, j, k):
            return int(((r + c) % self.shape[0]) * self.shape[1] + c == j)

        def px(r, c, j, k):
            return int(r * self.shape[1] + (c + k) % self.shape[1] == j)

        def py(r, c, j, k):
            return int(((r + k) % self.shape[0]) * self.shape[1] + c == j)

        def p_stack(r, c, j, k):
            i = r * self.shape[1] + c
            rs = i // k
            cs = i % k

            if rs >= self.shape[0] or cs >= self.shape[1]:
                return 0

            return int(rs * self.shape[1] + cs == j)

        def p_squeeze(r, c, j, k):
            i = r * self.shape[1] + c
            rs = i // k[1]
            cs = i % k[1]
            return int(rs * self.shape[1] + cs == j)

        self.dim = self.shape[0]
        self.size = self.shape[0] * self.shape[1]

        n = self.num_slots // 2
        self.perm_a = BatchedMat.Perm(pa, lambda k: [i for i in range(-self.dim + 1, self.dim)], n)
        self.perm_b = BatchedMat.Perm(pb, lambda k: [i * self.dim for i in range(-self.dim + 1, self.dim)], n)
        self.perm_x = BatchedMat.Perm(px, lambda k: [k, k - self.dim], n)
        self.perm_y = BatchedMat.Perm(py, lambda k: [self.dim * k, self.dim * k - self.size], n)

        self.perm_stack = BatchedMat.Perm(p_stack, lambda k: [-(k - self.shape[1]) * i for i in range(self.shape[0])], n)
        self.perm_squeeze = BatchedMat.Perm(p_squeeze, lambda k: [(self.shape[1] - k[1]) * i for i in range(k[0])], n)

    def empty_ciphertext(self):
        result = Ciphertext()
        encryptor.encrypt(BatchedInteger.encode(np.array([0])), result)
        return result

    def index_map(self, i, j):
        return i * self.shape[1] + j

    def reverse_map(self, i):
        return (i // self.shape[1], i % self.shape[1])

    def get_diag_positions(self, n):
        inds = []
        side_len = self.num_slots // 2
        pos = (0, n % side_len)

        for i in range(side_len):
            inds.append(pos)
            pos = (pos[0] + 1, (pos[1] + 1) % side_len)

        return inds

    def perm_diags(self, perm, k=None):
        diags = []
        diag_inds = perm.diag_inds(k)

        #diags = [[1] * 4096] * len(diag_inds)

        for i in diag_inds:
            inds = self.get_diag_positions(i)
            diag = []

            for x, y in inds:
                # if x >= self.shape[0] * self.shape[1]:
                #     diag.append(int(x == y))
                # else:
                #     #diag.append(perm(x // self.shape[1], x % self.shape[1], y % (self.dim ** 2), k))
                diag.append(perm(x // self.shape[1], x % self.shape[1], y, k))

            diags.append(diag)

        diags = np.array(diags)
        return diags, diag_inds

    def lin_trans(self, perm, k, ciphertext):
        import time

        result = self.empty_ciphertext()
        diags, diag_inds = self.perm_diags(perm, k)
        num_diags = len(diag_inds)
        debug = False

        for i in range(num_diags):
            diag = list(diags[i]) + ([0] * (self.num_slots // 2))
            diag_ind = diag_inds[i]

            start_time = time.time()
            ciphertext_rot = Ciphertext()
            evaluator.rotate_rows(ciphertext, diag_ind, gal_keys, ciphertext_rot)
            if debug: print("--- rot: %s seconds ---" % (time.time() - start_time))

            plain_matrix = Plaintext()
            #diag_cipher = Ciphertext()
            batch_encoder.encode(uIntVector(diag), plain_matrix)
            #encryptor.encrypt(plain_matrix, diag_cipher)

            product = Ciphertext()

            start_time = time.time()
            #evaluator.multiply(ciphertext_rot, diag_cipher, product)
            evaluator.multiply_plain(ciphertext_rot, plain_matrix, product)

            if debug: print("--- mult: %s seconds ---" % (time.time() - start_time))

            start_time = time.time()
            evaluator.add_inplace(result, product)
            if debug: print("--- add: %s seconds ---" % (time.time() - start_time))

        return result

    def add(self, batched_mat):
        if self.shape != batched_mat.shape:
            raise Exception('Incompatible matrix operands for matrix addition: %s + %s' %(self.shape, batched_mat.shape))
        cipher_new = super().add(batched_mat).ciphertext
        return BatchedMat(cipher_new, self.scale, self.shape)

    def mult(self, batched_mat):
        if self.shape[1] != batched_mat.shape[0]:
            raise Exception('Incompatible matrix operands for matrix multiplication: %s x %s' %(self.shape, batched_mat.shape))

        rectangle_mode = self.shape[0] != self.shape[1] or batched_mat.shape[0] != batched_mat.shape[1]

        matA = self
        matB = batched_mat

        evaluate_ciphertext(self.ciphertext)

        if rectangle_mode:
            new_dim = max(max(self.shape[0], self.shape[1]), max(batched_mat.shape[0], batched_mat.shape[1]))

            matA_cipher = self.lin_trans(self.perm_stack, new_dim, self.ciphertext)
            matB_cipher = batched_mat.lin_trans(batched_mat.perm_stack, new_dim, batched_mat.ciphertext)

            evaluate_ciphertext(matA_cipher)

            evaluator.relinearize_inplace(matA_cipher, relin_keys)
            evaluator.relinearize_inplace(matB_cipher, relin_keys)

            matA = BatchedMat(matA_cipher, self.scale, (new_dim, new_dim))
            matB = BatchedMat(matB_cipher, batched_mat.scale, (new_dim, new_dim))

        p = matA.lin_trans(matA.perm_a, None, matA.ciphertext)
        q = matB.lin_trans(matB.perm_b, None, matB.ciphertext)

        evaluate_ciphertext(p)

        evaluator.relinearize_inplace(p, relin_keys)
        evaluator.relinearize_inplace(q, relin_keys)

        cipher_new = matA.empty_ciphertext()

        p_temp = p
        q_temp = q

        for i in range(matA.dim):
            if i > 0:
                p_temp = matA.lin_trans(matA.perm_x, i, p)
                q_temp = matA.lin_trans(matA.perm_y, i, q)
                evaluator.relinearize_inplace(p_temp, relin_keys)
                evaluator.relinearize_inplace(q_temp, relin_keys)

            temp = Ciphertext()
            evaluator.multiply(p_temp, q_temp, temp)
            evaluator.relinearize_inplace(temp, relin_keys)
            evaluator.add_inplace(cipher_new, temp)

        result_shape = (self.shape[0], batched_mat.shape[1])

        evaluate_ciphertext(cipher_new)

        if rectangle_mode:
            cipher_new = matA.lin_trans(matA.perm_squeeze, result_shape, cipher_new)

        evaluate_ciphertext(cipher_new)

        return BatchedMat(cipher_new, matA.scale * batched_mat.scale, result_shape)

    def debug(self, verbose=False):
        plaintext = Plaintext()
        decryptor.decrypt(self.ciphertext, plaintext)
        raw = BatchedInteger.decode(plaintext)
        raw = [value / self.scale for value in raw]
        mat = np.zeros(self.shape)
        for i in range(self.length):
            x, y = self.reverse_map(i)
            mat[x][y] = raw[i]
        if verbose:
            print('Dims: [%s, %s]' % self.shape)
            print(mat)
        return mat



class EncryptedInteger:
    def __init__(self, ciphertext):
        self.ciphertext = ciphertext

    def __create__(n):
        ciphertext = Ciphertext()
        encryptor.encrypt(EncryptedInteger.__encode(n), ciphertext)
        return EncryptedInteger(ciphertext)

    def add(self, int_enc):
        ciphertext_result = Ciphertext()
        evaluator.add(self.ciphertext, int_enc.ciphertext, ciphertext_result)
        return EncryptedInteger(ciphertext_result)

    def add_in_place(self, int_enc):
        evaluator.add_inplace(self.ciphertext, int_enc.ciphertext)

    def multiply(self, int_enc):
        ciphertext_result = Ciphertext()
        evaluator.multiply(self.ciphertext, int_enc.ciphertext, ciphertext_result)
        return EncryptedInteger(ciphertext_result)

    def multiply_in_place(self, int_enc):
        evaluator.multiply_inplace(self.ciphertext, int_enc.ciphertext)

    def add_plain(self, n):
        ciphertext_result = Ciphertext()
        evaluator.add_plain(self.ciphertext, EncryptedInteger.__encode(n), ciphertext_result)
        return EncryptedInteger(ciphertext_result)

    def multiply_plain(self, n):
        ciphertext_result = Ciphertext()
        evaluator.multiply_plain(self.ciphertext, EncryptedInteger.__encode(n), ciphertext_result)
        return EncryptedInteger(ciphertext_result)

    def __encode(n):
        #return int_encoder.encode(n)
        n = n % plain_modulus
        return Plaintext(hex(n)[2:])

    def decode(int_plain):
        #return int_encoder.decode_int32(int_plain)
        result = int(int_plain.to_string(), 16)
        if result > (plain_modulus // 2):
            return -(plain_modulus - result)
        return result

class EncryptedReal:
    def __init__(self, int_enc, scale):
        self.int_enc = int_enc
        self.scale = scale

    def __create__(real_raw, scale):
        int_enc = EncryptedInteger.__create__(int(real_raw * scale))
        return EncryptedReal(int_enc, scale)

    def add(self, real_enc):
        int_enc = self.int_enc.add(real_enc.int_enc)
        return EncryptedReal(int_enc, self.scale)

    def add_in_place(self, real_enc):
        self.int_enc.add_in_place(real_enc.int_enc)

    def multiply(self, real_enc):
        int_enc = self.int_enc.multiply(real_enc.int_enc)
        return EncryptedReal(int_enc, self.scale * real_enc.scale)

    def multiply_in_place(self, real_enc):
        self.int_enc.multiply_in_place(real_enc.int_enc)
        self.scale *= real_enc.scale

    def add_plain(self, real_raw):
        int_enc = self.int_enc.add_plain(int(real_raw * self.scale))
        return EncryptedReal(int_enc, self.scale) 

    def multiply_plain(self, real_raw):
        int_enc = self.int_enc.multiply_plain(int(real_raw * self.scale))
        return EncryptedReal(int_enc, self.scale ** 2)

class EncryptedRealVector:
    def __init__(self, data, scale):
        self.data = data
        self.scale = scale

    def __create__(vec_raw, scale):
        data = [EncryptedReal.__create__(real_raw, scale) for real_raw in vec_raw]

        return EncryptedRealVector(data, scale)

    def element(self, i):
        return self.data[i]

    def len(self):
        return len(self.data)

    def add(self, vec_enc):
        self.__vec_enc_check(vec_enc)

        new_data = []
        for i in range(self.len()):
            new_data.append(self.data[i].add(vec_enc.data[i]))

        return EncryptedRealVector(new_data, self.scale)

    def add_in_place(self, vec_enc):
        self.__vec_enc_check(vec_enc)

        for i in range(self.len()):
            self.data[i].add_in_place(vec_enc.data[i])

    def multiply_element_wise(self, vec_enc):
        self.__vec_enc_check(vec_enc)

        new_data = [self.data[i].multiply(vec_enc.data[i]) for i in range(self.len())]
        return EncryptedRealVector(new_data, self.scale * vec_enc.scale)

    def multiply_element_wise_plain(self, vec_raw):
        self.__vec_raw_check(vec_raw)

        new_data = [self.data[i].multiply_plain(vec_raw[i]) for i in range(self.len())]

        return EncryptedRealVector(new_data, self.scale)
            
    def get_sum(self):
        if self.len() == 1:
            return self.data[0]

        total = self.data[0].add(self.data[1])
        for i in range(2, self.len()):
            total.add_in_place(self.data[i])

        return total

    def dot(self, vec_enc):
        self.__vec_enc_check(vec_enc)

        return self.multiply_element_wise(vec_enc).get_sum()

    def dot_plain(self, vec_raw):
        self.__vec_raw_check(vec_raw)
        
        return self.multiply_element_wise_plain(vec_raw).get_sum()

    def __vec_enc_check(self, vec_enc):
        if self.scale != vec_enc.scale:
            raise Exception("Scale mismatch: %s != %s" %(self.scale, vec_enc.scale))
        if self.len() != vec_enc.len():
            raise Exception("Length mismatch: %s != %s" %(self.len(), vec_enc.len()))

    def __vec_raw_check(self, vec_raw):
        if self.len() != len(vec_raw):
            raise Exception("Length mismatch: %s != %s" %(self.len(), len(vec_raw)))

class EncryptedRealMatrix():
    def __init__(self, data, scale):
        self.data = data
        self.scale = scale
        self.shape = [len(data), data[0].len()]

    def __create__(plain_mat, scale):
         data = [EncryptedRealVector.__create__(row, scale) for row in plain_mat]
         return EncryptedRealMatrix(data, scale)

    def element(self, i, j):
        return self.data[i].element(j)

    def row(self, i):
        return self.data[i]

    def col(self, i):
        vec_data = [self.element(j, i) for j in range(self.shape[0])]
        return EncryptedRealVector(vec_data, self.scale)

    def add(self, real_mat):
        if self.shape != real_mat.shape:
            raise Exception("Error: cannot add matrices with different shapes.")

        new_data = []
        for i in range(self.shape[0]):
            new_data.append(self.data[i].add(real_mat.data[i]))

        return EncryptedRealMatrix(new_data, self.scale)

    def add_in_place(self, real_mat):
        if self.shape != real_mat.shape:
            raise Exception("Error: cannot add matrices with different shapes.")

        for i in range(self.shape[0]):
            self.data[i].add_in_place(real_mat.data[i])

    def mult(self, real_mat):
        if self.shape[1] != real_mat.shape[0]:
            raise Exception("Error cannot multiply %s matrix with %s matrix" %(self.shape, real_mat.shape))

        data_new = []
        scale_new = None

        for i in range(self.shape[0]):
            row_new = [None for _ in range(real_mat.shape[1])]
            for j in range(real_mat.shape[1]):
                row_a = self.row(i)
                col_b = real_mat.col(j)
                row_new[j] = row_a.dot(col_b)

                if scale_new is not None and scale_new != row_new[j].scale:
                    raise Exception("Scale error in matrix multiply")

                scale_new = row_new[j].scale

            data_new.append(EncryptedRealVector(row_new, scale_new))

        return EncryptedRealMatrix(data_new, scale_new)



def encrypt_real(real_raw):
    return EncryptedReal.__create__(real_raw, scale=scaling_factor)

def encrypt_real_vec(vec_raw):
    return EncryptedRealVector.__create__(vec_raw, scale=scaling_factor)

def encrypt_real_mat(mat_raw):
    return EncryptedRealMatrix.__create__(mat_raw, scale=scaling_factor)

def decrypt(ciphertext):
    plaintext = Plaintext()
    decryptor.decrypt(ciphertext, plaintext)
    return EncryptedInteger.decode(plaintext)

def decrypt_int(int_enc):
    return decrypt(int_enc.ciphertext)

def decrypt_real(real_enc):
    return decrypt_int(real_enc.int_enc) / real_enc.scale

def decrypt_real_vec(vec_enc):
    return [decrypt_real(real_enc) for real_enc in vec_enc.data]

def decrypt_real_mat(mat_enc):
    return [decrypt_real_vec(vec_enc) for vec_enc in mat_enc.data]

def reveal_ciphertext(ciphertext):
    plaintext = Plaintext()
    decryptor.decrypt(ciphertext, plaintext)
    print(plaintext.to_string())'''
