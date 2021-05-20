from typing import List

from seal import *
import numpy as np

class CiphertextCRT:
    def __init__(self, ciphers: List[Ciphertext]):
        self.ciphers = ciphers

    def get(self, i):
        return self.ciphers[i]

class SchemeBase:
    def evaluate_ciphertext(self, ciphertext):
        pass

class DecryptorBase:
    def decrypt(self, cipher: Ciphertext) -> Plaintext:
        pass

class Scheme(SchemeBase):
    def summary(self):
        pass

    def _batch_encode(self, v: np.ndarray, debug=False) -> Plaintext:
        pass

    def _batch_decode(self, plain: Plaintext) -> uIntVector:
        pass

    def encrypt(self, v: np.ndarray) -> Ciphertext:
        pass

    def zero(self):
        pass

    def add(self, cipher_a: Ciphertext, cipher_b: Ciphertext) -> Ciphertext:
        pass

    def add_raw(self, cipher_a: Ciphertext, v: np.ndarray) -> Ciphertext:
        pass

    def add_in_place(self, cipher_a: Ciphertext, cipher_b: Ciphertext):
        pass

    def sub_in_place(self, cipher_a: Ciphertext, cipher_b: Ciphertext):
        pass

    def add_raw_in_place(self, cipher_a: Ciphertext, v: np.ndarray):
        pass

    def multiply(self, cipher_a: Ciphertext, cipher_b: Ciphertext) -> Ciphertext:
        pass

    def multiply_raw(self, cipher_a: Ciphertext, v: np.ndarray) -> Ciphertext:
        pass

    def multiply_in_place(self, cipher_a: Ciphertext, cipher_b: Ciphertext):
        pass

    def multiply_raw_in_place(self, cipher_a: Ciphertext, v: np.ndarray):
        pass

    def square(self, cipher_a: Ciphertext) -> Ciphertext:
        pass

    def square_in_place(self, cipher_a: Ciphertext):
        pass

    def slot_count(self):
        pass

    def rotate_in_place(self, cipher: Ciphertext, n: int):
        pass

    def rotate(self, cipher: Ciphertext, n: int):
        pass

    def rotate_rows(self, cipher1: Ciphertext, n: int) -> Ciphertext:
        pass

    def relinearise(self, cipher: Ciphertext):
        pass


class SchemeCKKS(Scheme):
    def __init__(self, encryptor, evaluator, encoder, relin_keys, gal_keys, default_scale):
        self.encryptor = encryptor
        self.evaluator = evaluator
        self.encoder = encoder
        self.relin_keys = relin_keys
        self.gal_keys = gal_keys
        self.default_scale = default_scale

    def mod_switch(self, ciphertext_a: Ciphertext, to_switch):
        self.evaluator.mod_switch_to_inplace(to_switch, ciphertext_a.parms_id())

    def summary(self):
        print('Slot count: %s' % self.slot_count())

    def _batch_encode(self, v: np.ndarray, debug=False) -> Plaintext:
        matrix = [0] * self.encoder.slot_count()
        matrix[:len(v)] = v
        matrix = DoubleVector(matrix)

        plaintext = Plaintext()
        self.encoder.encode(matrix, self.default_scale, plaintext)

        return plaintext

    def _batch_decode(self, plain: Plaintext) -> DoubleVector:
        raw = DoubleVector()
        self.encoder.decode(plain, raw)
        return raw

    def encrypt(self, v: np.ndarray) -> Ciphertext:
        ciphertext = Ciphertext()
        self.encryptor.encrypt(self._batch_encode(v), ciphertext)
        return ciphertext

    def zero(self):
        return self.encrypt(np.array([0]))

    def add(self, cipher_a: Ciphertext, cipher_b: Ciphertext) -> Ciphertext:
        self.mod_switch(cipher_a, cipher_b)

        cipher_new = Ciphertext()
        self.evaluator.add(cipher_a, cipher_b, cipher_new)

        return cipher_new

    def add_in_place(self, cipher_a: Ciphertext, cipher_b: Ciphertext):
        self.mod_switch(cipher_a, cipher_b)
        self.evaluator.add_inplace(cipher_a, cipher_b)

    def sub_in_place(self, cipher_a: Ciphertext, cipher_b: Ciphertext):
        self.mod_switch(cipher_a, cipher_b)
        self.evaluator.sub_inplace(cipher_a, cipher_b)

    def add_raw(self, cipher_a: Ciphertext, v: np.ndarray) -> Ciphertext:
        plaintext = self._batch_encode(v)
        self.mod_switch(cipher_a, plaintext)

        cipher_new = Ciphertext()
        self.evaluator.add_plain(cipher_a, plaintext, cipher_new)

        return cipher_new

    def add_raw_in_place(self, cipher_a: Ciphertext, v: np.ndarray):
        plaintext = self._batch_encode(v)

        self.mod_switch(cipher_a, plaintext)

        self.evaluator.add_plain_inplace(cipher_a, plaintext)

    def multiply(self, cipher_a: Ciphertext, cipher_b: Ciphertext) -> Ciphertext:
        self.mod_switch(cipher_a, cipher_b)

        cipher_new = Ciphertext()
        self.evaluator.multiply(cipher_a, cipher_b, cipher_new)

        self.evaluator.rescale_to_next_inplace(cipher_new)
        cipher_new.scale(self.default_scale)

        return cipher_new

    def multiply_in_place(self, cipher_a: Ciphertext, cipher_b: Ciphertext):
        self.mod_switch(cipher_a, cipher_b)

        self.evaluator.multiply_inplace(cipher_a, cipher_b)

        self.evaluator.rescale_to_next_inplace(cipher_a)
        cipher_a.scale(self.default_scale)

    def multiply_raw(self, cipher_a: Ciphertext, v: np.ndarray) -> Ciphertext:
        if np.sum(v) == 0:
            return self.zero()

        plaintext = self._batch_encode(v)
        self.mod_switch(cipher_a, plaintext)

        cipher_new = Ciphertext()

        self.evaluator.multiply_plain(cipher_a, plaintext, cipher_new)

        self.evaluator.rescale_to_next_inplace(cipher_new)
        cipher_new.scale(self.default_scale)

        return cipher_new

    def multiply_raw_in_place(self, cipher_a: Ciphertext, v: np.ndarray):
        plaintext = self._batch_encode(v)
        self.mod_switch(cipher_a, plaintext)

        self.evaluator.multiply_plain_inplace(cipher_a, plaintext)

        self.evaluator.rescale_to_next_inplace(cipher_a)
        cipher_a.scale(self.default_scale)

    def square(self, cipher_a: Ciphertext) -> Ciphertext:
        cipher_new = Ciphertext()
        self.evaluator.square(cipher_a, cipher_new)
        self.evaluator.relinearize_inplace(cipher_new, self.relin_keys)
        self.evaluator.rescale_to_next_inplace(cipher_new)
        cipher_new.scale(self.default_scale)

        return cipher_new

    def square_in_place(self, cipher_a: Ciphertext, flag):
        self.evaluator.square_inplace(cipher_a)

        self.evaluator.relinearize_inplace(cipher_a, self.relin_keys)
        self.evaluator.rescale_to_next_inplace(cipher_a)
        cipher_a.scale(self.default_scale)

    def slot_count(self):
        return self.encoder.slot_count()

    def rotate_in_place(self, cipher: Ciphertext, n: int):
        self.evaluator.rotate_vector_inplace(cipher, n, self.gal_keys)

    def rotate(self, cipher: Ciphertext, n: int):
        cipher_out = Ciphertext()
        self.evaluator.rotate_vector(cipher, n, self.gal_keys, cipher_out)
        return cipher_out

    def rotate_rows(self, cipher: Ciphertext, n: int):
        ciphertext_rot = Ciphertext()
        self.evaluator.rotate_rows(cipher, n, self.gal_keys, ciphertext_rot)
        return ciphertext_rot

    def relinearise(self, cipher: Ciphertext):
        self.evaluator.relinearize_inplace(cipher, self.relin_keys)

    def perm(self, cipher: Ciphertext, elt: int):
        ciphertext_perm = Ciphertext()
        self.evaluator.apply_galois(cipher, elt, self.gal_keys_s, ciphertext_perm)
        return ciphertext_perm

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
        self.decryptor = Decryptor(context, keygen.secret_key())
        self.encryptor = Encryptor(context, keygen.public_key())
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

    def add_raw(self, cipher_a: Ciphertext, v: np.ndarray) -> Ciphertext:
        cipher_new = Ciphertext()
        self.evaluator.add(cipher_a, self._batch_encode(v), cipher_new)
        return cipher_new

    def add_in_place(self, cipher_a: Ciphertext, cipher_b: Ciphertext):
        self.evaluator.add_inplace(cipher_a, cipher_b)

    def add_raw_in_place(self, cipher_a: Ciphertext, v: np.ndarray):
        self.evaluator.add_plain_inplace(cipher_a, self._batch_encode(v))

    def multiply(self, cipher_a: Ciphertext, cipher_b: Ciphertext) -> Ciphertext:
        cipher_new = Ciphertext()
        self.evaluator.multiply(cipher_a, cipher_b, cipher_new)
        return cipher_new

    def multiply_raw(self, cipher_a: Ciphertext, v: np.ndarray) -> Ciphertext:
        if np.sum(v) == 0:
            return self.zero()

        cipher_new = Ciphertext()
        self.evaluator.multiply_plain(cipher_a, self._batch_encode(v, debug=True), cipher_new)
        return cipher_new

    def multiply_in_place(self, cipher_a: Ciphertext, cipher_b: Ciphertext):
        self.evaluator.multiply_inplace(cipher_a, cipher_b)

    def multiply_raw_in_place(self, cipher_a: Ciphertext, v: np.ndarray):
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

class CRTScheme(SchemeBase):
    def __init__(self, poly_modulus_degree, qs, default_scale, default_weight_scale):
        self.schemes = []
        self.n = None
        self.default_scale = None
        self.default_weight_scale = None

        for i in range(len(qs)):
            self.schemes.append(SchemeBFV(poly_modulus_degree, qs[i]))

        self.n = len(self.schemes)

        self.default_scale = default_scale
        self.default_weight_scale = default_weight_scale

    def get_plaintext_modulus(self) -> int:
        n = self.schemes[0].plaintext_modulus
        for i in range(1, self.n):
            n *= self.schemes[i].plaintext_modulus
        return n

    def summary(self):
        print('CRT Scheme uses %s ciphertexts' % self.n)
        print('Plaintext modulus: %s ' % self.get_plaintext_modulus())

    def encrypt(self, v: np.ndarray) -> CiphertextCRT:
        ciphers_new = [self.schemes[i].encrypt(v) for i in range(self.n)]
        return CiphertextCRT(ciphers_new)

    def zero(self) -> CiphertextCRT:
        ciphers_new = [self.schemes[i].zero() for i in range(self.n)]
        return CiphertextCRT(ciphers_new)

    def add(self, cipher_a: CiphertextCRT, cipher_b: CiphertextCRT) -> CiphertextCRT:
        ciphers_new = [self.schemes[i].add(cipher_a.get(i), cipher_b.get(i)) for i in range(self.n)]
        return CiphertextCRT(ciphers_new)

    def add_int(self, cipher_a: CiphertextCRT, v: np.ndarray) -> CiphertextCRT:
        ciphers_new = [self.schemes[i].add_raw(cipher_a.get(i), v) for i in range(self.n)]
        return CiphertextCRT(ciphers_new)

    def add_in_place(self, cipher_a: CiphertextCRT, cipher_b: CiphertextCRT):
        for i in range(self.n):
            self.schemes[i].add_in_place(cipher_a.get(i), cipher_b.get(i))

    def add_int_in_place(self, cipher_a: CiphertextCRT, v: np.ndarray):
        for i in range(self.n):
            self.schemes[i].add_raw_in_place(cipher_a.get(i), v)

    def multiply(self, cipher_a: CiphertextCRT, cipher_b: CiphertextCRT) -> CiphertextCRT:
        ciphers_new = [self.schemes[i].multiply(cipher_a.get(i), cipher_b.get(i)) for i in range(self.n)]
        return CiphertextCRT(ciphers_new)

    def multiply_int(self, cipher_a: CiphertextCRT, v: np.ndarray) -> CiphertextCRT:
        ciphers_new = [self.schemes[i].multiply_raw(cipher_a.get(i), v) for i in range(self.n)]
        return CiphertextCRT(ciphers_new)

    def multiply_in_place(self, cipher_a: CiphertextCRT, cipher_b: CiphertextCRT):
        for i in range(self.n):
            self.schemes[i].multiply_in_place(cipher_a.get(i), cipher_b.get(i))

    def multiply_int_in_place(self, cipher_a: CiphertextCRT, v: np.ndarray):
        for i in range(self.n):
            self.schemes[i].multiply_raw_in_place(cipher_a.get(i), v)

    def square(self, cipher_a: CiphertextCRT) -> CiphertextCRT:
        ciphers_new = [self.schemes[i].square(cipher_a.get(i)) for i in range(self.n)]
        return CiphertextCRT(ciphers_new)

    def square_in_place(self, cipher_a: CiphertextCRT):
        for i in range(self.n):
            self.schemes[i].square_in_place(cipher_a.get(i))

    def slot_count(self):
        return self.schemes[0].slot_count()

    def rotate_in_place(self, cipher: CiphertextCRT, n: int):
        for i in range(self.n):
            self.schemes[i].rotate_in_place(cipher.get(i), n)

    def relinearise(self, cipher: CiphertextCRT):
        for i in range(self.n):
            self.schemes[i].relinearise(cipher.get(i))

    def evaluate_ciphertext(self, ciphertext: CiphertextCRT):
        for i in range(self.n):
            self.schemes[i].evaluate_ciphertext(ciphertext.get(i))

class DecryptorCKKS(DecryptorBase):
    def __init__(self, decryptor):
        self.decryptor = decryptor

    def decrypt(self, cipher: Ciphertext) -> Plaintext:
        plaintext = Plaintext()
        self.decryptor.decrypt(cipher, plaintext)
        return plaintext

def init_scheme_ckks(poly_mod_degree, primes, scale_factor):
    params = EncryptionParameters(scheme_type.CKKS)
    params.set_poly_modulus_degree(poly_mod_degree)
    params.set_coeff_modulus(CoeffModulus.Create(poly_mod_degree, primes))

    context = SEALContext.Create(params)

    keygen = KeyGenerator(context)

    public_key = keygen.public_key()
    secret_key = keygen.secret_key()

    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)
    encoder = CKKSEncoder(context)

    relin_keys = keygen.relin_keys()
    gal_keys = keygen.galois_keys()

    default_scale = 2.0 ** scale_factor

    return SchemeCKKS(
        encryptor, evaluator, encoder, relin_keys, gal_keys, default_scale
    ), DecryptorCKKS(decryptor)








