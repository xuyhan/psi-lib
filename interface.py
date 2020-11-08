from seal import *

# set parameters for BFV

params = EncryptionParameters(scheme_type.BFV)

poly_modulus_degree = 4096
params.set_poly_modulus_degree(poly_modulus_degree)
params.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
params.set_plain_modulus(1048576)

context = SEALContext.Create(params)

keygen = KeyGenerator(context)
public_key = keygen.public_key()
secret_key = keygen.secret_key()

encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)
decryptor = Decryptor(context, secret_key)
encoder = IntegerEncoder(context)

scaling_factor = 1024

# Level 0: support for integers

def encrypt(n):
    ciphertext = Ciphertext()
    encryptor.encrypt(encoder.encode(n), ciphertext)
    return ciphertext


def decrypt(ciphertext):
    plaintext = Plaintext()
    decryptor.decrypt(ciphertext, plaintext)
    return encoder.decode_int32(plaintext)


def add(ciphertext_a, ciphertext_b):
    ciphertext_c = Ciphertext()
    evaluator.add(ciphertext_a, ciphertext_b, ciphertext_c)
    return ciphertext_c


def add_in_place(ciphertext_a, ciphertext_b):
    evaluator.add_inplace(ciphertext_a, ciphertext_b)


def add_plain(ciphertext, n):
    ciphertext_out = Ciphertext()
    evaluator.add_plain(ciphertext, encoder.encode(n), ciphertext_out)
    return ciphertext_out


def multiply(ciphertext_a, ciphertext_b):
    ciphertext_c = Ciphertext()
    evaluator.multiply(ciphertext_a, ciphertext_b, ciphertext_c)
    return ciphertext_c


def multiply_in_place(ciphertext_a, ciphertext_b):
    evaluator.multiply_inplace(ciphertext_a, ciphertext_b)


def multiply_plain(ciphertext, n):
    ciphertext_out = Ciphertext()
    evaluator.multiply_plain(ciphertext, encoder.encode(n), ciphertext_out)
    return ciphertext_out

# Level 1: support for real vectors

class EncryptedReal:
    def __init__(self, ciphertext, scale):
        self.ciphertext = ciphertext
        self.scale = scale

    def __create__(plain_real, scale):
        ciphertext = encrypt(int(plain_real * scale))
        return EncryptedReal(ciphertext, scale)

    def add(self, encrypt_real):
        ciphertext = add(self.ciphertext, encrypt_real.ciphertext)
        return EncryptedReal(ciphertext, self.scale)

    def add_in_place(self, encrypted_real):
        add_in_place(self.ciphertext, encrypted_real.ciphertext)

    def add_plain(self, plain_real):
        ciphertext = add_plain(self.ciphertext, int(plain_real * self.scale))
        return EncryptedReal(ciphertext, self.scale) 

    def multiply(self, encrypted_real):
        ciphertext = multiply(self.ciphertext, encrypted_real.ciphertext)
        return EncryptedReal(ciphertext, self.scale * encrypted_real.scale)

    def multiply_in_place(self, encrypted_real):
        multiply_in_place(self.ciphertext, encrypted_real.ciphertext)
        self.scale *= encrypt_real.scale

    def multiply_plain(self, plain_real):
        ciphertext = multiply_plain(self.ciphertext, int(plain_real * self.scale))
        return EncryptedReal(ciphertext, self.scale ** 2)


class EncryptedRealVector:
    def __init__(self, data, scale):
        self.data = data
        self.scale = scale

    def __create__(plain_vec, scale):
        data = [EncryptedReal.__create__(value_plain, scale) for value_plain in plain_vec]
        return EncryptedRealVector(data, scale)

    def element(self, i):
        return self.data[i]

    def len(self):
        return len(self.data)

    def add(self, real_vec):
        if self.scale != real_vec.scale:
            raise Exception("Scale mismatch: %s != %s" %(self.scale, real_vec.scale))
        if self.len() != real_vec.len():
            raise Exception("Length mismatch: %s != %s" %(self.len(), real_vec.len()))

        new_data = []
        for i in range(self.len()):
            new_data.append(self.data[i].add(real_vec.data[i]))

        return EncryptedRealVector(new_data, self.scale)

    def add_in_place(self, real_vec):
        if self.scale != real_vec.scale:
            raise Exception("Scale mismatch: %s != %s" %(self.scale, real_vec.scale))
        if self.len() != real_vec.len():
            raise Exception("Length mismatch: %s != %s" %(self.len(), real_vec.len()))

        for i in range(self.len()):
            self.data[i].add_in_place(real_vec.data[i])

    def multiply_element_wise(self, real_vec):
        if self.scale != real_vec.scale:
            raise Exception("Scale mismatch: %s != %s" %(self.scale, real_vec.scale))
        if self.len() != real_vec.len():
            raise Exception("Length mismatch: %s != %s" %(self.len(), real_vec.len()))

        new_data = [self.data[i].multiply(real_vec.data[i]) for i in range(self.len())]
        return EncryptedRealVector(new_data, self.scale * real_vec.scale)

    def multiply_element_wise_plain(self, real_vec):
        if self.len() != len(real_vec):
            raise Exception("Length mismatch: %s != %s" %(self.len(), real_vec.len()))

        new_data = [self.data[i].multiply_plain(real_vec[i]) for i in range(self.len())]
        return EncryptedRealVector(new_data, self.scale)
            
    def get_sum(self):
        if self.len() == 1:
            return self.data[0]

        total = self.data[0].add(self.data[1])
        for i in range(2, self.len()):
            total.add_in_place(self.data[i])

        return total

    def dot(self, real_vec):
        if self.len() != real_vec.len():
            raise Exception("Length mismatch: %s != %s" %(self.len(), len(real_vec)))

        return self.multiply_element_wise(real_vec).get_sum()

    def dot_plain(self, real_vec):
        if self.len() != len(real_vec):
            raise Exception("Length mismatch: %s != %s" %(self.len(), len(real_vec)))
        
        return self.multiply_element_wise_plain(real_vec).get_sum()


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


def decrypt_real(encrypted_real):
    return decrypt(encrypted_real.ciphertext) / encrypted_real.scale


def decrypt_real_vec(encrypted_real_vec):
    return [decrypt_real(encrypted_real) for encrypted_real in encrypted_real_vec.data]


def decrypt_real_mat(encrypted_real_mat):
    return [decrypt_real_vec(encrypt_real_vec) for encrypt_real_vec in encrypted_real_mat.data]


def encrypt_real(value_plain):
    return EncryptedReal.__create__(value_plain, scale=scaling_factor)


def encrypt_real_vec(vec_plain):
    return EncryptedRealVector.__create__(vec_plain, scale=scaling_factor)


def encrypt_real_mat(mat_plain):
    return EncryptedRealMatrix.__create__(mat_plain, scale=scaling_factor)


def evaluate_ciphertext(ciphertext):
    plaintext = Plaintext()
    decryptor.decrypt(ciphertext, plaintext)

    print("size: %s, budget: %s, value: %s" %(
        str(ciphertext.size()),
        str(decryptor.invariant_noise_budget(ciphertext)) + " bits",
        encoder.decode_int32(plaintext)
    ))


