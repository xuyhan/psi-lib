from seal import *
from seal_helper import *
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

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
    def __init__(self, plain_value, scale):
        if plain_value is not None:
            self.ciphertext = encrypt(int(plain_value * scale))
        self.scale = scale

    def add(self, encrypt_real):
        result = EncryptedReal(None, self.scale)
        result.ciphertext = add(self.ciphertext, encrypt_real.ciphertext)
        return result

    def add_in_place(self, encrypted_real):
        add_in_place(self.ciphertext, encrypted_real.ciphertext)

    def add_plain(self, plain_real):
        result = EncryptedReal(None, self.scale)
        result.ciphertext = add_plain(self.ciphertext, int(plain_real * self.scale))
        return result 

    def multiply(self, encrypted_real):
        result = EncryptedReal(None, self.scale * encrypted_real.scale)
        result.ciphertext = multiply(self.ciphertext, encrypted_real.ciphertext)
        return result

    def multiply_in_place(self, encrypted_real):
        multiply_in_place(self.ciphertext, encrypted_real.ciphertext)
        self.scale *= encrypt_real.scale

    def multiply_plain(self, plain_real):
        result = EncryptedReal(None, self.scale ** 2)
        result.ciphertext = multiply_plain(self.ciphertext, int(plain_real * self.scale))
        return result


class EncryptedRealVector:
    def __init__(self, plain_vec, scale):
        if plain_vec is not None:
            self.data = [EncryptedReal(value_plain, scale) for value_plain in plain_vec]

        self.scale = scale

    def len(self):
        return len(self.data)

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

        result = EncryptedRealVector(None, self.scale * real_vec.scale)
        result.data = [self.data[i].multiply(real_vec.data[i]) for i in range(self.len())]
        return result

    def multiply_element_wise_plain(self, real_vec):
        if self.len() != len(real_vec):
            raise Exception("Length mismatch: %s != %s" %(self.len(), real_vec.len()))

        result = EncryptedRealVector(None, self.scale)
        result.data = [self.data[i].multiply_plain(real_vec[i]) for i in range(self.len())]
        return result
            
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


def decrypt_real(encrypted_real):
    return decrypt(encrypted_real.ciphertext) / encrypted_real.scale


def decrypt_real_vec(encrypted_real_vec):
    return [decrypt_real(encrypted_real) for encrypted_real in encrypted_real_vec.data]


def encrypt_real(value_plain):
    return encrypt(int(value_plain * scaling_factor))


def encrypt_real_vec(vec_plain):
    return EncryptedRealVector(vec_plain, scale=scaling_factor)


def evaluate_ciphertext(ciphertext):
    plaintext = Plaintext()
    decryptor.decrypt(ciphertext, plaintext)

    print("size: %s, budget: %s, value: %s" %(
        str(ciphertext.size()),
        str(decryptor.invariant_noise_budget(ciphertext)) + " bits",
        encoder.decode_int32(plaintext)
    ))


def test_dot_product():
    """
    Dot product test.
    """
    v1 = np.random.uniform(-1, 1, 300)
    v2 = np.random.uniform(-1, 1, 300)

    print(v1)
    print(v2)

    v1_encrypted = EncryptedRealVector(v1, scale=scaling_factor)
    v2_encrypted = EncryptedRealVector(v2, scale=scaling_factor)


    print("Result via HE: %s" %decrypt_real(v1_encrypted.dot_plain(v2)))
    print("Numpy result: %s" %np.dot(v1, v2))


def test_linear_regression():
    iris = load_iris()

    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    target_df = pd.DataFrame(data=iris.target, columns=['species'])

    # linear regression: predict sepal length from sepal width, petal length and petal width
    X = iris_df.drop(labels='sepal length (cm)', axis=1)
    y = iris_df['sepal length (cm)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print('Using linear regression model...')
    print('Mean Absolute Error:', mean_absolute_error(y_test, pred))

    print('Using HE inference...')

    # encrypt data and coefficients
    encrypted_X = [encrypt_real_vec(vec) for vec in X_test.to_numpy()]
    pred = []
    for encrypted_feature_vec in encrypted_X:
        pred.append(
            decrypt_real(encrypted_feature_vec.dot_plain(model.coef_).add_plain(model.intercept_))
        )

    print('Mean Absolute Error:', mean_absolute_error(y_test, pred))


def test_logistic_regression():
    iris = load_iris()

    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    target_df = pd.DataFrame(data=iris.target, columns=['species'])

    # logistic regression: predict species
    X = iris_df
    y = target_df['species'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = LogisticRegression(random_state=0, max_iter=1000, multi_class='ovr')
    pred = model.fit(X_train, y_train).predict(X_test)
    print(pred)

    print('\nUsing logistic regression model...')
    print('Classification accuracy: %s' %(np.sum(pred == y_test) / len(pred)))
    print('Mean Absolute Error:', mean_absolute_error(y_test, pred))

    # encrypt data and coefficients 
    # do one vs many classification

    predictions = []
    plain_X = X_test.to_numpy()
    encrypted_X = [encrypt_real_vec(vec) for vec in plain_X]
    M = len(encrypted_X)

    for m in range(M):
        encrypted_vec = encrypted_X[m]
        plain_vec = plain_X[m]
        pred = []
        for i in range(len(model.coef_)):
            pred.append(
                decrypt_real(encrypted_vec.dot_plain(model.coef_[i]).add_plain(model.intercept_[i]))
            )
        predictions.append(np.argmax(pred))

    print(predictions)

    print('Classification accuracy: %s' %(np.sum(predictions == y_test) / len(predictions)))

print("starting")
test_dot_product()
