import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

from schemes import *
import time

def test_basic():
    # int_enc_a = EncryptedInteger.__create__(10)
    # i = 0
    # while True:
    #     i += 1
    #     int_enc_b = EncryptedInteger.__create__(20)
    #     int_enc_a = int_enc_a.add(int_enc_b)
    #     budget = evaluate_ciphertext(int_enc_a.ciphertext)
    #     if budget == 0:
    #         break

    # print('Budget exhausted after %s iterations' %i)

    # int_enc_a = EncryptedInteger.__create__(10)
    # i = 0
    # while True:
    #     i += 1
    #     int_enc_b = EncryptedInteger.__create__(20)
    #     int_enc_a = int_enc_a.multiply(int_enc_b)
    #     budget = evaluate_ciphertext(int_enc_a.ciphertext)
    #     if budget == 0:
    #         break

    # print('Budget exhausted after %s iterations' %i)

    real_enc = EncryptedReal.__create__(0.4456, 1024)
    real_enc = real_enc.multiply_raw(0.543)
    reveal_ciphertext(real_enc.int_enc.ciphertext)

 
    # print('10*20=%s' %decrypt_int(int_enc_c))


def test_dot_product():
    """
    Dot product test.
    """
    v1 = np.random.uniform(-1, 1, 50)
    v2 = np.random.uniform(-1, 1, 50)

    v1_encrypted = encrypt_real_vec(v1)
    v2_encrypted = encrypt_real_vec(v2)
    result = v1_encrypted.dot_plain(v2)

    print("Result via HE: %s" %decrypt_real(result))
    print("Numpy result: %s" %np.dot(v1, v2))


def test_matrix_mult():
    mat_a_plain = np.random.rand(3, 3)
    mat_b_plain = np.random.rand(3, 3)
    mat_a_enc = encrypt_real_mat(mat_a_plain)
    mat_b_enc = encrypt_real_mat(mat_b_plain)

    print("Adding...")
    print("Non-HE rfgesult:")
    print(mat_a_plain + mat_b_plain)
    print("HE result:")
    print(decrypt_real_mat(mat_a_enc._add_ip(mat_b_enc)))

    mat_a_plain = np.random.rand(30, 6)
    mat_b_plain = np.random.rand(6, 30)
    mat_a_enc = encrypt_real_mat(mat_a_plain)
    mat_b_enc = encrypt_real_mat(mat_b_plain)

    print("Multiplying...")
    print("Non-HE result:")
    print(np.matmul(mat_a_plain, mat_b_plain))
    print("HE result:")
    print(decrypt_real_mat(mat_a_enc._mult(mat_b_enc)))



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

    print('\nUsing logistic regression model...')
    print('Classification accuracy: %s' %(np.sum(pred == y_test) / len(pred)))

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

    print('Classification accuracy: %s' %(np.sum(predictions == y_test) / len(predictions)))


def error(a, b):
    error_mat = np.abs((a - b) / a)
    avg_error = round(np.average(error_mat), 3)
    max_error = round(np.max(error_mat), 3)
    print('Average error: %s' %avg_error)
    print('Max error: %s' %max_error)

def matrx_mult():
    print('Matrix multiplication demo\n')

    mat1 = np.random.uniform(-1, 1, (30, 30))
    mat2 = np.random.uniform(-1, 1, (30, 31))
    a = np.matmul(mat1, mat2)
    s = 1024

    # print('Matrix multiplication via naive method...')
    # start_time = time.time()
    # enc_mat1 = EncryptedRealMatrix.__create__(mat1, scale=s)
    # enc_mat2 = EncryptedRealMatrix.__create__(mat2, scale=s)
    # enc_result = enc_mat1.mult(enc_mat2)
    # print("--- Took %s seconds ---" % (time.time() - start_time))
    # b = np.array(decrypt_real_mat(enc_result))
    # error(a, b)

    print()
    print('Matrix multiplication via permutation method...')
    start_time = time.time()
    batched_mat1 = BatchedMat(mat1, scale=s, shape=mat1.shape)
    batched_mat2 = BatchedMat(mat2, scale=s, shape=mat2.shape)
    b = batched_mat1._mult(batched_mat2).debug()
    print("--- Took %s seconds ---" % (time.time() - start_time))
    error(a, b)

def dot_prod():
    vec1 = np.random.uniform(-1, 1, 60)
    vec2 = np.random.uniform(-1, 1, 60)

    a = np.array([np.dot(vec1, vec2)])

    print('Vector dot product via naive method...')
    start_time = time.time()
    v1_encrypted = encrypt_real_vec(vec1)
    v2_encrypted = encrypt_real_vec(vec2)
    b = np.array([decrypt_real(v1_encrypted.dot(v2_encrypted))])
    print("--- Took %s seconds ---" % (time.time() - start_time))
    error(a, b)

    print('Vector dot product via batching method...')
    start_time = time.time()

    v1_encrypted = BatchedVec(vec1, scale=1024)
    v2_encrypted = BatchedVec(vec2, scale=1024)
    b = np.array([v1_encrypted.dot(v2_encrypted).debug()])

    print("--- Took %s seconds ---" % (time.time() - start_time))
    error(a, b)

if __name__ == '__main__':
    matrx_mult()

