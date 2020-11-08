import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

from interface import *

def test_dot_product():
    """
    Dot product test.
    """
    v1 = np.random.uniform(-1, 1, 300)
    v2 = np.random.uniform(-1, 1, 300)

    v1_encrypted = encrypt_real_vec(v1)
    v2_encrypted = encrypt_real_vec(v2)


    print("Result via HE: %s" %decrypt_real(v1_encrypted.dot_plain(v2)))
    print("Numpy result: %s" %np.dot(v1, v2))


def test_matrix_mult():
    mat_a_plain = np.random.rand(3, 3)
    mat_b_plain = np.random.rand(3, 3)
    mat_a_enc = encrypt_real_mat(mat_a_plain)
    mat_b_enc = encrypt_real_mat(mat_b_plain)

    print("Adding...")
    print("Non-HE result:")
    print(mat_a_plain + mat_b_plain)
    print("HE result:")
    print(decrypt_real_mat(mat_a_enc.add(mat_b_enc)))

    mat_a_plain = np.random.rand(30, 6)
    mat_b_plain = np.random.rand(6, 30)
    mat_a_enc = encrypt_real_mat(mat_a_plain)
    mat_b_enc = encrypt_real_mat(mat_b_plain)

    print("Multiplying...")
    print("Non-HE result:")
    print(np.matmul(mat_a_plain, mat_b_plain))
    print("HE result:")
    print(decrypt_real_mat(mat_a_enc.mult(mat_b_enc)))



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





if __name__ == '__main__':
    print("starting")
    test_matrix_mult()
