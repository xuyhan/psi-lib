import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

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

class FastMatrix:
    def __init__(self, data, shape, scheme: Scheme):
        self.shape = shape
        self.scheme = scheme

        if isinstance(data, Ciphertext):
            self.ciphertext = data
        else:
            self.ciphertext = scheme.encrypt(data)

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

        n = self.scheme.slot_count()
        self.perm_a = Perm(pa, lambda k: [i for i in range(-self.dim + 1, self.dim)], n)
        self.perm_b = Perm(pb, lambda k: [i * self.dim for i in range(-self.dim + 1, self.dim)], n)
        self.perm_x = Perm(px, lambda k: [k, k - self.dim], n)
        self.perm_y = Perm(py, lambda k: [self.dim * k, self.dim * k - self.size], n)

        self.perm_stack = Perm(p_stack, lambda k: [-(k - self.shape[1]) * i for i in range(self.shape[0])], n)
        self.perm_squeeze = Perm(p_squeeze, lambda k: [(self.shape[1] - k[1]) * i for i in range(k[0])], n)

    def index_map(self, i, j):
        return i * self.shape[1] + j

    def reverse_map(self, i):
        return (i // self.shape[1], i % self.shape[1])

    def get_diag_positions(self, n):
        inds = []
        side_len = self.scheme.slot_count()
        pos = (0, n % side_len)

        for i in range(side_len):
            inds.append(pos)
            pos = (pos[0] + 1, (pos[1] + 1) % side_len)

        return inds

    def perm_diags(self, perm, k=None):
        diags = []
        diag_inds = perm.diag_inds(k)

        for i in diag_inds:
            inds = self.get_diag_positions(i)
            diag = []

            for x, y in inds:
                diag.append(perm(x // self.shape[1], x % self.shape[1], y, k))

            diags.append(diag)

        diags = np.array(diags)
        return diags, diag_inds

    def lin_trans(self, perm, k, ciphertext):
        result = self.scheme.zero()

        diags, diag_inds = self.perm_diags(perm, k)
        num_diags = len(diag_inds)

        j = 0

        for i in range(num_diags):
            diag = list(diags[i])         # + ([0] * (self.scheme.slot_count() // 2))
            diag_ind = diag_inds[i]

            self.scheme.rotate_in_place(ciphertext, diag_ind - j)
            j = diag_ind

            product = self.scheme.multiply_raw(ciphertext, diag)
            self.scheme.add_in_place(result, product)

        self.scheme.rotate_in_place(ciphertext, -j)

        return result

    def mult(self, fast_mat):
        if self.shape[1] != fast_mat.shape[0]:
            raise Exception('Incompatible matrix operands for matrix multiplication: %s x %s' % (self.shape, fast_mat.shape))

        matA = self
        matB = fast_mat

        p = matA.lin_trans(matA.perm_a, None, matA.ciphertext)
        q = matB.lin_trans(matB.perm_b, None, matB.ciphertext)

        self.scheme.relinearise(p)
        self.scheme.relinearise(q)

        cipher_new = self.scheme.zero()

        p_temp = p
        q_temp = q

        start_time = time.process_time()

        for i in range(matA.dim):
            if i > 0:
                p_temp = matA.lin_trans(matA.perm_x, i, p)
                q_temp = matA.lin_trans(matA.perm_y, i, q)

                self.scheme.relinearise(p_temp)
                self.scheme.relinearise(q_temp)

            if i % 10 == 0:
                debug('rot', str((time.process_time() - start_time)), debug_colours.GREEN)
                start_time = time.process_time()

            temp = self.scheme.multiply(p_temp, q_temp)
            self.scheme.relinearise(temp)
            self.scheme.add_in_place(cipher_new, temp)

        result_shape = (self.shape[0], fast_mat.shape[1])

        return FastMatrix(cipher_new, result_shape, self.scheme)

    # def add(self, batched_mat):
    #     if self.shape != batched_mat.shape:
    #         raise Exception('Incompatible matrix operands for matrix addition: %s + %s' %(self.shape, batched_mat.shape))
    #     cipher_new = super().add(batched_mat).ciphertext
    #     return FastMatrix(cipher_new, self.scale, self.shape)

    def debug(self, verbose=False):
        plaintext = self.scheme.decrypt(self.ciphertext)
        raw = self.scheme._batch_decode(plaintext)

        mat = np.zeros(self.shape)

        for i in range(self.shape[0] * self.shape[1]):
            x, y = self.reverse_map(i)
            mat[x][y] = raw[i]

        return mat


if __name__ == '__main__':
    debug('alternative', 'start', debug_colours.GREEN)
    debug('encrypting matrix', 'start', debug_colours.GREEN)

    scheme = get_ckks_scheme(8192 * 4)
    scheme.summary()

    br1 = HERealDouble(np.arange(9), scheme)
    start_time = time.process_time()
    for _ in range(10):
        s = br1.sum(9)
    debug('time', str(time.process_time() - start_time), debug_colours.BOLD)

    print(s.debug(9))

    exit(0)

    a = np.random.uniform(-1, 1, (60, 60))
    b = np.random.uniform(-1, 1, (60, 60))

    fm1 = FastMatrix(a.flatten(), (60, 60), scheme)
    fm2 = FastMatrix(b.flatten(), (60, 60), scheme)

    fm3 = fm1.mult(fm2)

    r1 = np.array(fm3.debug())
    r2 = np.matmul(a, b)
    err = np.sum((r1 - r2) ** 2) / (r1.shape[0] * r1.shape[1])
    debug('main', 'error: ' + str(err), debug_colours.GREEN)


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
    print(plaintext.to_string())

def test_basic():
    int_enc_a = EncryptedInteger.__create__(10)
    i = 0
    while True:
        i += 1
        int_enc_b = EncryptedInteger.__create__(20)
        int_enc_a = int_enc_a.add(int_enc_b)
        budget = evaluate_ciphertext(int_enc_a.ciphertext)
        if budget == 0:
            break

    print('Budget exhausted after %s iterations' %i)

    int_enc_a = EncryptedInteger.__create__(10)
    i = 0
    while True:
        i += 1
        int_enc_b = EncryptedInteger.__create__(20)
        int_enc_a = int_enc_a.multiply(int_enc_b)
        budget = evaluate_ciphertext(int_enc_a.ciphertext)
        if budget == 0:
            break

    print('Budget exhausted after %s iterations' %i)

    real_enc = EncryptedReal.__create__(0.4456, 1024)
    real_enc = real_enc.multiply_raw(0.543)
    reveal_ciphertext(real_enc.int_enc.ciphertext)

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

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = LinearRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    print('Using linear regression model...')
    print('Mean Absolute Error:', mean_absolute_error(y_test, pred))

    print('Using HE inference...')

    # encrypt data and coefficients
    encrypted_X = [encrypt_real_vec(vec) for vec in x_test.to_numpy()]
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

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = LogisticRegression(random_state=0, max_iter=1000, multi_class='ovr')
    pred = model.fit(x_train, y_train).predict(x_test)

    print('\nUsing logistic regression model...')
    print('Classification accuracy: %s' %(np.sum(pred == y_test) / len(pred)))

    predictions = []
    plain_X = x_test.to_numpy()
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

    if False:
        print('Matrix multiplication via naive method...')
        start_time = time.time()
        enc_mat1 = EncryptedRealMatrix.__create__(mat1, scale=s)
        enc_mat2 = EncryptedRealMatrix.__create__(mat2, scale=s)
        enc_result = enc_mat1.mult(enc_mat2)
        print("--- Took %s seconds ---" % (time.time() - start_time))
        b = np.array(decrypt_real_mat(enc_result))
        error(a, b)

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

