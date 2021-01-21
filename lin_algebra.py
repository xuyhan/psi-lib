from typing import List
from multiprocessing import Pool
from batched_real import BatchedReal
from batched_real_integer import BatchedRealInteger

import numpy as np

class BatchedRealVec:
    def __init__(self, data: List[BatchedReal]):
        self.data = data

    def len(self):
        return len(self.data)

    def relinearise(self):
        for batched_real in self.data:
            batched_real.relinearise()

    def set_element(self, i, batched_real: BatchedReal):
        self.data[i] = batched_real

    def element(self, i: int) -> BatchedReal:
        return self.data[i]

    def subregion(self, i, j):
        return BatchedRealVec(self.data[i: j])

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

        return BatchedRealVec(new_data)

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

        return BatchedRealVec(new_data)

    def multiply_element_wise_plain(self, vec_raw):
        self.__vec_raw_check(vec_raw)
        new_data = [self.data[i].multiply_raw(vec_raw[i]) for i in range(len(self.data))]
        return BatchedRealVec(new_data)

    def square(self):
        new_data = np.array([self.data[i].square() for i in range(len(self.data))])
        return BatchedRealVec(new_data)

    def square_in_place(self):
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
        if self.len() != batched_real_vec.len():
            raise Exception("Length mismatch: %s != %s" % (self.data.len(), batched_real_vec.data.len()))

    def __vec_raw_check(self, vec_raw):
        if self.len() != vec_raw.shape[0]:
            raise Exception("Length mismatch: %s != %s" % (len(self.data), len(vec_raw)))

class BatchedRealMat:
    def __init__(self, data: List[BatchedRealVec]):
        self.data = data

    @staticmethod
    def __init_np__(data: np.ndarray):
        data_updated = []
        for row in data:
            data_updated.append(BatchedRealVec(row))
        return BatchedRealMat(data_updated)

    def relinearise(self):
        for row in self.data:
            row.relinearise()

    def shape(self):
        return len(self.data), self.data[0].len()

    def set_element(self, i: int, j: int, batched_real: BatchedReal):
        self.data[i].set_element(j, batched_real)

    def element(self, i: int, j: int) -> BatchedReal:
        return self.data[i].element(j)

    def row(self, i: int) -> BatchedRealVec:
        return self.data[i]

    def col(self, i: int) -> BatchedRealVec:
        vec_data = [self.element(j, i) for j in range(self.shape()[0])]
        return BatchedRealVec(vec_data)

    def set_element(self, row: int, col: int, val: BatchedReal):
        self.data[row].set_element(col, val)

    def subregion(self, r_start, r_end, c_start, c_end):
        data_subregion = []
        for vec in self.data[r_start: r_end]:
            data_subregion.append(vec.subregion(c_start, c_end))
        return BatchedRealMat(data_subregion)

    def add(self, real_mat):
        if self.shape() != real_mat.shape():
            raise Exception("Error: cannot add matrices with different shapes.")

        new_data = []
        for i in range(self.shape()[0]):
            new_data.append(self.data[i].add(real_mat.data[i]))

        return BatchedRealMat(new_data)

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
        vec = BatchedRealVec(list(result))
        return BatchedRealMat([vec])

    def square_in_place(self):
        for vec in self.data:
            vec.square_in_place()

    def mult(self, real_mat):
        if self.shape()[1] != real_mat.shape()[0]:
            raise Exception("Error cannot multiply %s matrix with %s matrix" %(self.shape, real_mat.shape))
        data_new = []
        for i in range(self.shape()[0]):
            row_new = [None for _ in range(real_mat.shape()[1])]
            for j in range(real_mat.shape()[1]):
                row_a = self.row(i)
                col_b = real_mat.col(j)
                row_new[j] = row_a.dot(col_b)
            data_new.append(BatchedRealVec(row_new))
        return BatchedRealMat(data_new)

    def mult_plain(self, raw_mat: np.ndarray, multiprocessing=False, debug=False):
        if self.shape()[1] != raw_mat.shape[0]:
            raise Exception("Error cannot multiply %s matrix with %s matrix" %(self.shape(), raw_mat.shape))
        data_new = []
        shape_a = self.shape()
        shape_b = raw_mat.shape

        pool = Pool(processes=12)

        if debug:
            print('Starting matrix multiplication...')

            total_steps = shape_a[0] * shape_b[1]
            progress_step = total_steps // 10

        for i in range(shape_a[0]):
            row_a = self.row(i)
            row_new = [None for _ in range(shape_b[1])]

            def parallel_func(j):
                return row_a.dot_plain(raw_mat[:, j])

            for j in range(shape_b[1]):
                row_new[j] = parallel_func(j)

                progress = i * shape_a[0] + j
                if debug and progress % progress_step == 0:
                    print('Progress: %.2f' % (progress / total_steps))

            data_new.append(BatchedRealVec(row_new))

        return BatchedRealMat(data_new)

    # def pad_in_place(self):
    #     old_h, old_w = self.shape()
    #     for vec in self.data:
    #         vec.prepend(self.creator.zero(self.element(0, 0)))
    #         vec.append(self.creator.zero(self.element(0, 0)))
    #
    #     self.data = [self.creator.zero_vec(old_w + 2, self.element(0, 0))] + self.data
    #     self.data.append(self.creator.zero_vec(old_w + 2, self.element(0, 0)))

    def transpose(self):
        data = []
        for j in range(self.shape()[1]):
            row_data = []
            for i in range(self.shape()[0]):
                row_data.append(self.row(i).element(j))
            data.append(BatchedRealVec(data=row_data))
        return BatchedRealMat(data)

    def column_concat(self, batched_real_mat):
        if self.shape()[0] != batched_real_mat.shape()[0]:
            raise Exception('Row mismatch: %s != %s' % (self.shape()[0], batched_real_mat.shape()[0]))

        for i in range(self.shape()[0]):
            self.row(i).extend(batched_real_mat.row(i))

    def noise(self):
        self.element(0, 0).noise()

