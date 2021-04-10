from typing import List

import numpy as np
from tqdm import tqdm

from batched_real import HEReal


class BatchedRealVec:
    def __init__(self, data: List[HEReal]):
        self.data = list(data)

    def len(self):
        return len(self.data)

    def relinearise(self):
        for batched_real in self.data:
            batched_real.relinearise()

    def set_element(self, i, batched_real: HEReal):
        self.data[i] = batched_real

    def element(self, i: int) -> HEReal:
        return self.data[i]

    def subregion(self, i, j):
        return BatchedRealVec(self.data[i: j])

    def append(self, batched_real: HEReal):
        self.data.append(batched_real)

    def prepend(self, batched_real: HEReal):
        self.data = [batched_real] + self.data

    def extend(self, batched_real_vec):
        self.data.extend(batched_real_vec.data)

    def add(self, batched_real_vec):
        self.__vec_enc_check(batched_real_vec)

        new_data = []
        for i in range(len(self.data)):
            new_data.append(self.data[i].add(batched_real_vec.data[i]))

        return BatchedRealVec(new_data)

    def add_raw(self, vec_raw: np.ndarray):
        self.__vec_raw_check(vec_raw)

        new_data = []
        for i in range(len(self.data)):
            new_data.append(self.data[i].add_raw(vec_raw[i]))

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

    def hadamard_mult_plain_in_place(self, vec_raw):
        self.__vec_raw_check(vec_raw)
        [self.data[i].multiply_raw_in_place(vec_raw[i]) for i in range(len(self.data))]

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

class HETensor:
    def __init__(self, data):
        if isinstance(data, HEReal):
            self.data = [BatchedRealVec([data])]
            return

        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                self.data = [BatchedRealVec(data)]
            elif data.ndim == 2:
                self.data = [BatchedRealVec(row) for row in data]
            else:
                raise Exception('Tensors with dim > 2 not supported in current version.')
            return

        '''
        Treating data as List[BatchedRealVec]
        '''
        self.data = data

    @staticmethod
    def __init_np__(data: np.ndarray):
        data_updated = []
        for row in data:
            data_updated.append(BatchedRealVec(row))
        return HETensor(data_updated)

    def detach(self) -> np.ndarray:
        shape = self.shape()
        out = np.zeros(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                out[i][j] = self.element(i, j)
        return out

    def relinearise(self):
        for row in self.data:
            row.relinearise()

    def shape(self):
        return len(self.data), self.data[0].len()

    def set_element(self, i: int, j: int, batched_real: HEReal):
        self.data[i].set_element(j, batched_real)

    def element(self, i: int, j: int) -> HEReal:
        return self.data[i].element(j)

    def row(self, i: int) -> BatchedRealVec:
        return self.data[i]

    def col(self, i: int) -> BatchedRealVec:
        vec_data = [self.element(j, i) for j in range(self.shape()[0])]
        return BatchedRealVec(vec_data)

    def set_element(self, row: int, col: int, val: HEReal):
        self.data[row].set_element(col, val)

    def subregion(self, r_start, r_end, c_start, c_end):
        data_subregion = []
        for vec in self.data[r_start: r_end]:
            data_subregion.append(vec.subregion(c_start, c_end))
        return HETensor(data_subregion)

    def add(self, real_mat):
        if self.shape() != real_mat.shape():
            raise Exception("Error: cannot add matrices with different shapes.")

        new_data = []
        for i in range(self.shape()[0]):
            new_data.append(self.data[i].add(real_mat.data[i]))

        return HETensor(new_data)

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
        return HETensor([vec])

    def square_in_place(self):
        for vec in self.data:
            vec.square_in_place()

    def scale(self, s):
        for i in range(self.shape()[0]):
            for j in range(self.shape()[1]):
                self.element(i, j).multiply_raw_in_place(s)

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
        return HETensor(data_new)

    def mult_plain(self, raw_mat: np.ndarray, multiprocessing=False, debug=False):
        if self.shape()[1] != raw_mat.shape[0]:
            raise Exception("Error cannot multiply %s matrix with %s matrix" %(self.shape(), raw_mat.shape))
        data_new = []
        shape_a = self.shape()
        shape_b = raw_mat.shape

        total = shape_a[0] * shape_b[1]
        bar = tqdm(total=total)

        for i in range(shape_a[0]):
            row_a = self.row(i)
            row_new = [None for _ in range(shape_b[1])]

            def parallel_func(j):
                return row_a.dot_plain(raw_mat[:, j])

            for j in range(shape_b[1]):
                row_new[j] = parallel_func(j)
                bar.update(1)

            data_new.append(BatchedRealVec(row_new))

        bar.close()

        return HETensor(data_new)

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
        return HETensor(data)

    def column_concat(self, batched_real_mat):
        if self.shape()[0] != batched_real_mat.shape()[0]:
            raise Exception('Row mismatch: %s != %s' % (self.shape()[0], batched_real_mat.shape()[0]))

        for i in range(self.shape()[0]):
            self.row(i).extend(batched_real_mat.row(i))

    def noise(self):
        self.element(0, 0).noise()

