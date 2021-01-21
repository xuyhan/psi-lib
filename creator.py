from schemes import *
from lin_algebra import BatchedReal, BatchedRealMat, BatchedRealVec
from batched_real_integer import BatchedRealInteger
from batched_real_double import BatchedRealDouble

import numpy as np

class Creator:
    def __init__(self, scheme: SchemeBase):
        self.scheme = scheme

    def encrypt_value(self, value: np.ndarray) -> BatchedReal:
        if isinstance(self.scheme, CRTScheme):
            return BatchedRealInteger(value, self.scheme.default_scale, self.scheme)
        elif isinstance(self.scheme, SchemeCKKS):
            return BatchedRealDouble(value, self.scheme)

    def encrypt(self, mat: np.ndarray) -> BatchedRealMat:
        if len(mat.shape) != 3:
            raise Exception('Expected input to have three dimensions. Received: %s' % len(mat.shape))

        mat_data = []

        for i in range(mat.shape[1]):
            row_data = []
            for j in range(mat.shape[2]):
                pixel_data = mat[:, i, j].reshape(1, -1)[0]
                batched_real = self.encrypt_value(np.array(pixel_data))
                row_data.append(batched_real)
            mat_data.append(BatchedRealVec(row_data))

        return BatchedRealMat(mat_data)

    def zero(self, batched_real: BatchedReal):
        if isinstance(self.scheme, CRTScheme):
            if batched_real is not None and not isinstance(batched_real, BatchedRealInteger):
                raise Exception('Invalid scheme and value combination.')

            scale = self.scheme.default_scale if batched_real is None else batched_real.scale
            return BatchedRealInteger(np.array([0]), scale, self.scheme)
        elif isinstance(self.scheme, SchemeCKKS):
            if batched_real is not None and not isinstance(batched_real, BatchedRealDouble):
                raise Exception('Invalid scheme and value combination.')

            return BatchedRealDouble(np.array([0]), self.scheme)

    def zero_vec(self, length: int, batched_real=None):
        data = [self.zero(batched_real) for _ in range(length)]
        return BatchedRealVec(data)

    def zero_mat(self, num_rows: int, num_cols: int, batched_real=None):
        data = []
        for _ in range(num_rows):
            data.append(self.zero_vec(num_cols, batched_real))
        return BatchedRealMat(data)

    def debug(self, batched_real_mat: BatchedRealMat, length: int) -> np.ndarray:
        num_rows, num_cols = batched_real_mat.shape()

        result = []

        for i in range(num_rows):
            row = []
            for j in range(num_cols):
                row.append(batched_real_mat.element(i, j).debug(length))
            result.append(row)

        return np.array(result)