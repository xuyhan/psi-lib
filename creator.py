from schemes import *
from lin_algebra import HEReal, HETensor, BatchedRealVec
from batched_real_integer import HERealInteger
from batched_real_double import HERealDouble
from low_lat import find_groups

import numpy as np
from tqdm import tqdm


class Creator:
    def __init__(self, scheme: SchemeBase):
        self.scheme = scheme

    def encrypt_value(self, value: np.ndarray) -> HEReal:
        if isinstance(self.scheme, CRTScheme):
            return HERealInteger(value, self.scheme.default_scale, self.scheme)
        elif isinstance(self.scheme, SchemeCKKS):
            return HERealDouble(value, self.scheme)

    def encrypt_simd(self, mat: np.ndarray) -> List[HETensor]:
        if len(mat.shape) != 4:
            raise Exception('Expected input to have four dimensions. Received: %s' % len(mat.shape))

        total = mat.shape[3] * mat.shape[2] * mat.shape[1]
        pbar = tqdm(total=total)

        channels = []
        for c in range(mat.shape[3]):
            mat_data = []
            for i in range(mat.shape[1]):
                row_data = []
                for j in range(mat.shape[2]):
                    pixel_data = mat[:, i, j, c].reshape(1, -1)[0]
                    batched_real = self.encrypt_value(np.array(pixel_data))
                    row_data.append(batched_real)

                    pbar.update(1)
                mat_data.append(BatchedRealVec(row_data))
            channels.append(HETensor(mat_data))

        pbar.close()
        return channels

    def encrypt_dense(self, mat: np.ndarray) -> HETensor:
        t = []

        for c in range(mat.shape[2]):
            pixels = mat[:, :, c].flatten()
            t.append(self.encrypt_value(pixels))

        return HETensor(np.array(t))

    def obtain_image_groups(self, mat: np.ndarray, k: int, s: int) -> np.ndarray:
        result = []
        for c in range(mat.shape[2]):
            fm = mat[:, :, c]

            gs = []
            for g in find_groups(fm, k, s):
                gs.append(self.encrypt_value(g))
            result.append(gs)

        return result

    def zero(self, batched_real=None):
        if isinstance(self.scheme, CRTScheme):
            if batched_real is not None and not isinstance(batched_real, HERealInteger):
                raise Exception('Invalid scheme and value combination.')

            scale = self.scheme.default_scale if batched_real is None else batched_real.scale
            return HERealInteger(np.array([0]), scale, self.scheme)
        elif isinstance(self.scheme, SchemeCKKS):
            if batched_real is not None and not isinstance(batched_real, HERealDouble):
                raise Exception('Invalid scheme and value combination.')

            return HERealDouble(np.array([0]), self.scheme)

    def zero_vec(self, length: int, batched_real=None):
        data = [self.zero(batched_real) for _ in range(length)]
        return BatchedRealVec(data)

    def zero_mat(self, num_rows: int, num_cols: int, batched_real=None) -> HETensor:
        data = []
        for _ in range(num_rows):
            data.append(self.zero_vec(num_cols, batched_real))
        return HETensor(data)

    def debug(self, batched_real_mat: HETensor, length: int) -> np.ndarray:
        num_rows, num_cols = batched_real_mat.shape()

        result = []

        for i in range(num_rows):
            row = []
            for j in range(num_cols):
                row.append(batched_real_mat.element(i, j).debug(length))
            result.append(row)

        return np.array(result)