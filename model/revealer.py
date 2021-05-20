from crypto.schemes import *
from model.lin_algebra import HETensor


class Revealer:
    def __init__(self, scheme: Scheme, decryptor: DecryptorBase):
        self.scheme = scheme
        self.decryptor = decryptor

    def reveal(self, he_tensor: HETensor, length: int) -> np.ndarray:
        num_rows, num_cols = he_tensor.shape()

        result = []

        for i in range(num_rows):
            row = []
            for j in range(num_cols):
                plaintext = self.decryptor.decrypt(he_tensor.element(i, j).ciphertext)
                raw = self.scheme._batch_decode(plaintext)
                row.append(raw[:length])

            result.append(row)

        return np.array(result)

    def reveal_outputs(self, output_data, length, data_mode=0):
        plain = np.array(self.reveal(output_data, length))

        if data_mode == 0:
            return plain

        return plain[0, 0, :]
