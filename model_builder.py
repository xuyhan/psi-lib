import numpy as np
from typing import List

from interface import BatchedRealMat, BatchedReal, Creator

class Layer:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = None

    def check_input_shape(self, input_data: List[BatchedRealMat]):
        dim1 = len(input_data)
        dim2, dim3 = input_data[0].shape()

        received_shape = (dim1, dim2, dim3)
        if received_shape != self.input_shape:
            raise Exception('Input data has wrong shape: received %s, expected %s' %(received_shape, self.input_shape))

    def description(self):
        pass

    def configure_input(self, input_shape):
        pass

    def load_weights(self, weights: np.ndarray, extra: np.ndarray):
        pass

    def apply(self, input_data: List[BatchedRealMat]) -> List[BatchedRealMat]:
        return input_data


class InputLayer(Layer):
    def __init__(self, input_shape):
        super().__init__(input_shape=input_shape, output_shape=input_shape)

    def description(self):
        return 'Input Layer, input shape = ' + str(self.input_shape)


class ConvLayer(Layer):
    def __init__(self, kernel_size, stride, padding, num_maps):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_maps = num_maps

        super().__init__(input_shape=None, output_shape=None)

    def description(self):
        return 'Conv2D layer, input shape = {}, output shape = {}, weights shape = {}'\
            .format(self.input_shape, self.output_shape, self.weights.shape)

    def configure_input(self, input_shape):
        if len(input_shape) != 3:
            raise Exception('Expected convolution layer input to have 3 dimensions: received %s' % len(input_shape))

        out_h = (input_shape[1] + self.padding * 2 - self.kernel_size[0] + 1) // 2
        out_w = (input_shape[2] + self.padding * 2 - self.kernel_size[1] + 1) // 2

        if out_h < 1 or out_w < 1:
            raise Exception('Invalid convolutional layer output shape: (%s, %s)' % (out_h, out_w))

        self.input_shape = input_shape
        self.output_shape = (self.num_maps, out_h, out_w)

    def load_weights(self, weights: np.ndarray, extra: np.ndarray):
        expected = (self.kernel_size[0], self.kernel_size[1], self.input_shape[0], self.output_shape[0])
        expected_extra = (self.output_shape[0],)

        if weights.shape != expected:
            raise Exception('Incompatible weights for convolution layer. Expected %s, received %s' % (expected, weights.shape))

        if extra.shape != expected_extra:
            raise Exception('Incompatible bias weights for convolution layer. Expected %s, received %s' % (expected_extra, weights.shape))

        self.weights = weights
        self.biases = extra

    def apply(self, input_data: List[BatchedRealMat]) -> List[BatchedRealMat]:
        print('-----Applying conv layer-----')

        for layer in input_data:
            for _ in range(self.padding):
                layer.pad_in_place()

        scale = input_data[0].scale ** 2
        output_layers = []

        total = self.output_shape[0] * self.input_shape[0] * self.output_shape[1] * self.output_shape[2]
        step = total // 10
        progress = 0

        for j_output in range(self.output_shape[0]):
            map_j = BatchedRealMat.zeros(num_rows=self.output_shape[1],
                                         num_cols=self.output_shape[2],
                                         scale=scale,
                                         scheme=input_data[0].scheme)

            for i_input in range(self.input_shape[0]):
                kernel = np.squeeze(self.weights[:, :, i_input, j_output])
                layer = input_data[i_input]
                layer_mapped = np.zeros((self.output_shape[1], self.output_shape[2]), dtype=object)

                for r in range(self.output_shape[1]):
                    for c in range(self.output_shape[2]):
                        r_v = r * self.stride
                        c_v = c * self.stride
                        kernel_area = layer.subregion(r_v, r_v + self.kernel_size[0], c_v, c_v + self.kernel_size[1])

                        layer_mapped[r][c] = kernel_area.flatten().mult_plain(kernel.flatten().reshape(-1, 1)).element(0, 0)
                        layer_mapped[r][c].add_raw_in_place(self.biases[j_output])

                        progress += 1
                        if progress % step == 0:
                            print('Progress: %.2f percent' % (progress / total * 100))

                layer_mapped = BatchedRealMat.__init_np__(data=layer_mapped,
                                                          scale=scale,
                                                          scheme=input_data[0].scheme)
                map_j.add_in_place(layer_mapped)

            output_layers.append(map_j)

        return output_layers

def AveragePooling(Layer):
    def __init__(self):
        super().__init__(input_shape=None, output_shape=None)

    def configure_input(self, input_shape):
        self.input_shape = input_shape

class Flatten(Layer):
    def __init__(self):
        super().__init__(input_shape=None, output_shape=None)

    def description(self):
        return 'Flatten layer, input shape = {}, output shape = {}'.format(self.input_shape, self.output_shape)

    def configure_input(self, input_shape):
        self.input_shape = input_shape
        k = max(1, input_shape[0])
        for i in range(1, len(input_shape)):
            k *= max(1, input_shape[i])
        self.output_shape = (1, 1, k)

    def apply(self, input_data: List[BatchedRealMat]) -> List[BatchedRealMat]:
        print('-----Applying flatten layer-----')
        temp = input_data[0].flatten()
        for i in range(1, len(input_data)):
            temp.column_concat(input_data[i].flatten())
        return [temp]

class DenseLayer(Layer):
    def __init__(self, output_length):
        super().__init__(input_shape=None, output_shape=(1, 1, output_length))

    def description(self):
        return 'Dense layer, input shape = {}, output shape = {}, weights shape = {}'.\
            format(self.input_shape, self.output_shape, self.weights.shape)

    def configure_input(self, input_shape):
        if input_shape[0] > 1 or input_shape[1] > 1:
            raise Exception('Dense layer can only take flattened input. Received ' + str(input_shape))
        self.input_shape = input_shape

    def load_weights(self, weights: np.ndarray, extra: np.ndarray):
        expected = (self.input_shape[2], self.output_shape[2])
        expected_extra = (self.output_shape[2],)
        if weights.shape != expected:
            raise Exception('Incompatible weights for dense layer. Expected %s, received %s' % (expected, weights.shape))
        if extra.shape != expected_extra:
            raise Exception('Expected to have bias weights of dimension %s, received %s' % (expected_extra, extra.shape))
        self.weights = weights
        self.biases = extra

    def apply(self, input_data: List[BatchedRealMat]) -> List[BatchedRealMat]:
        print('-----Applying dense layer-----')
        mat = input_data[0].mult_plain(self.weights, debug=True)
        mat.add_raw_in_place(raw_mat=self.biases.reshape(1, -1), debug=True)
        return [mat]

class ActivationLayer(Layer):
    def __init__(self, mode='square'):
        super().__init__(input_shape=None, output_shape=None)
        self.mode = mode

    def description(self):
        return 'Activation layer, mode = {}, input shape = {}, output shape = {}'.\
            format(self.mode, self.input_shape, self.output_shape)

    def configure_input(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape

    def apply(self, input_data: List[BatchedRealMat]) -> List[BatchedRealMat]:
        print('-----Applying activation layer-----')

        if self.mode == 'square':
            for layer in input_data:
                layer.square_in_place()
        return input_data

class Model:
    def __init__(self):
        self.layers = []
        self.compiled = False
        self.weights_loaded = False

    def summary(self):
        if not self.compiled or not self.weights_loaded:
            raise Exception('Model summary cannot be outputed before compilation and weight-loading.')


        text = '-------Summary of model-------\n'

        for layer in self.layers:
            text += layer.description() + '\n'

        print(text)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, input_data: List[BatchedRealMat], creator: Creator, length: int) -> List[int]:
        output_shape = self.layers[-1].output_shape
        if output_shape[0] > 1 or output_shape[1] > 1:
            raise Exception('Cannot predict when last layer is not flat'
                            )
        output_data = self.apply(input_data)[0]

        decrypted = np.array(creator.debug(output_data, length))
        # for i in range(10):
        #     l = list(decrypted[:, :, i][0])
        #     print([self.sigmoid(x) for x in l])

        predictions = []
        outputs = []

        for i in range(length):
            vals = decrypted[:, :, i][0]

            predictions.append(np.argmax(vals))
            outputs.append(vals)

        return predictions, outputs


    def apply(self, input_data: List[BatchedRealMat]) -> List[BatchedRealMat]:
        if not self.compiled:
            raise Exception('Model cannot be used before compilation')
        prev_output = input_data
        for i in range(len(self.layers)):
            self.layers[i].check_input_shape(prev_output)
            prev_output = self.layers[i].apply(prev_output)
            prev_output[0].noise()
        return prev_output

    def add(self, layer):
        self.layers.append(layer)

    def compile(self):
        if not self.layers:
            raise Exception('Model is empty')

        if not isinstance(self.layers[0], InputLayer):
            raise Exception('No input layer found')

        input_layer = self.layers[0]
        prev_output_shape = input_layer.output_shape

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            layer.configure_input(input_shape=prev_output_shape)
            prev_output_shape = layer.output_shape

        self.compiled = True

    def load_weights(self, weights):
        self.weights_loaded = True

        if len(weights) != len(self.layers):
            raise Exception('Weights error: does not match number of layers in model.')

        for i in range(0, len(self.layers)):
            self.layers[i].load_weights(weights=weights[i][0], extra=weights[i][1])

    def get_input_shape(self):
        if not self.compiled:
            raise Exception('Model not compiled yet')
        return self.layers[0].input_shape

    def get_output_shape(self):
        if not self.compiled:
            raise Exception('Model not compiled yet')
        return self.layers[-1].output_shape


def accuracy(a, b):
    if len(a) != len(b):
        raise Exception('Length mismatch')
    count = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            count += 1
    return count / len(a)

if __name__ == '__main__':
    from interface import get_scheme, Creator

    # scheme = get_scheme()
    # creator = Creator(scheme, scale=2048)
    # scheme.summary()
    # m = np.array([[
    #     [.1,.2,-.3],
    #     [.4,.55,.6]
    # ]])
    # n = np.array([
    #     [-.14,.25,.53],
    #     [.44,-.5,.66],
    #     [.44, -.5,.166]
    # ])
    #
    # m_enc = creator.encrypt(m)
    #
    # z = m_enc.mult_plain(n)
    # z.square_in_place()
    #
    # #m_enc.square_in_place()
    # print(creator.debug(z, 1))
    # z.noise()
    #
    # print((np.matmul(m, n) ** 2) ** 2)
    # exit(0)


    model = Model()
    model.add(InputLayer(input_shape=(1, 28, 28)))
    model.add(ConvLayer(kernel_size=(3, 3), stride=2, padding=0, num_maps=5))
    model.add(ActivationLayer(mode='square'))
    model.add(Flatten())
    model.add(DenseLayer(output_length=10))
    model.add(DenseLayer(output_length=10))
    #model.add(ActivationLayer(mode='square'))
    #model.add(DenseLayer(output_length=10))

    weights = np.load('weights.npy', allow_pickle=True)

    model.compile()
    model.load_weights(weights)
    model.summary()

    mnist_train_features = np.load('mnist_test_features.npy')[:2000]
    mnist_train_features = mnist_train_features.squeeze()

    preds_base = np.load('model_preds.npy')[:2000]
    outputs_base = np.load('model_outputs.npy')[:2000]

    # preds_new = []
    # outputs_new = []
    #
    #
    # def conv2d(image, w, w2):
    #     if w.shape != (3, 3, 1, 5):
    #         raise Exception('a')
    #
    #     result = []
    #
    #     for k in range(5):
    #         x = 0
    #         mat = np.zeros((13, 13))
    #         while x <= 24:
    #             y = 0
    #             while y <= 24:
    #                 a = image[y:y+3,x:x+3].flatten()
    #                 b = w[:, :, 0, k].flatten()
    #                 mat[y // 2][x // 2] = np.dot(a, b) + w2[k]
    #                 y += 2
    #             x += 2
    #         result.append(mat)
    #
    #     return np.array(result)
    #
    # for i in range(2000):
    #     image = mnist_train_features[i]
    #     conv_result = conv2d(image, weights[1][0], weights[1][1])
    #
    #     flat = conv_result.reshape(1, -1)
    #     flat = flat ** 2
    #
    #     x = np.matmul(flat, weights[4][0])
    #
    #     x = x.flatten()
    #
    #     x += weights[4][1]
    #
    #     outputs_new.append(list(x))
    #     preds_new.append(np.argmax(x))
    #
    #
    # preds_new = np.array(preds_new)
    #
    # print(outputs_new[543])
    # print(outputs_base[543])

    scheme = get_scheme(512 * 16 * 1, [17,18,20,21,22,23,24])
    creator = Creator(scheme, scale=16)
    print('------Summary of scheme-------')
    scheme.summary()

    print('Encrypting %s items' % len(mnist_train_features))
    encrypted = creator.encrypt(mat=mnist_train_features)

    print('Beginning inference...')
    preds_new, outputs_new = model.predict(input_data=[encrypted], creator=creator, length=2000)

    print(accuracy(preds_new, preds_base))
