import numpy as np
from typing import List

from tqdm import tqdm

from data_loader import load_data
from schemes import *
from lin_algebra import *
from creator import Creator
import math
import time

import low_lat
from logger import debug, debug_colours

class Layer:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = None
        self.weight_scale = None
        self.creator = None

    def check_input_shape(self, input_data: List[HETensor]):
        dim1 = len(input_data)
        dim2, dim3 = input_data[0].shape()

        received_shape = (dim1, dim2, dim3)
        if received_shape != self.input_shape:
            raise Exception('Input data has wrong shape: received %s, expected %s' % (received_shape, self.input_shape))

    def description(self):
        pass

    def configure_input(self, input_shape):
        pass

    def load_weights(self, weights: np.ndarray, extra: np.ndarray):
        pass

    def set_weight_scale(self, scale):
        self.weight_scale = scale

    def apply(self, input_data: List[HETensor]) -> List[HETensor]:
        return input_data

    def apply_fast(self, input_data: HETensor) -> HETensor:
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
        self.dense_mode = False

        super().__init__(input_shape=None, output_shape=None)

    def description(self):
        return 'Conv2D layer, input shape = {}, output shape = {}, weights shape = {}' \
            .format(self.input_shape, self.output_shape, self.weights.shape)

    def configure_input(self, input_shape):
        if len(input_shape) != 3:
            raise Exception('Expected convolution layer input to have 3 dimensions: received %s' % len(input_shape))

        out_h = (input_shape[1] + self.padding * 2 - self.kernel_size[0] + 1) / self.stride
        out_w = (input_shape[2] + self.padding * 2 - self.kernel_size[1] + 1) / self.stride

        out_h = int(math.ceil(out_h))
        out_w = int(math.ceil(out_w))

        if out_h < 1 or out_w < 1:
            raise Exception('Invalid convolutional layer output shape: (%s, %s)' % (out_h, out_w))

        self.input_shape = input_shape
        self.output_shape = (self.num_maps, out_h, out_w)

    def load_weights(self, weights: np.ndarray, extra: np.ndarray):
        expected = (self.kernel_size[0], self.kernel_size[1], self.input_shape[0], self.output_shape[0])
        expected_extra = (self.output_shape[0],)

        if weights.shape != expected:
            raise Exception(
                'Incompatible weights for convolution layer. Expected %s, received %s' % (expected, weights.shape))

        if extra.shape != expected_extra:
            raise Exception('Incompatible bias weights for convolution layer. Expected %s, received %s' % (
            expected_extra, weights.shape))

        self.weights = weights
        self.biases = extra

    def apply_fast(self, input_data: List[HETensor]) -> List[HETensor]:
        print(self.description())

        if not self.dense_mode:
            out = []

            for j in tqdm(range(self.weights.shape[-1])):
                fm_out = None

                for i, t in enumerate(input_data):
                    kernel = self.weights[:, :, i, j]

                    t_detach = t.detatch()[0]

                    if fm_out is None:
                        fm_out = low_lat.convolve(t_detach, kernel)
                    else:
                        fm_out.add_in_place(low_lat.convolve(t_detach, kernel))

                    fm_out.add_raw_in_place(self.biases[i])

                out.append(fm_out)

            return [HETensor(np.array(out))]

        # for j in tqdm(range(self.weights.shape[-1])):
        #     for i, t in enumerate(input_data):
        #         kernel = self.weights[:, :, i, j]
        #         W = low_lat.conv_weights(d_in=self.input_shape[-1], kernel=kernel, s=self.stride)
        #

    def apply(self, input_data: List[HETensor]) -> List[HETensor]:
        print(self.description())

        output_layers = []

        total = self.output_shape[0] * self.output_shape[1] * self.output_shape[2]
        pbar = tqdm(total=total)

        weights = np.transpose(self.weights, (2, 0, 1, 3))

        for j_output in range(self.output_shape[0]):
            layer_mapped = self.creator.zero_mat(self.output_shape[1], self.output_shape[2])

            for r in range(self.output_shape[1]):
                for c in range(self.output_shape[2]):
                    r_v = r * self.stride
                    c_v = c * self.stride

                    kernel = np.squeeze(weights[:, :, :, j_output])
                    kernel_area = None

                    for i_input in range(0, self.input_shape[0]):
                        layer = input_data[i_input]
                        kernel_area_layer = layer.subregion(r_v, r_v + self.kernel_size[0], c_v,
                                                            c_v + self.kernel_size[1])
                        if i_input == 0:
                            kernel_area = kernel_area_layer.flatten()
                        else:
                            kernel_area.column_concat(kernel_area_layer.flatten())

                    val = kernel_area.flatten().row(0)
                    val = val.multiply_element_wise_plain(kernel.flatten().reshape(1, -1)[0])
                    val = val.get_sum()
                    val.add_raw_in_place(self.biases[j_output])

                    layer_mapped.set_element(r, c, val)

                    pbar.update(1)

                    # progress += 1
                    # print('Progress: %.2f percent' % (progress / total * 100))

            output_layers.append(layer_mapped)

        pbar.close()

        return output_layers


def AveragePooling(Layer):
    def __init__(self):
        super().__init__(input_shape=None, output_shape=None)

    def configure_input(self, input_shape):
        self.input_shape = input_shape


class Flatten(Layer):
    def __init__(self, groups=1):
        super().__init__(input_shape=None, output_shape=None)
        self.groups = groups

    def description(self):
        return 'Flatten layer, input shape = {}, output shape = {}'.format(self.input_shape, self.output_shape)

    def configure_input(self, input_shape):
        self.input_shape = input_shape
        k = max(1, input_shape[0])
        for i in range(1, len(input_shape)):
            k *= max(1, input_shape[i])
        self.output_shape = (1, 1, k)

    def apply_fast(self, input_data: List[HETensor]) -> List[HETensor]:
        print(self.description())

        n = self.input_shape[-1] * self.input_shape[-2]
        messages = input_data[0].detatch()[0]
        out = []

        cpg = self.input_shape[0] // self.groups  # channels per group
        sgs = n * cpg  # split group size

        debug('Flatten::apply_fast', 'channels per group: ' + str(cpg), debug_colours.GREEN)
        debug('Flatten::apply_fast', 'group size: ' + str(sgs), debug_colours.GREEN)

        for i in range(self.groups):
            gr = low_lat.concat(messages[i * cpg: (i + 1) * cpg], n)
            out.append(gr)

        return [HETensor(np.array(out))]

    def apply(self, input_data: List[HETensor]) -> List[HETensor]:
        print('-----Applying flatten layer-----')
        temp = input_data[0].flatten()
        for i in range(1, len(input_data)):
            temp.column_concat(input_data[i].flatten())
        return [temp]


class DenseLayer(Layer):
    def __init__(self, output_length):
        super().__init__(input_shape=None, output_shape=(1, 1, output_length))

    def description(self):
        return 'Dense layer, input shape = {}, output shape = {}, weights shape = {}'. \
            format(self.input_shape, self.output_shape, self.weights.shape)

    def configure_input(self, input_shape):
        if input_shape[0] > 1 or input_shape[1] > 1:
            raise Exception('Dense layer should take flattened input. Received ' + str(input_shape))

        self.input_shape = input_shape

    def load_weights(self, weights: np.ndarray, extra: np.ndarray):
        expected = (self.input_shape[2], self.output_shape[2])
        expected_extra = (self.output_shape[2],)
        if weights.shape != expected:
            raise Exception(
                'Incompatible weights for dense layer. Expected %s, received %s' % (expected, weights.shape))
        if extra.shape != expected_extra:
            raise Exception(
                'Expected to have bias weights of dimension %s, received %s' % (expected_extra, extra.shape))
        self.weights = weights
        self.biases = extra

    def apply_fast(self, input_data: List[HETensor]) -> List[HETensor]:
        print(self.description())

        # flattened case
        # real = input_data[0].detatch()[0, 0]
        # W = self.weights.transpose((1, 0))
        #
        # out = low_lat.dense_to_dense(real, W)
        # #out = low_lat.dense_to_sparse(real, W)
        # #out.add_raw_in_place(list(self.biases))
        #
        # return [HETensor(out)]

        # # split input case
        debug('DenseLayer::apply_fast', 'split_input mode on', debug_colours.GREEN)

        groups = input_data[0].detatch()[0]
        sgs = self.input_shape[-1] // len(groups)

        debug('DenseLayer::apply_fast', 'group size: ' + str(sgs), debug_colours.GREEN)

        result = None

        for i, t in enumerate(groups):
            W = self.weights.transpose((1, 0))[:, i * sgs: (i + 1) * sgs]
            o = low_lat.dense_to_dense(t, W)

            if result is None:
                result = o
            else:
                result.add_in_place(o)

        return [HETensor(result)]

    def apply(self, input_data: List[HETensor]) -> List[HETensor]:
        print('-----Applying dense layer-----')
        mat = input_data[0].mult_plain(self.weights, self.weight_scale, debug=True)
        mat.add_raw_in_place(raw_mat=self.biases.reshape(1, -1), debug=True)
        return [mat]


class ActivationLayer(Layer):
    def __init__(self, mode='square'):
        super().__init__(input_shape=None, output_shape=None)
        self.mode = mode

    def description(self):
        return 'Activation layer, mode = {}, input shape = {}, output shape = {}'. \
            format(self.mode, self.input_shape, self.output_shape)

    def configure_input(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape

    def apply(self, input_data: List[HETensor]) -> List[HETensor]:
        print(self.description())

        if self.mode == 'square':
            for layer in input_data:
                layer.square_in_place()

        return input_data

    def apply_fast(self, input_data: List[HETensor]) -> List[HETensor]:
        return self.apply(input_data)

class Model:
    def __init__(self, creator):
        self.layers = []
        self.compiled = False
        self.weights_loaded = False
        self.creator = creator
        self.data_mode = None

    def summary(self):
        if not self.compiled or not self.weights_loaded:
            raise Exception('Model summary cannot be outputed before compilation and weight-loading.')

        text = '-------Summary of model-------\n'

        for layer in self.layers:
            text += layer.description() + '\n'

        print(text)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, input_data, creator: Creator, classes: int):
        if isinstance(input_data, HETensor):
            input_data = [input_data]

        output_data = self.apply(input_data)[0]

        decrypted = np.array(creator.debug(output_data, classes))

        vals = decrypted[0, 0, :]

        return np.argmax(vals), vals

    def apply(self, input_data: List[HETensor]) -> List[HETensor]:
        if not self.compiled:
            raise Exception('Model cannot be used before compilation')

        prev_output = input_data

        for i in range(len(self.layers)):
            if self.data_mode == 0:
                self.layers[i].check_input_shape(prev_output)
                prev_output = self.layers[i].apply(prev_output)
            else:
                prev_output = self.layers[i].apply_fast(prev_output)

        return prev_output

    def add(self, layer):
        layer.creator = self.creator
        self.layers.append(layer)

    def compile(self, data_mode=0):
        if not self.layers:
            raise Exception('Model is empty')

        if not isinstance(self.layers[0], InputLayer):
            raise Exception('No input layer found')

        self.data_mode = data_mode

        input_layer = self.layers[0]
        prev_output_shape = input_layer.output_shape

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            layer.configure_input(input_shape=prev_output_shape)
            prev_output_shape = layer.output_shape

        self.compiled = True

    def load_weights(self, weights, scale: int):
        self.weights_loaded = True

        if len(weights) != len(self.layers):
            raise Exception('Weights error: does not match number of layers in model.')

        for i in range(0, len(self.layers)):
            self.layers[i].set_weight_scale(scale)
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


def cifar():
    N = 8192
    scheme = get_ckks_scheme(512 * 16 * 2)
    creator = Creator(scheme)
    scheme.summary()

    model = Model(creator)
    model.add(InputLayer(input_shape=(3, 32, 32)))
    model.add(ConvLayer(kernel_size=(3, 3), stride=2, padding=0, num_maps=16))
    model.add(ActivationLayer(mode='square'))
    model.add(ConvLayer(kernel_size=(3, 3), stride=2, padding=0, num_maps=16))
    model.add(ActivationLayer(mode='square'))
    model.add(Flatten())
    model.add(DenseLayer(output_length=32))
    model.add(ActivationLayer(mode='square'))
    model.add(DenseLayer(output_length=32))
    model.add(ActivationLayer(mode='square'))
    model.add(DenseLayer(output_length=10))

    weights, test_features, preds_base, outputs_base = load_data(name='CIFAR')

    test_features = test_features[:N]
    preds_base = preds_base[:N]
    outputs_base = outputs_base[:N]

    model.compile()
    model.load_weights(weights, scale=16)
    model.summary()

    print('Encrypting %s items' % len(test_features))

    preds_new, _ = model.predict(input_data=creator.encrypt_simd(mat=test_features),
                                 creator=creator,
                                 length=N)

    print(accuracy(preds_new, preds_base))


def mnist():
    N = 8192
    scheme = get_ckks_scheme(512 * 16 * 2)

    creator = Creator(scheme)
    scheme.summary()

    model = Model(creator)
    model.add(InputLayer(input_shape=(1, 28, 28)))
    model.add(ConvLayer(kernel_size=(4, 4), stride=3, padding=0, num_maps=50))
    model.add(ActivationLayer(mode='square'))
    model.add(Flatten())
    model.add(DenseLayer(output_length=32))
    model.add(Flatten())
    model.add(DenseLayer(output_length=10))

    weights, test_features, preds_base, outputs_base = load_data(name='MNIST')

    test_features = test_features[:N]
    preds_base = preds_base[:N]
    outputs_base = outputs_base[:N]

    model.compile()
    model.load_weights(weights, scale=16)
    model.summary()

    print('Encrypting %s items' % len(test_features))

    preds_new, _ = model.predict(input_data=creator.encrypt_simd(mat=test_features),
                                 creator=creator,
                                 length=N)

    print(accuracy(preds_new, preds_base))


def mnist_fast():
    scheme = get_ckks_scheme(512 * 16 * 2 * 1, g_keys=[8192*2+1])

    creator = Creator(scheme)
    scheme.summary()

    model = Model(creator)
    model.add(InputLayer(input_shape=(1, 28, 28)))
    model.add(ConvLayer(kernel_size=(3, 3), stride=1, padding=0, num_maps=66))
    model.add(ActivationLayer(mode='square'))
    model.add(Flatten(groups=11))
    model.add(DenseLayer(output_length=32))
    model.add(ActivationLayer(mode='square'))
    model.add(DenseLayer(output_length=10))

    weights, test_features, preds_base, outputs_base = load_data(name='MNIST')

    model.compile(data_mode=1)
    model.load_weights(weights, scale=16)
    model.summary()

    for i in range(1):
        image = test_features[i]
        preds_base_i = preds_base[i]
        outputs_base_i = outputs_base[i]

        data = creator.encrypt_dense(mat=image)
        img_groups = creator.obtain_image_groups(mat=image, k=3, s=1)

        pred_new, outputs = model.predict(input_data=img_groups,
                                     creator=creator,
                                     classes=10)

        debug('mnist_fast', str(outputs), debug_colours.GREEN)

        print(pred_new)
        print(preds_base_i)
        print(outputs_base_i)

if __name__ == '__main__':
    mnist_fast()
    #scheme = get_bfv_scheme(512 * 16 * 1, [20,21,22], 16, 16)

    # N = 8192
    # scheme = get_ckks_scheme(512 * 16 * 2)
    #
    # creator = Creator(scheme)
    #
    # mat_a = np.array([[[[0.5], [0.4]],
    #                   [[-0.5], [0.5]]]])
    #
    # mat_a_enc = creator.encrypt(mat=mat_a)[0]
    #
    # mat_a_enc.square_in_place()
    # mat_a_enc.square_in_place()
    # mat_a_enc.square_in_place()
    # mat_a_enc.square_in_place()
    # mat_a_enc.square_in_place()
    # mat_a_enc.square_in_place()
    #
    # mat_a_dec = creator.debug(mat_a_enc, 1)
    #
    # print(mat_a_dec)
    # #print(mat_a * mat_b)

    exit(0)
