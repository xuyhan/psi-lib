import math
import time

from data_loader import load_data
from logger import debug, debug_colours
from schemes import *

import low_lat
from creator import Creator
from lin_algebra import *
from utils import merge_ops


class Layer:
    def __init__(self, input_shape, output_shape, requires_weights=True):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = None
        self.weight_scale = None
        self.creator = None
        self.requires_weights = requires_weights

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
    def __init__(self, kernel_size, stride, padding, num_maps, dense_mode=False):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_maps = num_maps
        self.dense_mode = dense_mode

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
        def method1(in_data):
            out = []

            mult_cc = 0
            add_pc = 0
            add_cc = 0

            for j in tqdm(range(self.weights.shape[-1])):
                fm_out = None

                kernel = self.weights[:, :, :, j].transpose((2, 0, 1))

                groups = input_data[0].detach()[0]

                if fm_out is None:
                    fm_out, m, a = low_lat.convolve3D(groups, self.output_shape[-1], kernel)
                else:
                    fm, m, a = low_lat.convolve3D(groups, self.output_shape[-1], kernel)
                    fm_out.add_in_place(fm)
                    add_cc += a

                mult_cc += m

                bias = [self.biases[j] for _ in range(self.output_shape[-1] * self.output_shape[-2])]
                fm_out.add_raw_in_place(bias)
                add_pc += 1

                out.append(fm_out)

            debug('conv2d', f'#multCC: {mult_cc}, #addPC: {add_pc}, #addCC: {add_cc}', debug_colours.BLUE)

            return [HETensor(np.array(out))]

        return method1(input_data)

    def apply(self, input_data: List[HETensor]) -> List[HETensor]:
        output_layers = []

        total = self.output_shape[0] * self.output_shape[1] * self.output_shape[2]
        pbar = tqdm(total=total)

        weights = np.transpose(self.weights, (2, 0, 1, 3))

        ops = {'addPC': 0}

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
                    val, ops_ = val.multiply_element_wise_plain(kernel.flatten().reshape(1, -1)[0])
                    ops = merge_ops(ops, ops_)
                    val, ops_ = val.get_sum()
                    ops = merge_ops(ops, ops_)

                    if val is not None:
                        val.add_raw_in_place(self.biases[j_output])
                        ops['addPC'] += 1
                        layer_mapped.set_element(r, c, val)

                    pbar.update(1)

            output_layers.append(layer_mapped)

        pbar.close()

        for k, v in ops.items():
            print(k + ' ' + str(v))

        return output_layers

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
        n = self.input_shape[-1] * self.input_shape[-2]
        messages = input_data[0].detach()[0]

        out = []

        cpg = self.input_shape[0] // self.groups  # channels per group
        sgs = n * cpg  # split group size

        debug('Flatten::apply_fast', 'channels per group: ' + str(cpg), debug_colours.GREEN)
        debug('Flatten::apply_fast', 'group size: ' + str(sgs), debug_colours.GREEN)

        rots_total = 0
        adds_total = 0

        for i in range(self.groups):
            gr, rots, adds = low_lat.concat(messages[i * cpg: (i + 1) * cpg], n)
            out.append(gr)
            rots_total += rots
            adds_total += adds

        debug('flatten::', f'#rots: {rots_total}, #add-CC: {adds_total}', debug_colours.BLUE)

        return [HETensor(np.array(out))]

    def apply(self, input_data: List[HETensor]) -> List[HETensor]:
        print('-----Applying flatten layer-----')
        temp = input_data[0].flatten()
        for i in range(1, len(input_data)):
            temp.column_concat(input_data[i].flatten())
        return [temp]

class Unflatten(Layer):
    def __init__(self, channels):
        super().__init__(input_shape=None, output_shape=None, requires_weights=False)
        self.channels = channels

    def description(self):
        return 'Unflatten layer, input shape = {}, output shape = {}'.format(self.input_shape, self.output_shape)

    def configure_input(self, input_shape):
        if np.prod(input_shape) % self.channels != 0:
            raise Exception('Cannot unflatten %s to %s channels' %(str(input_shape), str(self.channels)))
        t = np.prod(input_shape) // self.channels
        if not math.sqrt(t).is_integer():
            raise Exception('%s not square' % t)

        self.input_shape = input_shape
        self.output_shape = (self.channels, int(math.sqrt(t)), int(math.sqrt(t)))

    def apply_fast(self, input_data: List[HETensor]) -> List[HETensor]:
        vec_in = input_data[0].detach()[0, 0]
        messages, rots = low_lat.unconcat(vec_in, int(np.prod(self.output_shape)), int(np.prod(self.output_shape)) //
                                    self.channels)

        debug('unconcat::', f'#rots: {rots}', debug_colours.BLUE)\

        return [HETensor(np.array(messages))]

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
        debug('DenseLayer::apply_fast', 'split_input mode on', debug_colours.GREEN)

        groups = input_data[0].detach()[0]
        sgs = self.input_shape[-1] // len(groups)

        debug('DenseLayer::apply_fast', 'group size: ' + str(sgs), debug_colours.GREEN)

        result = None

        rots, mults, adds = 0, 0, 0

        for i, t in enumerate(groups):
            W = self.weights.transpose((1, 0))[:, i * sgs: (i + 1) * sgs]
            o, r, m, a = low_lat.dense_to_dense(t, W)
            rots += r
            mults += m
            adds += a

            if result is None:
                result = o
            else:
                result.add_in_place(o)

        result.add_raw_in_place(list(self.biases))

        debug('dense', f'#rots: {rots}, #multPC: {mults}, #addCC: {adds}, #addPC: 1', debug_colours.BLUE)

        if self.output_shape[-1] == 10:
            np.save('10outputs.npy', result.debug(4096))

        return [HETensor(result)]

    def apply(self, input_data: List[HETensor]) -> List[HETensor]:
        print('-----Applying dense layer-----')
        ops = {}
        mat, ops_ = input_data[0].mult_plain(self.weights, self.creator)
        ops = merge_ops(ops, ops_)
        ops_ = mat.add_raw_in_place(raw_mat=self.biases.reshape(1, -1))
        ops = merge_ops(ops, ops_)

        for k, v in ops.items():
            print(k + ' ' + str(v))

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
        if self.mode == 'square':
            num_ct = 0
            for layer in input_data:
                layer.square_in_place()
                num_ct += layer.size()
            debug('square::', f'#mult-CC: {num_ct}', debug_colours.BLUE)
            return input_data

        if self.mode == 'identity':
            return input_data

        raise Exception('unknown activation: ' + str(self.mode))

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

    def predict(self, input_data, creator: Creator, classes: int):
        if isinstance(input_data, HETensor):
            input_data = [input_data]

        output_data = self.apply(input_data)[0]

        decrypted = np.array(creator.debug(output_data, classes))

        if self.data_mode == 0:
            return decrypted
        return decrypted[0, 0, :]

    def apply(self, input_data: List[HETensor]) -> List[HETensor]:
        if not self.compiled:
            raise Exception('Model cannot be used before compilation')

        prev_output = input_data

        t0 = time.process_time()
        for i in range(len(self.layers)):
            if self.data_mode == 0:
                self.layers[i].check_input_shape(prev_output)
                prev_output = self.layers[i].apply(prev_output)
            else:
                prev_output = self.layers[i].apply_fast(prev_output)
            t1 = time.process_time()
            debug('Layer ' + str(i), str(t1 - t0), debug_colours.PURPLE)
            t0 = t1

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

        required_weights = 0
        for l in self.layers:
            if l.requires_weights:
                required_weights += 1

        if len(weights) != required_weights:
            raise Exception('Weights error: does not match number of layers in model.')

        j = 0
        for i in range(0, len(self.layers)):
            if not self.layers[i].requires_weights:
                continue
            self.layers[i].set_weight_scale(scale)
            self.layers[i].load_weights(weights=weights[j][0], extra=weights[j][1])
            j += 1

    def get_input_shape(self):
        if not self.compiled:
            raise Exception('Model not compiled yet')
        return self.layers[0].input_shape

    def get_output_shape(self):
        if not self.compiled:
            raise Exception('Model not compiled yet')
        return self.layers[-1].output_shape

def accuracy(a, b):
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

def depth_test():
    scheme = get_ckks_scheme(512 * 16 * 2 * 2, primes=[60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 60], scale_factor=50)
    creator = Creator(scheme)
    scheme.summary()

    a = np.arange(10) / 10
    v = creator.encrypt_value(a)
    v.square_in_place()
    v.square_in_place()
    v.square_in_place()
    v.square_in_place()
    print(v.debug(10))


