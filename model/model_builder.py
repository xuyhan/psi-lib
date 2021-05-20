import math
import time
from typing import Dict

import model.low_lat as low_lat
from crypto.schemes import *
from model.creator import Creator
from model.lin_algebra import *
from utils.logger import msg, OutFlags
from utils.utils import merge_ops, init_ops


class Layer:
    def __init__(self, input_shape, output_shape, name, requires_weights=True):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = None
        self.weight_scale = None
        self.creator = None
        self.name = name
        self.requires_weights = requires_weights

    def check_input_shape(self, input_data: List[HETensor]):
        dim1 = len(input_data)
        dim2, dim3 = input_data[0].shape()

        received_shape = (dim1, dim2, dim3)
        if received_shape != self.input_shape:
            raise Exception('Input data has wrong shape: received %s, expected %s' % (received_shape, self.input_shape))

    def configure_input(self, input_shape):
        pass

    def load_weights(self, weights: np.ndarray, extra: np.ndarray):
        pass

    def set_weight_scale(self, scale):
        self.weight_scale = scale

    def apply(self, input_data: List[HETensor]) -> (List[HETensor], Dict):
        return input_data, init_ops()

    def apply_fast(self, input_data: HETensor) -> (List[HETensor], Dict):
        return input_data, init_ops()


class InputLayer(Layer):
    def __init__(self, input_shape):
        super().__init__(input_shape=input_shape, output_shape=input_shape, requires_weights=False, name='input')


class ConvLayer(Layer):
    def __init__(self, kernel_size, stride, padding, num_maps, dense_mode=False):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_maps = num_maps
        self.dense_mode = dense_mode

        super().__init__(input_shape=None, output_shape=None, name='conv')

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

    def apply_fast(self, input_data: List[HETensor]) -> (List[HETensor], Dict):
        out = []

        ops = init_ops()

        for j in tqdm(range(self.weights.shape[-1])):
            fm_out = None

            kernel = self.weights[:, :, :, j].transpose((2, 0, 1))

            groups = input_data[0].detach()[0]

            if fm_out is None:
                fm_out, ops_ = low_lat.convolve3D(groups, self.output_shape[-1], kernel)
                ops = merge_ops(ops, ops_)
            else:
                fm, ops_ = low_lat.convolve3D(groups, self.output_shape[-1], kernel)
                fm_out.add_in_place(fm)
                ops['addCC'] += 1
                ops = merge_ops(ops, ops_)

            bias = [self.biases[j] for _ in range(self.output_shape[-1] * self.output_shape[-2])]
            fm_out.add_raw_in_place(bias)
            ops['addPC'] += 1
            out.append(fm_out)

        return [HETensor(np.array(out))], ops

    def apply(self, input_data: List[HETensor]) -> (List[HETensor], Dict):
        output_layers = []

        total = self.output_shape[0] * self.output_shape[1] * self.output_shape[2]
        pbar = tqdm(total=total)

        weights = np.transpose(self.weights, (2, 0, 1, 3))

        ops = init_ops()

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

        return output_layers, ops


class Flatten(Layer):
    def __init__(self, groups=1):
        super().__init__(input_shape=None, output_shape=None, requires_weights=False, name='flatten')
        self.groups = groups

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

        cpg = int(math.ceil(self.input_shape[0] / float(self.groups)))  # channels per group
        cpg_last = self.input_shape[0] % cpg
        cpg_last = cpg if cpg_last == 0 else cpg_last

        msg('Flatten::apply_fast', 'channels per group: ' + str(cpg), OutFlags.INFO)
        msg('Flatten::apply_fast', 'channels in last group: ' + str(cpg_last), OutFlags.INFO)

        ops = init_ops()

        for i in range(self.groups):
            cpg_ = cpg_last if i == self.groups - 1 else cpg
            gr, ops_ = low_lat.concat(messages[i * cpg: i * cpg + cpg_], n)
            out.append(gr)
            ops = merge_ops(ops, ops_)

        return [HETensor(np.array(out))], ops

    def apply(self, input_data: List[HETensor]) -> (List[HETensor], Dict):
        temp = input_data[0].flatten()
        for i in range(1, len(input_data)):
            temp.column_concat(input_data[i].flatten())

        return [temp], init_ops()


class Unflatten(Layer):
    def __init__(self, channels):
        super().__init__(input_shape=None, output_shape=None, requires_weights=False, name='unflatten')
        self.channels = channels

    def configure_input(self, input_shape):
        if np.prod(input_shape) % self.channels != 0:
            raise Exception('Cannot unflatten %s to %s channels' % (str(input_shape), str(self.channels)))
        t = np.prod(input_shape) // self.channels
        if not math.sqrt(t).is_integer():
            raise Exception('%s not square' % t)

        self.input_shape = input_shape
        self.output_shape = (self.channels, int(math.sqrt(t)), int(math.sqrt(t)))

    def apply_fast(self, input_data: List[HETensor]) -> (List[HETensor], Dict):
        vec_in = input_data[0].detach()[0, 0]
        ops = init_ops()
        messages, ops_ = low_lat.unconcat(vec_in, int(np.prod(self.output_shape)), int(np.prod(self.output_shape)) //
                                          self.channels)
        ops = merge_ops(ops, ops_)

        return [HETensor(np.array(messages))], ops


class DenseLayer(Layer):
    def __init__(self, output_length, prev_channels=1):
        self.prev_channels = prev_channels
        super().__init__(input_shape=None, output_shape=(1, 1, output_length), name='dense')

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

    def apply_fast(self, input_data: List[HETensor]) -> (List[HETensor], Dict):
        ops = init_ops()

        groups = input_data[0].detach()[0]
        # sgs = self.input_shape[-1] // len(groups)

        cpg = int(math.ceil(self.prev_channels / len(groups)))  # channels per group
        cpg_last = self.prev_channels % cpg
        cpg_last = cpg if cpg_last == 0 else cpg_last
        sgs = cpg * (self.input_shape[-1] // self.prev_channels)
        sgs_last = cpg_last * (self.input_shape[-1] // self.prev_channels)

        msg('DenseLayer::apply_fast', 'group size: ' + str(sgs), OutFlags.INFO)
        msg('DenseLayer::apply_fast', 'last group size: ' + str(sgs_last), OutFlags.INFO)

        result = None

        for i, t in enumerate(groups):
            sgs_ = sgs_last if i == len(groups) - 1 else sgs
            W = self.weights.transpose((1, 0))[:, i * sgs: i * sgs + sgs_]
            o, ops_ = low_lat.dense_to_dense(t, W)
            ops = merge_ops(ops, ops_)

            if result is None:
                result = o
            else:
                result.add_in_place(o)
                ops['addCC'] += 1

        result.add_raw_in_place(list(self.biases))
        ops['addPC'] += 1

        return [HETensor(result)], ops

    def apply(self, input_data: List[HETensor]) -> (List[HETensor], Dict):
        ops = init_ops()

        mat, ops_ = input_data[0].mult_plain(self.weights, self.creator)
        ops = merge_ops(ops, ops_)
        ops_ = mat.add_raw_in_place(raw_mat=self.biases.reshape(1, -1))
        ops = merge_ops(ops, ops_)

        return [mat], ops


class SubLayer(Layer):
    def __init__(self, size):
        super().__init__(input_shape=(2, 1, size), output_shape=(1, 1, size), requires_weights=False, name='distance')
        self.size = size

    def configure_input(self, input_shape):
        if input_shape != self.input_shape:
            raise Exception('Expected input to have shape %s' % self.input_shape)

    def apply(self, input_data: List[HETensor]) -> (List[HETensor], Dict):
        ops = init_ops()

        half_first = input_data[0]
        half_second = input_data[1]
        half_first.sub_in_place(half_second)
        half_first.square_in_place()

        ops['addCC'] += self.size
        ops['mulCC'] += self.size

        return [half_first], ops


class ActivationLayer(Layer):
    def __init__(self, mode='square'):
        super().__init__(input_shape=None, output_shape=None, requires_weights=False, name='activation_' + mode)
        self.mode = mode

    def configure_input(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape

    def apply(self, input_data: List[HETensor]) -> (List[HETensor], Dict):
        ops = init_ops()

        if self.mode == 'square':
            for layer in input_data:
                layer.square_in_place()
                ops['mulCC'] += layer.size()
        elif self.mode != 'identity':
            raise Exception('Unknown activation: ' + self.mode)

        return input_data, ops

    def apply_fast(self, input_data: List[HETensor]) -> (List[HETensor], Dict):
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

        print('-------Summary of model-------')

        format = '{:<27}' * 3
        print(OutFlags.BOLD + format.format('Name', 'Input shape', 'Output shape') + OutFlags.END)

        for layer in self.layers:
            print(format.format(layer.name, str(layer.input_shape), str(layer.output_shape)))

    def predict(self, input_data):
        if isinstance(input_data, HETensor):
            input_data = [input_data]

        output_data = self.apply(input_data)[0]

        return output_data

    def apply(self, input_data: List[HETensor]) -> List[HETensor]:
        if not self.compiled:
            raise Exception('Model cannot be used before compilation')

        prev_output = input_data
        ops_all = []
        deltas_all = []

        t_start = time.process_time()
        t0 = t_start
        for i, layer in enumerate(self.layers):
            msg('Applying layer', layer.name, OutFlags.INFO)

            if self.data_mode == 0:
                layer.check_input_shape(prev_output)
                prev_output, ops = layer.apply(prev_output)
            else:
                prev_output, ops = layer.apply_fast(prev_output)

            t1 = time.process_time()
            delta = t1 - t0
            msg('Time elapsed', str(round(delta, 5)), OutFlags.INFO)
            t0 = t1

            ops_all.append(ops)
            deltas_all.append(delta)

        msg('Total inference time', str(round(time.process_time() - t_start, 5)), OutFlags.INFO)

        print('------Summary of operations------')
        format = '{:<20}' + '{:<10}' * 7
        print(OutFlags.BOLD + format.format('Layer', '#HOPS', '#AddPC', '#AddCC', '#MulPC', '#MulCC', '#Rots',
                                            'Time') + OutFlags.END)
        table = []

        for i, layer in enumerate(self.layers):
            ops = ops_all[i]
            hops = np.sum([v for v in ops.values()])
            delta = round(deltas_all[i], 3)
            table_row = [layer.name, hops, ops['addPC'], ops['addCC'], ops['mulPC'], ops['mulCC'], ops['rots'], delta]
            table.append(table_row)
            print(format.format(*[v if v != 0 else '-' for v in table_row]))

        final_row = ['Total', ]
        for i in range(1, 8):
            final_row.append(round(np.sum([row[i] for row in table]), 3))

        print(format.format(*[v if v != 0 else '-' for v in final_row]))

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

        for i, layer in enumerate(self.layers):
            layer.name += str(i)

        self.compiled = True

    def load_weights(self, weights, scale: int):
        self.weights_loaded = True

        required_weights = 0
        for l in self.layers:
            if l.requires_weights:
                required_weights += 1

        if len(weights) != required_weights:
            raise Exception('Weights error: does not match number of layers in model: %s' % required_weights)

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


def depth_test():
    scheme, _ = init_scheme_ckks(512 * 16 * 2 * 2, primes=[60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 60],
                                 scale_factor=50)
    creator = Creator(scheme)
    scheme.summary()

    a = np.arange(10) / 10
    v = creator.encrypt_value(a)
    v.square_in_place()
    v.square_in_place()
    v.square_in_place()
    v.square_in_place()
    print(v.msg(10))
