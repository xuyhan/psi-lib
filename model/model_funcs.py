import numpy as np
from tqdm import tqdm
import math


def conv_weights(d_in, W_c, s, padding='valid', pool=False):
    '''
    obtain weights from kernel, equivalent to Dense(c_in * d_in ** 2, c_out * d_out ** 2),
    expect W_c to be of shape (c_out, c_in, k, k)
    '''
    c_in = W_c.shape[1]
    c_out = W_c.shape[0]
    k = W_c.shape[-1]
    p = None
    if padding == 'valid':
        p = 0
        d_out = int(math.floor((d_in - k + p) / s)) + 1
    elif padding == 'same':
        d_out = math.ceil(d_in / s)

        if d_in % s == 0:
            p = max(k - s, 0)
        else:
            p = max(k - (d_in % s), 0)

    W = np.zeros((c_out * d_out ** 2, c_in * d_in ** 2))

    # iterate through each output
    for j in tqdm(range(c_out * d_out * d_out)):
        c_o = j // (d_out * d_out)
        y_o = j % (d_out * d_out) // d_out
        x_o = j % (d_out * d_out) % d_out

        # iterate through each weight of the filter for c_o
        for i in range(c_in * k * k):
            c_i = i // (k * k)
            k_r = (i % (k * k)) // k
            k_c = i % (k * k) % k

            w = W_c[c_o, c_i, k_r, k_c]

            # find the position of the input value that contributes to output j, under weight w.
            pad_offset = p // 2
            y = y_o * s + k_r - pad_offset
            x = x_o * s + k_c - pad_offset

            # out-of-bounds, could happen due to padding
            if y >= d_in or x >= d_in or y < 0 or x < 0:
                continue

            # tune average pooling values
            if w != 0 and pool:
                t = k ** 2

                x_ = x - k_c
                y_ = y - k_r

                p_x = k * max(
                    max(0, -x_),
                    max(0, x_ + k - d_in)
                )

                p_y = k * max(
                    max(0, -y_),
                    max(0, y_ + k - d_in)
                )

                t -= p_x + p_y - (p_x // k) * (p_y // k)

                w = 1 / t

            W[j][c_i * d_in ** 2 + y * d_in + x] = w

    return W


def conv_biases(d_in, W_c, biases, s, padding='valid'):
    c_out = W_c.shape[0]
    k = W_c.shape[-1]
    if padding == 'valid':
        d_out = int(math.floor((d_in - k) / s)) + 1
    elif padding == 'same':
        d_out = math.ceil(d_in / s)

    b = np.zeros(shape=(c_out, d_out, d_out))

    for c_o in range(c_out):
        b[c_o] = np.ones(shape=(d_out, d_out)) * biases[c_o]

    return b.flatten()


def avg_pool_weights(c_in, k):
    W = np.zeros(shape=(c_in, c_in, k, k))

    for c_i in range(c_in):
        W[c_i, c_i] = np.ones(shape=(k, k)) * (1 / (k ** 2))

    return W


def dense_weights(weights, c_in):
    new_weights = []
    for k in range(c_in):
        for i in range(weights.shape[0] // c_in):
            new_weights.append(weights[i * c_in + k])

    new_weights = np.array(new_weights)
    return new_weights


def dense_weights_reverse(weights, c_in):
    new_weights = []
    for k in range(weights.shape[0] // c_in):
        for i in range(c_in):
            new_weights.append(weights[k * c_in + i])

    new_weights = np.array(new_weights)
    return new_weights


def save_keras_weights(model, path, linear):
    w = []

    flatten = False
    channels_last = 1

    for i, layer in enumerate(model.layers):
        in_lin = False
        for W, b, j_s, j_e in linear:
            if i == j_s:
                w.append([W, b])
            if j_s <= i < j_e:
                in_lin = True
        if in_lin:
            continue

        if layer.name.startswith('conv2d'):
            w.append([layer.weights[0].numpy(), layer.weights[1].numpy()])
            channels_last = layer.weights[0].shape[-1]
            continue

        if layer.name.startswith('flatten'):
            flatten = True
            continue

        if layer.name.startswith('dense'):
            if flatten:
                w.append([dense_weights(layer.weights[0].numpy(), channels_last),
                          layer.weights[1].numpy()])
                flatten = False
            else:
                w.append([layer.weights[0].numpy(), layer.weights[1].numpy()])

    w = np.array(w, dtype=object)

    np.save(path, w)


def save_data(model, name, testX, linear):
    if testX is not None:
        preds = model.predict(testX)

        pred_labs = []
        for v in preds:
            pred_labs.append(np.argmax(v))

        np.save('data/' + name + '_preds.npy', np.array(pred_labs))
        np.save('data/' + name + '_outs.npy', np.array(preds))
        np.save('data/' + name + '_testX.npy', testX)

    save_keras_weights(model, 'data/' + name + '_weights.npy', linear)
