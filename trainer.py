import os

from tqdm import tqdm
from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten, Dropout, BatchNormalization, MaxPool2D, Activation, AveragePooling2D
from keras.layers import ZeroPadding2D

from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from keras.callbacks import EarlyStopping

import math
import matplotlib.pyplot as plt

def load_dataset_mnist():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

def load_dataset_cifar10():
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainX = trainX.reshape((trainX.shape[0], 32, 32, 3))
    testX = testX.reshape((testX.shape[0], 32, 32, 3))
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm

def square_activation(x):
    return K.square(x)

def identity(x):
    return x

def mnist_cryptonets():
    model = Sequential()
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(5, (5, 5), strides=2, padding='valid'))
    model.add(Activation(square_activation))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=1, padding='same'))
    model.add(Conv2D(50, (5, 5), strides=2, padding='valid'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=1, padding='same'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation(square_activation))
    model.add(Dense(10, activation='sigmoid'))

    opt = Adam(lr=0.0003)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def mnist_opt():
    model = Sequential()
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(5, (3, 3), strides=1, padding='valid'))
    model.add(Activation(square_activation))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=1, padding='same'))
    model.add(Conv2D(50, (3, 3), strides=1, padding='valid'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=1, padding='same'))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation(square_activation))
    model.add(Dense(10, activation='sigmoid'))

    opt = Adam(lr=0.0003)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def define_model_cifar():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), strides=2))
    model.add(Activation(square_activation))
    model.add(Conv2D(64, (1, 1), strides=1))
    model.add(Activation(square_activation))
    model.add(Dropout(0.3))
    model.add(Flatten())
    # model.add(Dense(64))
    # model.add(Activation(square_activation))
    # model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))

    opt = SGD(lr=0.00003)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# name = 'MNIST-CRYPTONETS'
#
# trainX, trainY, testX, testY = load_dataset_mnist()
# trainX, testX = prep_pixels(trainX, testX)
#
# datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
# train_generator = datagen.flow(trainX, trainY)
#
# model = mnist_cryptonets()
# #model = tf.keras.models.load_model('saved_model/' + name)
#
# model.fit(
#     trainX, trainY,
#     epochs=3,
#     validation_data=(testX, testY),
#     workers=6
# )
#
# model.summary()

def save_keras_weights(model, path, linear=None):
    def rearrange_dense_weights(weights, groups):
        new_weights = []
        for k in range(groups):
            for i in range(weights.shape[0] // groups):
                new_weights.append(weights[i * groups + k])
        new_weights = np.array(new_weights)
        return new_weights

    w = [[None, None]]

    flatten = False
    channels_last = 1

    for i, layer in enumerate(model.layers):
        if linear is not None and linear[0] <= i < linear[1]:
            continue

        if layer.name.startswith('activation'):
            w.append([None, None])
            continue

        if layer.name.startswith('conv2d'):
            w.append([layer.weights[0].numpy(), layer.weights[1].numpy()])
            channels_last = layer.weights[0].shape[-1]
            continue

        if layer.name.startswith('flatten'):
            w.append([None, None])
            flatten = True
            continue

        if layer.name.startswith('dense'):
            if flatten:
                w.append([rearrange_dense_weights(layer.weights[0].numpy(), channels_last),
                          layer.weights[1].numpy()])
                flatten = False
            else:
                w.append([layer.weights[0].numpy(), layer.weights[1].numpy()])

    w = np.array(w, dtype=object)

    print(w.shape)

    np.save(path, w)

def save_data(model, name, linear=None):
    preds = model.predict(testX)
    pred_labs = []
    for v in preds:
        pred_labs.append(np.argmax(v))

    np.save('data/' + name + '_preds.npy', np.array(pred_labs))
    np.save('data/' + name + '_outs.npy', np.array(preds))
    np.save('data/' + name + '_testX.npy', testX)

    save_keras_weights(model, 'data/' + name + '_weights.npy', linear)

# model.save('saved_model/' + name)
# save_data(model, name, linear=(3, 6))


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


# W_c = model.layers[4].weights[0]
# W_b = model.layers[4].weights[1]
# W_c = np.transpose(W_c, (3, 2, 0, 1))
#
# W_1 = conv_weights(d_in=28, W_c=avg_pool_weights(c_in=64, k=3), s=3, padding='same')
# W_2 = conv_weights(d_in=14, W_c=W_c, s=1, padding='valid')
# b_2 = conv_biases(d_in=14, W_c=W_c, biases=W_b, s=1, padding='valid')
# W_3 = conv_weights(d_in=12, W_c=avg_pool_weights(c_in=128, k=3), s=3, padding='same')
#
# W = np.matmul(W_3, np.matmul(W_2, W_1))
# b = np.matmul(W_3, b_2)
#
# random_in = np.arange(28 * 28 * 64)
#
# ref = np.matmul(W_3, np.matmul(W_2, np.matmul(W_1, random_in)))
# test = np.matmul(W, random_in) + b
#
# print(len(ref))
# print(len(test))
#
# plt.plot(ref, label='true')
# plt.plot(test + 100, label='approx')
# plt.show()


name = 'MNIST-OPT'

trainX, trainY, testX, testY = load_dataset_mnist()
trainX, testX = prep_pixels(trainX, testX)

model2 = mnist_opt()

callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

model2.fit(
    trainX, trainY,
    epochs=1,
    validation_data=(testX, testY),
    workers=6,
    callbacks=[callback]
)

model2.summary()

W_conv = model2.layers[4].weights[0]
W_conv = np.transpose(W_conv, (3, 2, 0, 1))
W_conv_biases = model2.layers[4].weights[1]

M_1 = conv_weights(d_in=28, W_c=avg_pool_weights(c_in=5, k=3), s=1, padding='same', pool=True)
M_2 = conv_weights(d_in=28, W_c=W_conv, s=1, padding='valid')
M_3 = conv_weights(d_in=26, W_c=avg_pool_weights(c_in=50, k=3), s=1, padding='same', pool=True)
M_4 = dense_weights(model2.layers[7].weights[0].numpy(), 50).transpose()

b_2 = conv_biases(d_in=28, W_c=W_conv, biases=W_conv_biases, s=1, padding='valid')
b_4 = model2.layers[7].weights[1].numpy()


M = np.matmul(M_2, M_1)
M = np.matmul(M_3, M)
M = np.matmul(M_4, M)

print(M.shape)

if False:
    d_in = 13
    c_in = 5
    s = 2

    W_c = model.layers[4].weights[0]
    W_b = model.layers[4].weights[1]

    W_c = np.transpose(W_c, (3, 2, 0, 1))
    W = conv_weights(d_in=d_in, W_c=W_c, s=s, padding='valid')

    image = np.random.uniform(-1, 1, (1, d_in, d_in, c_in))
    a = np.transpose(model.layers[4](image).numpy(), (0, 3, 1, 2)).flatten()
    b = np.matmul(W, np.transpose(image, (0, 3, 1, 2)).flatten())
    b += conv_biases(d_in=d_in, W_c=W_c, biases=W_b, s=s, padding='valid')

    plt.figure()
    plt.plot(a[:100])
    plt.plot(b[:100] + 0.5)
    plt.show()
#
# if True:
#     d_in = 15
#     s = 2
#     k = 3
#     c_in = 5
#
#     W = conv_weights(d_in=d_in, W_c=avg_pool_weights(c_in, k), s=s, padding='same', pool=True)
#
#     image = np.random.uniform(-1, 1, (1, d_in, d_in, c_in))
#     a = np.transpose(model.layers[3](image).numpy(), (0, 3, 1, 2)).flatten()
#     b = np.matmul(W, np.transpose(image, (0, 3, 1, 2)).flatten())
#
#     plt.figure()
#     plt.plot(a[:200], label='true')
#     plt.plot(b[:200], label='approx')
#     plt.legend()
#     plt.show()