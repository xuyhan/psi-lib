import os

from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten, Dropout, BatchNormalization, MaxPool2D, Activation
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# load train and test dataset
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

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(66, (3, 3), strides=1))
    model.add(Activation(square_activation))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation(square_activation))
    model.add(Dense(10, activation='softmax'))

    opt = Adam(lr=0.002)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

trainX, trainY, testX, testY = load_dataset_mnist()
trainX, testX = prep_pixels(trainX, testX)

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

model = define_model()
model.fit(
    trainX,
    trainY,
    epochs=8,
    validation_data=(testX, testY),
    workers=6
)
model.summary()

model.save_weights('model_weights')


def save_keras_weights(model, path):
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

    for layer in model.layers:
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


def save_data(model, name):
    preds = model.predict(testX)
    pred_labs = []
    for v in preds:
        pred_labs.append(np.argmax(v))

    np.save('data/' + name + '_preds.npy', np.array(pred_labs))
    np.save('data/' + name + '_outs.npy', np.array(preds))
    np.save('data/' + name + '_testX.npy', testX)

    save_keras_weights(model, 'data/' + name + '_weights.npy')

save_data(model, 'MNIST')



# exit(0)
#
# def conv(input_data, in_maps, out_maps, out_w, out_h, kernel_w, kernel_h, weights, biases):
#     weights = np.transpose(weights, (2, 0, 1, 3))
#
#     output_layers = []
#
#     for j_output in range(out_maps):
#         layer_mapped = np.zeros((out_h, out_w), dtype=object)
#
#         for r in range(out_h):
#             for c in range(out_w):
#                 r_v = r * 2
#                 c_v = c * 2
#
#                 kernel = weights[:, :, :, j_output]
#                 kernel_area = input_data[:, r_v:r_v + kernel_w, c_v:c_v+kernel_h]
#
#                 # if in_maps == 3:
#                 #     print(kernel)
#                 #     kernel = kernel.flatten()
#                 #     print(kernel)
#                 #
#                 #     print(kernel_area)
#                 #     kernel_area = kernel_area.flatten()
#                 #     print(kernel_area)
#                 #
#                 #     exit(0)
#                 #
#                 kernel = kernel.flatten()
#                 kernel_area = kernel_area.flatten()
#
#                 layer_mapped[r][c] = np.dot(kernel, kernel_area)
#                 layer_mapped[r][c] += biases[j_output]
#
#         output_layers.append(layer_mapped)
#
#     return np.array(output_layers)
#
# def dense(input_data, weights, biases):
#     return np.matmul(input_data, weights) + biases
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# def apply(example):
#     data = np.transpose(example, (2, 0, 1))
#     data = conv(data, 1, 3, 13, 13, 3, 3, w[1][0], w[1][1])
#     data = conv(data, 3, 3, 6, 6, 3, 3, w[2][0], w[2][1])
#     data = data.flatten()
#     data = dense(data, w[4][0], w[4][1])
#     data = np.array(data, dtype=float)
#     return sigmoid(data)
#
# print(apply(testX[0]))
# print(preds[0])



