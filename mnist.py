from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


from keras import backend as K


def square_activation(x):
    return K.square(x)

def identity(x):
    return x

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(5, (5, 5), strides=4, input_shape=(28, 28, 1), activation=identity))
    model.add(Conv2D(50, (5, 5), strides=4, activation=square_activation))
    model.add(Flatten())
    model.add(Dense(10, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model



# load dataset
import numpy as np


trainX, trainY, testX, testY = load_dataset()
trainX, testX = prep_pixels(trainX, testX)

# trainY = np.random.uniform(-3, 3, (len(trainY), 845))

model = define_model()
model.summary()
model.fit(
    x=trainX,
    y=trainY,
    epochs=8,
)

# model.load_weights('model_weights')


preds = model.predict(testX)
pred_labs = []
for v in preds:
    pred_labs.append(np.argmax(v))
print(pred_labs)

print(model.layers[0].weights[1].numpy())

def rearrange_dense_weights(weights, groups):
    new_weights = []
    for k in range(groups):
        for i in range(weights.shape[0] // groups):
            new_weights.append(weights[i * groups + k])
    new_weights = np.array(new_weights)
    return new_weights

print(model.layers[0].weights[0].numpy().shape)
print(model.layers[1].weights[0].numpy().shape)

w = []
w.append([None, None])
w.append([model.layers[0].weights[0].numpy(), model.layers[0].weights[1].numpy()])
w.append([None, None])
w.append([model.layers[1].weights[0].numpy(), model.layers[1].weights[1].numpy()])
w.append([None, None])
w.append([None, None])
w.append([rearrange_dense_weights(model.layers[3].weights[0].numpy(), 50), model.layers[3].weights[1].numpy()])

w = np.array(w, dtype=object)

np.save('data/weights.npy', w)
np.save('data/model_preds.npy', np.array(pred_labs))
np.save('data/model_outputs.npy', np.array(model.predict(testX)))

exit(0)

def conv(input_data, in_maps, out_maps, out_w, out_h, kernel_w, kernel_h, weights, biases):
    weights = np.transpose(weights, (2, 0, 1, 3))

    output_layers = []

    for j_output in range(out_maps):
        layer_mapped = np.zeros((out_h, out_w), dtype=object)

        for r in range(out_h):
            for c in range(out_w):
                r_v = r * 2
                c_v = c * 2

                kernel = weights[:, :, :, j_output]
                kernel_area = input_data[:, r_v:r_v + kernel_w, c_v:c_v+kernel_h]

                # if in_maps == 3:
                #     print(kernel)
                #     kernel = kernel.flatten()
                #     print(kernel)
                #
                #     print(kernel_area)
                #     kernel_area = kernel_area.flatten()
                #     print(kernel_area)
                #
                #     exit(0)
                #
                kernel = kernel.flatten()
                kernel_area = kernel_area.flatten()

                layer_mapped[r][c] = np.dot(kernel, kernel_area)
                layer_mapped[r][c] += biases[j_output]

        output_layers.append(layer_mapped)

    return np.array(output_layers)

def dense(input_data, weights, biases):
    return np.matmul(input_data, weights) + biases

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def apply(example):
    data = np.transpose(example, (2, 0, 1))
    data = conv(data, 1, 3, 13, 13, 3, 3, w[1][0], w[1][1])
    data = conv(data, 3, 3, 6, 6, 3, 3, w[2][0], w[2][1])
    data = data.flatten()
    data = dense(data, w[4][0], w[4][1])
    data = np.array(data, dtype=float)
    return sigmoid(data)

print(apply(testX[0]))
print(preds[0])



