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
    model.add(Conv2D(5, (3, 3), strides=2, input_shape=(28, 28, 1), activation=square_activation))
    model.add(Flatten())
    model.add(Dense(10, activation=identity))
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
model.fit(
    x=trainX,
    y=trainY,
    epochs=10,
)


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

w = []
w.append([None, None])
w.append([model.layers[0].weights[0].numpy(), model.layers[0].weights[1].numpy()])
w.append([None, None])
w.append([None, None])
w.append([rearrange_dense_weights(model.layers[2].weights[0].numpy(), 5), model.layers[2].weights[1].numpy()])
w.append([model.layers[3].weights[0].numpy(), model.layers[3].weights[1].numpy()])

w = np.array(w, dtype=object)

np.save('weights.npy', w)
np.save('model_preds.npy', np.array(pred_labs))
np.save('model_outputs.npy', np.array(model.predict(testX)))

