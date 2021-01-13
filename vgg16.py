import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, ZeroPadding2D,
                          MaxPooling2D, Activation, Dense, Dropout, Flatten)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import os

MODEL_NAME = "VGG16"

def vgg16():
    x = Input(shape = (28, 28, 1))
    y = ZeroPadding2D(padding = (2, 2))(x) 

    y = multiConvLayer(y, 64, 2) 
    y = multiConvLayer(y, 128, 2) 
    y = multiConvLayer(y, 256, 3) 
    y = multiConvLayer(y, 512, 3) 
    y = multiConvLayer(y, 512, 3) 
    y = Flatten()(y)
    y = Dense(units = 4096, activation='relu')(y)
    y = Dense(units = 4096, activation='relu')(y)
    y = Dense(units = 10)(y)
    y = Activation('softmax')(y)

    return Model(x, y, name = MODEL_NAME)

def multiConvLayer(x, value, n):
    y = x
    for _ in range(n):
        y = Conv2D(value, (3, 3), padding = 'same')(y)
        y = Activation('relu')(y)
    y = MaxPooling2D(strides = (2, 2))(y)
    return y

def checkDir():
    # Used to save the model
    # Not implemented yet
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Used to save the logs
    if not os.path.exists('logs'):
        os.makedirs('logs')

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = to_categorical(y_train) # encode one-hot vector
    y_test = to_categorical(y_test)

    num_of_test_data = 50000
    x_val = x_train[num_of_test_data:]
    y_val = y_train[num_of_test_data:]
    x_train = x_train[:num_of_test_data]
    y_train = y_train[:num_of_test_data]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main():
    # Create directories
    checkDir()

    model = vgg16()
    adam = Adam(lr=1e-4, decay=1e-6)
    model.compile(adam, 'categorical_crossentropy', metrics=['accuracy'])

    # Load dataset
    training_data, validation_data, test_data = load_mnist()

    tensorboard = TensorBoard(write_grads=True, write_images=True)
    chkpoint = ModelCheckpoint("models/weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True)

    # Using 20% of the data as validation data. Used only to verify the correctness of the model, not for training.
    model.fit(training_data[0], training_data[1], epochs=5, callbacks=[tensorboard, chkpoint], validation_split=0.2)

main()
