import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, ZeroPadding2D,
                          MaxPooling2D, Activation, Dense, Dropout, Flatten)
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
import os
import sys

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

MODEL_NAME = "VGG16"

def vgg16_mnist():
    x = Input(shape = (28, 28, 1))
    y = ZeroPadding2D(padding = (2, 2))(x) 

    y = multiConvLayer(y, 64, 2) # size 32x32
    y = multiConvLayer(y, 128, 2) # size 16x16
    y = multiConvLayer(y, 256, 3) # size 8x8
    y = multiConvLayer(y, 512, 3) # size 4x4
    y = multiConvLayer(y, 512, 3) # size 2x2
    y = Flatten()(y)
    y = Dense(units = 4096, activation='relu')(y)
    y = Dense(units = 4096, activation='relu')(y)
    y = Dense(units = 10)(y)
    y = Activation('softmax')(y)

    return Model(x, y, name = MODEL_NAME)

def vgg16_digits():
    x = Input(shape = (16, 15, 1))
    y = ZeroPadding2D(padding = (8, 9))(x) 

    y = multiConvLayer(y, 64, 2) # size 32x32
    y = multiConvLayer(y, 128, 2) # size 16x16
    y = multiConvLayer(y, 256, 3) # size 8x8
    y = multiConvLayer(y, 512, 3) # size 4x4
    y = multiConvLayer(y, 512, 3) # size 2x2
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
    y = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(y)
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

def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models/model.h5")

    print("Saved model to disk")

def load_model():
    # load json and create model
    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/model.h5")

    print("Loaded model from disk")

    return loaded_model

def load_digits():
    Data = np.transpose(np.loadtxt("Digits_Dataset.txt"))
    Data = Data.transpose()
    Data = Data.astype(int)
    Data.shape

    x = 0
    Classes = np.array([])
    Classes = Classes.astype(int)

    for i in range(1, 2001):
        Classes = np.append(Classes, x)
        if(i != 0 and i % 200 == 0):
            x = x + 1
            continue

    X_Digits, X_test_Digits, Y_Digits, Y_test_Digits = train_test_split(Data, Classes, test_size=0.2, stratify = Classes, random_state=10)
    batch_size = 25
    Y_Digits_Total = np.concatenate((Y_Digits, Y_test_Digits), axis = None)

    X_Digits = X_Digits / 6.
    X_test_Digits = X_test_Digits / 6.

    X_Digits = X_Digits.reshape(-1, 16, 15, 1)
    X_test_Digits = X_test_Digits.reshape (-1, 16, 15, 1)

    Y_Digits = to_categorical(Y_Digits)
    Y_test_Digits = to_categorical(Y_test_Digits)

    X_train_Digits, X_val_Digits, Y_train_Digits, Y_val_Digits = train_test_split(X_Digits, Y_Digits, test_size = 0.2, random_state=10, stratify = Y_Digits)

    return (X_train_Digits, Y_train_Digits), (X_val_Digits, Y_val_Digits), (X_test_Digits, Y_test_Digits)

def main():
    # Create directories
    checkDir()

    # Define Adam Optimizer
    adam = Adam(lr=1e-4, decay=1e-6)
    
    # Load dataset
    training_data, val_data, test_data = load_digits()

    tensorboard = TensorBoard(write_grads=True, write_images=True)
    chkpoint = ModelCheckpoint("models/weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True)

    if len(sys.argv) == 1:
        print("Please mention \"train\" or \"test\" as arguments, without the \" symbol.")
    elif sys.argv[1] == "train":
        model = vgg16_digits()
        model.compile(adam, 'categorical_crossentropy', metrics=['accuracy'])
        # Using 20% of the data as validation data. Used only to verify the correctness of the model, not for training.
        model.fit(training_data[0], training_data[1], validation_data=(val_data[0], val_data[1]), epochs=50, callbacks=[tensorboard, chkpoint])
        save_model(model)
    elif sys.argv[1] == "test":
        model = load_model()
        model.compile(adam, 'categorical_crossentropy', metrics=['accuracy'])
    else:
        print("Invalid argument")
        sys.exit()

    score = model.evaluate(test_data[0], test_data[1], verbose=0)
    print(score)

main()
