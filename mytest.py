# Bhatt,Saumya

import pytest
import numpy as np
from cnn import CNN
import os

def test_evaluate():
    from tensorflow.keras.datasets import mnist
    import tensorflow.keras as keras

    batch_size = 128
    num_classes = 10
    epochs = 3

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = CNN()
    model.add_input_layer(shape=(28, 28, 1), name="Input")
    model.append_conv2d_layer(32, kernel_size=(3, 3))
    model.append_conv2d_layer(64, kernel_size=(3, 3))
    model.append_maxpooling2d_layer(pool_size=(2, 2))
    model.append_flatten_layer()
    model.append_dense_layer(128, activation="relu")
    model.append_dense_layer(num_classes, activation="softmax")

    model.set_loss_function("categorical_crossentropy")
    model.set_metric("accuracy")
    model.set_optimizer("Adagrad")

    model.train(x_train, y_train, batch_size=batch_size, num_epochs=epochs)

    score = model.evaluate(x_test, y_test)

    actual = np.array([0.05093684684933396, 0.9907])

    # loss
    np.testing.assert_almost_equal(actual[0], score[0], decimal=2)

    # accuracy
    np.testing.assert_almost_equal(actual[1], score[1], decimal=2)

def test_train():
    from tensorflow.keras.datasets import mnist
    import tensorflow.keras as keras

    batch_size = 128
    num_classes = 10
    epochs = 1

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = CNN()
    model.add_input_layer(shape=(28, 28, 1), name="Input")
    model.append_conv2d_layer(32, kernel_size=(3, 3))
    model.append_conv2d_layer(64, kernel_size=(3, 3))
    model.append_maxpooling2d_layer(pool_size=(2, 2))
    model.append_flatten_layer()
    model.append_dense_layer(128, activation="relu")
    model.append_dense_layer(num_classes, activation="softmax")

    model.set_loss_function("categorical_crossentropy")
    model.set_metric("accuracy")
    model.set_optimizer("Adagrad")

    model.train(x_train, y_train, batch_size=batch_size, num_epochs=epochs)

    score = model.evaluate(x_test, y_test)

    actual = np.array([0.06997422293154523, 0.9907])

    # loss
    np.testing.assert_almost_equal(actual[0], score[0], decimal=2)

    # accuracy
    np.testing.assert_almost_equal(actual[1], score[1], decimal=2)
