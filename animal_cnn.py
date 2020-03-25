from keras.models import Sequentials
from keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np

classes = ["dog", "cat"]
num_classes = len(classes)
image_size = 50
input_image_num = 200


def main():
    X_train, X_test, Y_train, Y_test = np.load("./animal.npy")

    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    Y_train = np.utils.to_categorical(y_train, num_classes)
    y_test = np.utils.to_categorical(y_test, num_classes)

    model = model_train(X_train, Y_train)
    model_eval(model, X_test, Y_test)


def model_train():


def model_eval():


if __name__ == "__main__":
    main()
