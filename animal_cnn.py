from keras.models import Sequential
# from tensorflow.keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras
import numpy as np

classes = ["dog", "cat", "turtle"]
num_classes = len(classes)
image_size = 50
input_image_num = 200
epoch_num = 10


def main():
    X_train, X_test, Y_train, Y_test = np.load(
        "./animal.npy", allow_pickle=True)

    X_train = X_train.astype("float") / 256 # 要確認
    X_test = X_test.astype("float") / 256
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_test = np_utils.to_categorical(Y_test, num_classes)

    # if not os.path.exists('animal_cnn.h5'):
    model = model_train(X_train, Y_train, X_test, Y_test)
    # else:
    # model =

    print(model.summary())

    model_eval(model, X_test, Y_test)


def model_train(X_train, Y_train, X_test, Y_test):
    model = Sequential()

    model.add(Conv2D(32, (3, 3,), padding='same',
                     input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3,), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())    # Flat処理、一列にする

    model.add(Dense(512))  # 全結合層
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 最適化処理
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])  # 損失関数

    # hist = model.fit(X_train, Y_train, batch_size=32, epochs=epoch_num, validation_split=0.2)
    hist = model.fit(X_train, Y_train, batch_size=32, epochs=epoch_num, validation_data=(X_test, Y_test))

    print(hist.history)
    model.save('./animal_cnn.h5')

    return model


def model_eval(model, X, Y):
    scores = model.evaluate(X, Y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Accuracy: ', scores[1])


if __name__ == "__main__":
    main()
