from keras.models import Sequentials
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras
import numpy as np

classes = ["dog", "cat"]
num_classes = len(classes)
image_size = 50
input_image_num = 200
epoch_num = 20


def main():
    X_train, X_test, Y_train, Y_test = np.load(
        "./animal.npy", allow_pickle=True)

    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    Y_train = np.utils.to_categorical(y_train, num_classes)
    y_test = np.utils.to_categorical(y_test, num_classes)

    if not os.path.exists('animal_cnn.h5'):
        model = model_train(X_train, Y_train)
    else:
        # model =

    model_eval(model, X_test, Y_test)


def model_train(X, Y):
    model = Sequentials()

    model.add(Conv2D(32, (3, 3,), padding='same',
                     input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3,), padding='same')
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
    mdoel.add(Activation('softmax'))

    opt=eras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 最適化処理
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])  # 損失関数

    model.fit(X, Y, batch_size=32, nb_epoch=epoch_num)

    mode.save('./animal_cnn.h5')

    return model

def model_eval(model, X, Y):
    scores=model.evaluate(X, Y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Accuracy: ', scores[1])


if __name__ == "__main__":
    main()
