from keras.models import Sequential

# from tensorflow.keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras
import numpy as np
import pickle
import os
import time
import h5py
import matplotlib.pyplot as plt

classes = ["dog", "cat"]
num_classes = len(classes)
image_size = 50
input_image_num = 300
epoch_num = 100
database_path = "./database/epoch_" + str(epoch_num)


def main():
    if not os.path.exists(database_path):
        os.mkdir(database_path)

    X_train, X_test, Y_train, Y_test = np.load(
        "./database/animal.npy", allow_pickle=True
    )

    X_train = X_train.astype("float") / 256  # 要確認
    X_test = X_test.astype("float") / 256
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_test = np_utils.to_categorical(Y_test, num_classes)

    if not os.path.exists(database_path + "/animal_cnn.h5"):
        print("\n===== Training =====")
        start_time = time.time()
        model = model_train(X_train, Y_train, X_test, Y_test)
        print("Calc Time: [" + str(time.time() - start_time) + "]")
    else:
        print("\n===== Model Load =====")
        model = keras.models.load_model(database_path + "/animal_cnn.h5")

    print("\n===== Model Summary =====")
    print(model.summary())

    print("\n===== Model Evaluation =====")
    model_eval(model, X_test, Y_test)


def model_train(X_train, Y_train, X_test, Y_test):
    model = Sequential()

    model.add(Conv2D(32, (3, 3,), padding="same",
                     input_shape=X_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3,), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())  # Flat処理、一列にする

    model.add(Dense(512))  # 全結合層
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 最適化処理
    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )  # 損失関数

    print(model.summary())
    # hist = model.fit(X_train, Y_train, batch_size=32, epochs=epoch_num, validation_split=0.2)
    hist = model.fit(
        X_train,
        Y_train,
        batch_size=32,
        epochs=epoch_num,
        validation_data=(X_test, Y_test),
    )

    # print(hist.history)
    model.save(database_path + "/animal_cnn.h5")

    hist_file = open(database_path + "/history.pkl", "wb")
    pickle.dump(hist.history, hist_file)
    hist_file.close()

    return model


def model_eval(model, X, Y):
    scores = model.evaluate(X, Y, verbose=1)
    print("Test Loss: ", scores[0])
    print("Test Accuracy: ", scores[1])

    hist_file = open(database_path + "/history.pkl", "rb")
    hist_data = pickle.load(hist_file)
    hist_file.close()

    hist_visualize(hist_data)

    return


def hist_visualize(hist_data):
    print(hist_data)

    acc = hist_data['accuracy']
    loss = hist_data['loss']
    val_acc = hist_data['val_accuracy']
    val_loss = hist_data['val_loss']

    epochs = range(len(acc))

    fig = plt.figure(figsize=(7, 4))

    # Accuracy
    ax_acc = fig.add_subplot(1, 2, 1)
    ax_acc.plot(epochs, acc, label='Training')
    ax_acc.plot(epochs, val_acc, label='Validation')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy Score')
    ax_acc.set_ylim([0, 1])
    ax_acc.set_title('Accuracy')
    ax_acc.legend(loc='best')
    ax_acc.grid()

    # Loss
    ax_loss = fig.add_subplot(1, 2, 2)
    ax_loss.plot(epochs, loss, label='Training')
    ax_loss.plot(epochs, val_loss, label='Validation')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss Score')
    ax_loss.set_ylim([0, 1])
    ax_loss.set_title('Loss')
    ax_loss.legend(loc='best')
    ax_loss.grid()

    # Save and Show
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.15,
                        top=0.9, wspace=0.3, hspace=0.2)
    plt.savefig(database_path + '/train_result.png')
    # plt.show()
    return


if __name__ == "__main__":
    main()
