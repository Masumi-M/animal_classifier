from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.utils import np_utils
import keras
import sys
import numpy as np
from PIL import Image

classes = ["dog", "cat"]
num_classes = len(classes)
image_size = 256
input_image_num = 300
epoch_num = 50
kernel_size = 5
lay1_width = 32
lay2_width = 64
database_path = "./database/epoch_" + str(epoch_num) + "_img" + str(image_size) + "_kernel" + str(kernel_size) 

def build_model():
    model = Sequential()

    model.add(Conv2D(lay1_width, (kernel_size, kernel_size,), padding="same",
                     input_shape=(image_size, image_size, 3)))
    model.add(Activation("relu"))
    model.add(Conv2D(lay1_width, (kernel_size, kernel_size)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(lay2_width, (kernel_size, kernel_size,), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(lay2_width, (kernel_size, kernel_size)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())  # Flat処理、一列にする

    model.add(Dense(512))  # 全結合層
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 最適化処理
    opt = "adam"
    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )  # 損失関数

    model = keras.models.load_model(database_path + "/animal_cnn.h5")
    return model


def main():
    image = Image.open(sys.argv[1])
    image = image.convert('RGB')
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    X = []
    X.append(data)
    X = np.array(X)
    model = build_model()

    result = model.predict([X])[0]
    print(result)
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    print('\nPredict Result: ' +
          classes[predicted] + " (" + str(percentage) + "%)")
    for i_class in range(num_classes):
        print(classes[i_class] +
              "\t(" + str(int(result[i_class] * 100)) + " %)")
    return


if __name__ == '__main__':
    main()
