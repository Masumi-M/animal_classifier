from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.utils import np_utils
import keras
import sys
import numpy as np
from PIL import Image
import parameter

classes = parameter.classes
num_classes = parameter.num_classes
image_size = parameter.image_size
input_image_num = parameter.input_image_num
epoch_num = parameter.epoch_num
kernel_size = parameter.kernel_size
lay1_width = parameter.lay1_width
lay2_width = parameter.lay2_width
opt = parameter.opt
lay3_width = parameter.lay3_width
database_path = parameter.database_path


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

    model.add(Conv2D(lay3_width, (kernel_size, kernel_size,), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(lay3_width, (kernel_size, kernel_size)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())  # Flat処理、一列にする

    model.add(Dense(512))  # 全結合層
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

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
