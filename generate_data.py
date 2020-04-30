from PIL import Image
import os
import glob
import numpy as np
from sklearn import model_selection
import parameter
import numpy as np

classes = parameter.classes
num_classes = parameter.num_classes
image_size = parameter.image_size
input_image_num = parameter.input_image_num
val_data_num = parameter.val_data_num
database_path_current = parameter.database_path


def custom_randomize(rand_array, data):
    temp = np.copy(data[0])
    for i_length in range(len(rand_array)):
        i_rand = rand_array[i_length]
        data[0] = np.copy(data[i_rand])
        data[i_rand] = np.copy(temp)
        temp = np.copy(data[0])

    return data


def main():
    # 画像の読み込み
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []  # 正解ラベル（dog => 0, cat => 1）

    for index, animal_class in enumerate(classes):
        photos_dir = "./database/" + animal_class
        files = glob.glob(photos_dir + "/*.jpg")
        for i, file in enumerate(files):
            if i >= input_image_num:
                break
            image = Image.open(file)
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            data = np.asarray(image)

            if i < val_data_num:
                X_test.append(data)
                Y_test.append(index)
            else:
                X_train.append(data)
                Y_train.append(index)

                # Rotation
                # for angle in range(-20, 20, 10):
                #     img_r = image.rotate(angle)
                #     data = np.asarray(img_r)
                #     X_train.append(data)
                #     Y_train.append(index)

                # Transpose
                img_t = image.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_t)
                X_train.append(data)
                Y_train.append(index)

    rand_array_test = list(range(len(X_test)))
    np.random.shuffle(rand_array_test)
    Y_test_array = np.copy(Y_test)
    custom_randomize(rand_array_test, X_test)
    custom_randomize(rand_array_test, Y_test_array)

    rand_array_train = list(range(len(X_train)))
    np.random.shuffle(rand_array_train)
    Y_train_array = np.copy(Y_train)
    custom_randomize(rand_array_train, X_train)
    custom_randomize(rand_array_train, Y_train_array)

    for i_length in range(len(Y_test_array)):
        temp = Y_test_array[i_length]
        Y_test[i_length] = temp

    for i_length in range(len(Y_train_array)):
        temp = Y_train_array[i_length]
        Y_train[i_length] = temp

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # Cross Validation
    # X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    #     X, Y
    # )  # split in 3:1

    xy = (X_train, X_test, Y_train, Y_test)
    np.save(database_path_current + "/animal.npy", xy)

    print(len(X_train), len(X_test))
    print(len(Y_train), len(Y_test))


if __name__ == "__main__":
    main()
