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
                # X_train.append(data)
                # Y_train.append(index)

                # Rotation
                for angle in range(-20, 20, 10):
                    img_r = image.rotate(angle)
                    data = np.asarray(img_r)
                    X_train.append(data)
                    Y_train.append(index)

                # Transpose
                    img_t = image.transpose(Image.FLIP_LEFT_RIGHT)
                    data = np.asarray(img_t)
                    X_train.append(data)
                    Y_train.append(index)

    a_array = np.array([1, 2, 3, 4, 5, 6])
    b_array = np.array([2, 3, 4, 5, 6, 7])
    rand_array = list(range(len(a_array)))
    np.random.shuffle(rand_array)
    print(a_array)
    print(b_array)
    print(rand_array)
    custom_randomize(rand_array, a_array)
    custom_randomize(rand_array, b_array)
    print(a_array)
    print(b_array)

    print(X_test)
    print(Y_test)

    rand_array_test = list(range(len(X_test)))
    np.random.shuffle(rand_array_test)
    custom_randomize(rand_array, X_test)
    custom_randomize(rand_array, Y_test)

    rand_array_train = list(range(len(X_train)))
    np.random.shuffle(rand_array_train)
    custom_randomize(rand_array, X_train)
    custom_randomize(rand_array, Y_train)

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
