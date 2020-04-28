from PIL import Image
import os
import glob
import numpy as np
from sklearn import model_selection

classes = ["dog", "cat"]
num_classes = len(classes)
image_size = 150
input_image_num = 300
val_data_num = 100


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

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # Cross Validation
    # X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    #     X, Y
    # )  # split in 3:1

    xy = (X_train, X_test, Y_train, Y_test)
    np.save("./database/animal.npy", xy)

    print(len(X_train), len(X_test))
    print(len(Y_train), len(Y_test))


if __name__ == "__main__":
    main()
