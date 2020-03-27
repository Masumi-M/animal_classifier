from PIL import Image
import os
import glob
import numpy as np
from sklearn import model_selection

classes = ["dog", "cat"]
num_classes = len(classes)
image_size = 50
input_image_num = 200


def main():
    # 画像の読み込み
    X = []
    Y = []  # 正解ラベル（dog => 0, cat => 1）
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
            X.append(data)
            Y.append(index)

    X = np.array(X)
    Y = np.array(Y)

    # Cross Validation
    print(len(X))
    print(len(Y))

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X, Y
    )  # split in 3:1
    xy = (X_train, X_test, Y_train, Y_test)
    np.save("./database/animal.npy", xy)

    print(len(X_train), len(X_test))
    print(len(Y_train), len(Y_test))


if __name__ == "__main__":
    main()
