from PIL import Image
import os
import glob
import numpy as np
from sklearn import cross_validation


def main():
    classes = ["dog", "cat"]
    num_classes = len(classes)
    image_size = 50
    input_image_num = 3

    # 画像の読み込み
    X = []
    Y = []
    for index, animal_class in enumerate(classes):
        photos_dir = "./" + animal_class
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


if __name__ == '__main__':
    main()
