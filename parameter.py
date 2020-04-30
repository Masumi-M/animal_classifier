# Parameter List
import keras

classes = ["dog", "crow"]
num_classes = len(classes)
image_size = 256
input_image_num = 320
val_data_num = 80
# input_image_num = 6
# val_data_num = 3
epoch_num = 50
kernel_size = 5
lay1_width = 32
lay2_width = 64
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 最適化処理
# opt = "adam"

database_path = "./database/epoch_" + \
    str(epoch_num) + "_img" + str(image_size) + "_kernel" + str(kernel_size)
