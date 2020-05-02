# Parameter List
import keras

version = "5_0"
classes = ["dog", "crow"]
num_classes = len(classes)
image_size = 256
input_image_num = 320
val_data_num = 80
cross_num = 4
epoch_num = 50
kernel_size = 3
lay1_width = 64
lay2_width = 32
lay3_width = 16

opt_name = "rmsprop"
# opt_name = "adam"

if opt_name == "rmsprop":
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 最適化処理
elif opt_name == "adam":
    opt = "adam"

database_path = "./database/v" + version + "_epoch_" + \
    str(epoch_num) + "_img" + str(image_size) + \
    "_kernel" + str(kernel_size) + "_" + str(opt_name)
