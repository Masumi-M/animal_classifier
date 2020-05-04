# Parameter List
import keras

version = "7_1"
classes = ["dog", "cat"]
num_classes = len(classes)
image_size = 500    # 256 => 500
input_image_num = 320
val_data_num = 80
cross_num = 4
epoch_num = 70
# early_stopping_patient = 10
early_stopping_patient = epoch_num
kernel_size = 3
lay1_width = 16
lay2_width = 32
lay3_width = 64
lay4_width = 128
lay5_width = 256
conn1_width = 512
conn2_width = 256
conn3_width = 128
conn4_width = num_classes
batch_size = 128
opt_name = "rmsprop"
# opt_name = "adam"

if opt_name == "rmsprop":
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 最適化処理
elif opt_name == "adam":
    opt = "adam"

database_path = "./database/v" + version + "_epoch_" + \
    str(epoch_num) + "_img" + str(image_size) + \
    "_kernel" + str(kernel_size) + "_" + str(opt_name)
