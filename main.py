import download
import generate_data
import animal_cnn
import calc_mean
import notify_line
# import predict
import os
import time
import parameter
import shutil

start_time = time.time()
database_path = "./database"
if not os.path.exists(database_path):
    os.mkdir(database_path)

# Set Parameter
classes = parameter.classes
num_classes = parameter.num_classes

# Creaete Save Dir
database_path_current = parameter.database_path
if not os.path.exists(database_path_current):
    os.mkdir(database_path_current)

# Copy Parameter File
shutil.copyfile("./parameter.py", database_path_current + "/parameter.py")

# Download Image
for i_class in range(num_classes):
    if not os.path.exists(database_path + "/" + classes[i_class]):
        download.main(classes[i_class])

# 4-fold cross validation (1:3 x 4)
for i_cross_num in range(1, 1 + parameter.cross_num):
    print(i_cross_num)
    database_path_current_cross = database_path_current + \
        "/cross" + str(i_cross_num)
    if not os.path.exists(database_path_current_cross):
        os.mkdir(database_path_current_cross)

    if not os.path.exists(database_path_current_cross + "/animal_.npy"):
        print("===== Generate Data =====")
        generate_data.main(i_cross_num)

    animal_cnn.main(i_cross_num)

calc_mean.main()

calc_time = time.time() - start_time
notify_line.main(calc_time)

print("=== Main Script Finished ===")
