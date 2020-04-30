import download
import generate_data
import animal_cnn
import predict
import os
import parameter

database_path = "./database"
if not os.path.exists(database_path):
    os.mkdir(database_path)

classes = parameter.classes
num_classes = parameter.num_classes

for i_class in range(num_classes):
    if not os.path.exists(database_path + "/" + classes[i_class]):
        download.main(classes[i_class])

database_path_current = parameter.database_path
if not os.path.exists(database_path_current):
    os.mkdir(database_path_current)

# 4-fold cross validation (1:3 x 4)
for i_cross_num in range(parameter.cross_num):
    database_path_current_cross = database_path_current + \
        "_cross" + str(i_cross_num)
    if not os.path.exists(database_path_current_cross):
        os.mkdir(database_path_current_cross)

    if not os.path.exists(database_path_current_cross + "/animal_.npy"):
        print("===== Generate Data =====")
        generate_data.main(i_cross_num)

    animal_cnn.main(i_cross_num)


print("=== Main Script Finished ===")
