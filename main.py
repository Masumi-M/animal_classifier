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

if not os.path.exists(database_path_current + "/animal.npy"):
    print("===== Generate Data =====")
    generate_data.main()

animal_cnn.main()

print("=== Main Script Finished ===")
