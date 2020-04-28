import download
import generate_data
import animal_cnn
import predict
import notify_line
import os
import parameter.py

database_path = "./database"
if not os.path.exists(database_path):
    os.mkdir(database_path)

classes = parameter.classes
num_classes = parameter.num_classes

for i_class in range(num_classes):
    if not os.path.exists(database_path + "/" + classes[i_class]):
        download.main(classes[i_class])

if not os.path.exists(database_path + "/animal.npy"):
    print("===== Generate Data =====")
    generate_data.main()

animal_cnn.main()
notify_line

print("=== Main Script Finished ===")
