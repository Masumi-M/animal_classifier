import download
import generate_data
import animal_cnn
import predict
import os

database_path = "./database"
if not os.path.exists(database_path):
    os.mkdir(database_path)

classes = ["dog", "cat"]
num_classes = len(classes)

for i_class in range(num_classes):
    if not os.path.exists(database_path + "/" + classes[i_class]):
        download.main(classes[i_class])

if not os.path.exists(database_path + "/animal.npy"):
    print("===== Generate Data =====")
    generate_data.main()

animal_cnn.main()

print("=== Main Script Finished ===")
