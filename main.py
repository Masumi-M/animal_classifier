import download
import generate_data
import animal_cnn
import os

database_path = "./database"
if not os.path.exists(database_path):
    os.mkdir(database_path)

if not os.path.exists(database_path + "/cat"):
    download.main("cat")

if not os.path.exists(database_path + "/dog"):
    download.main("dog")

if not os.path.exists(database_path + "/turtle"):
    download.main("turtle")

if not os.path.exists(database_path + "animal.npy"):
    print("===== Generate Data =====")
    generate_data.main()

animal_cnn.main()

print("=== Main Script Finished ===")
