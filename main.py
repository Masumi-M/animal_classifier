import download
import generate_data
import animal_cnn
import os

database_path = "./database"
if not os.path.exists(database_path):
    os.mkdir(database_path)

if not os.path.exists(database_path + "cat"):
    download.main(database_path + "cat")

if not os.path.exists(database_path + "dog"):
    download.main(database_path + "dog")

if not os.path.exists(database_path + "turtle"):
    download.main(database_path + "turtle")

if not os.path.exists(database_path + "animal.npy"):
    generate_data.main()

animal_cnn.main()

print("=== Main Script Finished ===")
