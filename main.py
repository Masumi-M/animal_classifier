import download
import generate_data
import animal_cnn
import os

if not os.path.exists('cat'):
    download.main('cat')

if not os.path.exists('dog'):
    download.main('dog')

if not os.path.exists('turtle'):
    download.main('turtle')

if not os.path.exists('animal.npy'):
    generate_data.main()

animal_cnn.main()

print("=== Main Script Finished ===")
