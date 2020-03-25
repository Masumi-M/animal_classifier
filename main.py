import download
import generate_data
import os

if not os.path.exists('cat'):
    download.main('cat')

if not os.path.exists('dog'):
    download.main('dog')

if not os.path.exists('animal.npy'):
    generate_data.main()

print("=== Main Script Finished ===")
