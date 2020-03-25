import download
import os

if not os.path.exists('cat'):
    download.main('cat')

if not os.path.exists('dog'):
    download.main('dog')
