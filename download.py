from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os
import time
import sys

path_api_id = './api_id.txt'
path_api_secret = './api_secret.txt'

f = open(path_api_id)
api_id = f.read()
f.close()

f = open(path_api_secret)
api_secret = f.read()
f.close()

print(api_id)
print(api_secret)

wait_time = 1

animalname = sys.argv[1]
savedir = "./" + animalname

flickr = FlickrAPI(api_id, api_secret, format='parsed-json')

result = flickr.photos.search(
    text=animalname,
    per_page=400,
    media='photos',
    sort='relevance',
    safe_search=1,
    extras='url_q, licence'
)

photos = result['photos']
pprint(photos)
