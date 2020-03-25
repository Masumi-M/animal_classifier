from flickrapi import FlickrAPI
from urllib.request import urlretrieve
import certifi
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
if not os.path.exists(savedir):
    os.mkdir(savedir)

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

# pprint(photos)

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + 'jpg'
    if os.path.exists(filepath):
        continue
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)
