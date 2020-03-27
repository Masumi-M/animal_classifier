from flickrapi import FlickrAPI
from urllib.request import urlretrieve
import certifi
from pprint import pprint
import os
import time
import sys
from tqdm import tqdm


def main(animalname):
    pprint("Download [" + animalname + "]")
    path_api_id = "./api_id.txt"
    path_api_secret = "./api_secret.txt"

    f = open(path_api_id)
    api_id = f.read()
    f.close()

    f = open(path_api_secret)
    api_secret = f.read()
    f.close()

    wait_time = 1

    savedir = "./" + animalname
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    photo_num = 400

    flickr = FlickrAPI(api_id, api_secret, format="parsed-json")

    result = flickr.photos.search(
        text=animalname,
        per_page=photo_num,
        media="photos",
        sort="relevance",
        safe_search=1,
        extras="url_q, licence",
    )

    photos = result["photos"]

    pprint("  Fetch Completed.")
    pprint("  Start Download.")

    start_time = time.time()
    download_loader = tqdm(total=photo_num)
    for i, photo in enumerate(photos["photo"]):
        download_loader.update(1)
        url_q = photo["url_q"]
        filepath = savedir + "/" + photo["id"] + ".jpg"
        if os.path.exists(filepath):
            continue
        urlretrieve(url_q, filepath)
        time.sleep(wait_time)

    download_loader.close()
    pprint("Calc_time: [" + string(time.time() - start_time) + "]")


if __name__ == "__main__":
    main(sys.argv[1])
