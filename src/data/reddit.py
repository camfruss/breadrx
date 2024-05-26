"""
Refactor so that have Imgur and Reddit class
Use the Reddit API?

separate out into data manipulation file and image downloading file

clean
"""


import logging
import mimetypes
import os
import re
import urllib.error
import urllib.request
from time import sleep

# from openai import OpenAI
import pandas as pd
import requests
from alive_progress import alive_bar
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError


def configure():
    load_dotenv()

    opener = urllib.request.build_opener()
    opener.addheaders = [
        ("user-agent", "curl/8.8.1"),
        ("accept", "*/*")
    ]
    urllib.request.install_opener(opener)

    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)


def parse_reddit(url, filename):
    """

    :param filename:
    :param url: the url of the gallery
    :return: The first image in the gallery if present. Otherwise, None.
    """
    image_endpoint = None
    if "i.redd.it" in url:  #
        image_endpoint = url

    if "gallery" in url:
        response = urllib.request.urlopen(url)
        html = response.read()
        soup = BeautifulSoup(html, "html.parser")
        im_sources = [
            im["src"]
            for im in soup.find_all("img")
            if ".jpg" in im.get("src", "")
        ]
        if not im_sources:
            logging.debug("No image sources"); return
        image_endpoint = im_sources[0]

    if image_endpoint is None:
        logging.debug("Endpoint None"); return

    path = f"{filename}.{url.split('.')[-1]}"
    urllib.request.urlretrieve(image_endpoint, path)
    if not path.endswith(".jpg"):
        png_to_jpg(path)


def png_to_jpg(filename):
    """
    Where to save?
    :param filename:
    :return:
    """
    try:
        im = Image.open(filename)
        rgb_im = im.convert("RGB")
        if filename.endswith("png"):
            jpg_filename = filename.replace(".png", ".jpg")
            rgb_im.save(jpg_filename)
            os.remove(filename)
    except UnidentifiedImageError as uie:
        logging.debug("Unsupported file type")
        pass


def get_image_id(response):
    """

    :param response:
    :return:
    """
    if response.status_code != 200:
        logging.debug(f"Request failed with status code {response.status_code}")
        return None

    try:
        data = response.json()
        api_response = data.get("data", None)

        if not api_response:  # api_response is None
            return

        if type(api_response) is list:
            ext = mimetypes.guess_extension(api_response[0]["type"])
            return api_response[0]["link"], ext
        else:
            ext = mimetypes.guess_extension(api_response["type"])
            return api_response["link"], ext
    except TypeError:
        logging.debug("")
    except ValueError:
        logging.debug("Response content is not valid JSON")
    except KeyError:
        logging.debug("No image content")
    except Exception as e:
        logging.debug(f"Unknown error, {e.__cause__}")

    return None


def parse_imgur(url, filename):
    pattern = r"\/([^/\.]+)(?=[/\.]|$)"  # identifies the image_hash in an imgur link
    match = re.findall(pattern, url)[-1]
    if match is None:
        return

    API_ENDPOINTS = {
        "album": f"https://api.imgur.com/3/album/{match}/images",
        "album_image": f"https://api.imgur.com/3/album/{match}/image/",  # + image_id
        "image": f"https://api.imgur.com/3/image/{match}"
    }

    opener = urllib.request.build_opener()
    opener.addheaders = [
        ("user-agent", "curl/8.8.1"),
        ("accept", "*/*"),
        ("Authorization", f"Client-ID {os.getenv('IMGUR_CLIENT_ID')}")
    ]
    urllib.request.install_opener(opener)

    headers = {
        "user-agent": "curl/8.8.0",
        "accept": "*/*",
        'Authorization': f"Client-ID {os.getenv('IMGUR_CLIENT_ID')}"  # Replace with your actual header key and value
    }

    try:
        if ("/a/" in url) or ("gallery" in url):
            response = requests.get(API_ENDPOINTS["album"], headers=headers)
        else:
            response = requests.get(API_ENDPOINTS["image"], headers=headers)
    except urllib.error.HTTPError:
        logging.debug("urllib.error.HTTPerror imgur "); return

    try:
        endpoint, ext = get_image_id(response)
    except Exception as e:
        logging.debug("get_image_id error"); return

    path = f"{filename}{ext}"
    urllib.request.urlretrieve(endpoint, path)
    if not ext.endswith(".jpg"):
        png_to_jpg(path)




def main():
    configure()

    crumb_files = os.listdir("./filtered/crumb")
    no_crumb_files = os.listdir("./filtered/no_crumb")
    unknown_files = os.listdir("./filtered/unknown")

    crumb_file_names = set([os.path.splitext(file)[0] for file in crumb_files if os.path.isfile(os.path.join("./filtered/crumb", file))])
    no_crumb_file_names = set([os.path.splitext(file)[0] for file in no_crumb_files if os.path.isfile(os.path.join("./filtered/no_crumb", file))])
    unknown_file_names = set([os.path.splitext(file)[0] for file in unknown_files if os.path.isfile(os.path.join("./filtered/unknown", file))])

    still_unknown = list(unknown_file_names - no_crumb_file_names - crumb_file_names)

    # df = concat_subreddits(False)
    # df = extract_links(df, True)

    df = pd.read_csv("./subreddits23/extracted_links2.csv")
    df = df[df["post_id"].isin(still_unknown)]

    df = df[~df["links"].str.contains("imgur")]
    print(df, len(df))

    total = df.shape[0]
    with alive_bar(total, force_tty=True) as bar:
        for index, row in df.iterrows():
            filename = f"./images/{row['post_id']}"

            # if os.path.exists(filename):  # If the image is already saved, continue
            #     bar(); continue

            url = row["links"]
            if "imgur" in url:
                parse_imgur(url, filename)
            elif any(x in url for x in ["redd.it", "reddit"]):
                parse_reddit(url, filename)
            else:
                pass  # log error: extracted link didn't parse correctly
            bar(); sleep(1)


if __name__ == "__main__":
    main()
