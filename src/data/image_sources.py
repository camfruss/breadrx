from helper_functions import png_to_jpg

import logging
from dotenv import load_dotenv
import mimetypes
import os
import re
import requests
import urllib.error
import urllib.request

from bs4 import BeautifulSoup

def configure():
    load_dotenv()
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ("user-agent", "curl/8.8.1"),
        ("accept", "*/*")
    ]
    urllib.request.install_opener(opener)
    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

class RedditImages:

    def __init__(self, df, fin, fout):
        """

        Parameters
        ----------
        - df: dataframe consisting of exclusively reddit image links
        - path: the folder where all images are downloaded
            - e.g.: /home/data/images
        """
        self.df = df
        self.fin = fin

    def parse(self):
        """


        """
        urls = None  # TODO

        for url in urls:
            filename = None  # TODO

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
                    logging.debug("No image sources")
                    return
                image_endpoint = im_sources[0]

            if image_endpoint is None:
                logging.debug("Endpoint None")
                return

            path = f"{self.fout}/{filename}.{url.split('.')[-1]}"
            urllib.request.urlretrieve(image_endpoint, path)
            if not path.endswith(".jpg"):
                png_to_jpg(path)


class ImgurImages:

    def __init__(self, df, fin, fout):
        self.df = df
        self.fin = fin
        self.fout = fout

        opener = urllib.request.build_opener()
        opener.addheaders = [
            ("user-agent", "curl/8.8.1"),
            ("accept", "*/*"),
            ("Authorization", f"Client-ID {os.getenv('IMGUR_CLIENT_ID')}")
        ]
        urllib.request.install_opener(opener)

        self.headers = {
            "user-agent": "curl/8.8.0",
            "accept": "*/*",
            'Authorization': f"Client-ID {os.getenv('IMGUR_CLIENT_ID')}"
        }

    @staticmethod
    def _get_image_id(response):
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

    def parse_imgur(self, url, filename):
        """

        :param url:
        :param filename:
        :return:
        """
        pattern = r"\/([^/\.]+)(?=[/\.]|$)"  # identifies the image_hash in an imgur link
        match = re.findall(pattern, url)[-1]
        if match is None:
            return

        API_ENDPOINTS = {
            "album": f"https://api.imgur.com/3/album/{match}/images",
            "album_image": f"https://api.imgur.com/3/album/{match}/image/",  # + image_id
            "image": f"https://api.imgur.com/3/image/{match}"
        }

        try:
            if ("/a/" in url) or ("gallery" in url):
                response = requests.get(API_ENDPOINTS["album"], headers=self.headers)
            else:
                response = requests.get(API_ENDPOINTS["image"], headers=self.headers)
        except urllib.error.HTTPError:
            logging.debug("urllib.error.HTTPerror imgur ")
            return

        try:
            endpoint, ext = self._get_image_id(response)
        except Exception as e:
            logging.debug("get_image_id error")
            return

        path = f"{filename}{ext}"
        urllib.request.urlretrieve(endpoint, path)
        if not ext.endswith(".jpg"):
            png_to_jpg(path)
