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

class RedditParser:

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


