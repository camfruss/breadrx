from helper_functions import filter_links
from image_parser import ImageParser

import mimetypes
import os
import re
import requests


class ImgurParser(ImageParser):

    def __init__(self, df, *, filepath: str, path: str):
        super().__init__(df=df, filepath=filepath, path=path)

        patterns = ["imgur"]
        self.df = filter_links(self.df, patterns)

        self.headers = {
            "user-agent": "curl/8.8.0",
            "accept": "*/*",
            'Authorization': f"Client-ID {os.getenv('IMGUR_CLIENT_ID')}"
        }

    @staticmethod
    def _get_image_id(response):
        """ Given an IMGUR API response, finds the first image_id """
        if response.status_code != 200:
            return None

        try:
            data = response.json()
            api_response = data.get("data", None)
            if api_response is None:
                return

            if type(api_response) is list:
                ext = mimetypes.guess_extension(api_response[0]["type"])
                return api_response[0]["link"], ext
            else:
                ext = mimetypes.guess_extension(api_response["type"])
                return api_response["link"], ext
        except Exception:
            pass

    def _parse_row(self, link):

        pattern = r"\/([^/\.]+)(?=[/\.]|$)"  # identifies the image_hash in an imgur link
        match = re.findall(pattern, link)[-1]
        if match is None:
            return

        API_ENDPOINTS = {
            "album": f"https://api.imgur.com/3/album/{match}/images",
            "album_image": f"https://api.imgur.com/3/album/{match}/image/",  # + image_id
            "image": f"https://api.imgur.com/3/image/{match}"
        }

        try:
            if ("/a/" in link) or ("gallery" in link):
                response = requests.get(API_ENDPOINTS["album"], headers=self.headers)
            else:
                response = requests.get(API_ENDPOINTS["image"], headers=self.headers)
            endpoint, ext = self._get_image_id(response)
        except Exception as e:
            return

        return endpoint, ext
