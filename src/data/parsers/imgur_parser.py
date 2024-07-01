from .helper_functions import filter_links
from .image_parser import ImageParser

import mimetypes
import os
import pandas as pd
import re
import requests


class ImgurParser(ImageParser):

    def __init__(self, df: pd.DataFrame = None, *, filepath: str = None, path: str):
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

        data = response.json()
        api_response = data.get("data", None)
        if api_response is None:
            return None

        # depending on response type, returns different json
        if type(api_response) is list:
            ext = mimetypes.guess_extension(api_response[0]["type"])
            result = api_response[0]["link"]
        else:
            ext = mimetypes.guess_extension(api_response["type"])
            result = api_response["link"]

        if ext.endswith(("jpeg", "jpg")):
            return result

        return None

    def _parse_row(self, link):
        pattern = r"\/([^/\.]+)(?=[/\.]|$)"  # identifies the image_hash in an imgur link
        match = re.findall(pattern, link)[-1]
        if match is None:
            return None

        API_ENDPOINTS = {
            "album": f"https://api.imgur.com/3/album/{match}/images",
            "image": f"https://api.imgur.com/3/image/{match}"
        }

        try:
            if ("/a/" in link) or ("gallery" in link):
                response = requests.get(API_ENDPOINTS["album"], headers=self.headers)
            else:
                response = requests.get(API_ENDPOINTS["image"], headers=self.headers)
            endpoint = self._get_image_id(response)
            return endpoint
        except Exception as e:
            pass

        return None

