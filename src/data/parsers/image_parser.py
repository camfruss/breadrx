from abc import ABC, abstractmethod
from collections import defaultdict
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
from time import sleep
import urllib.error
import urllib.request

class ImageParser(ABC):

    def __init__(self, *, df: pd.DataFrame, filepath: str, path: str):
        """
        - df: dataframe containing image links
        - filepath: if no df is provided -> path to csv file containing image links
        - path: directory to save images
        """
        env_file = find_dotenv(".env")
        load_dotenv(env_file)

        opener = urllib.request.build_opener()
        opener.addheaders = [
            ("user-agent", "curl/8.8.1"),
            ("accept", "*/*")
        ]
        urllib.request.install_opener(opener)

        if df is None:
            self.df = pd.read_csv(filepath)
        else:
            self.df = df
        self.df = self.df.filter(["post_id", "image_link"])  # drop unnecessary columns

        self.endpoints = defaultdict(str)

        self.path = path
        os.makedirs(path, exist_ok=True)

    @abstractmethod
    def _parse_row(self, link):
        pass

    def parse(self, limit: int = None):
        """ Returns dict of id:endpoint key value pairs """
        for _, row in self.df[:limit].iterrows():
            endpoint = self._parse_row(row["image_link"])
            post_id = row["post_id"]
            if endpoint:
                self.endpoints[post_id] = endpoint

    def download(self, politeness: int = 1):
        for post_id, endpoint in self.endpoints.items():
            filepath = os.path.join(self.path, f"{post_id}.jpg")
            try:
                urllib.request.urlretrieve(endpoint, filepath)
            except Exception as e:
                print(f"Exception encountered: {e}\nendpoint: {endpoint}\nfilepath: {filepath}")
            sleep(politeness)
