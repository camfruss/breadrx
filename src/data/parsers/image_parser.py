from abc import ABC, abstractmethod
from collections import defaultdict
import os
import pandas as pd
import urllib.request


class ImageParser(ABC):

    def __init__(self, *, df: pd.DataFrame, filepath: str, path: str):
        """
        - df: dataframe containing image links
        - filepath: if no df is provided -> path to csv file containing image links
        - path: directory to save images
        """
        if df is None:
            self.df = pd.read_csv(filepath)
        else:
            self.df = df
        self.df = self.df.filter(["post_id", "image_link"])  # drop unnecessary columns

        self.endpoints = defaultdict(str)

        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)

    @abstractmethod
    def _parse_row(self, link):
        pass

    def parse(self):
        """ Returns dict of id:endpoint key value pairs """
        for _, row in self.df.iterrows():
            endpoint = self._parse_row(row["image_link"])
            post_id = row["post_id"]
            if endpoint:
                self.endpoints[post_id] = endpoint

    def download(self):
        for post_id, endpoint in self.endpoints.items():
            filepath = os.path.join(self.path, f"{post_id}.jpg")
            urllib.request.urlretrieve(endpoint, filepath)
