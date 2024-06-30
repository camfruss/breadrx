from imgur_parser import ImgurParser
from reddit_parser import RedditParser

import os
import pandas as pd

class ImageParser:

    # name: (pattern, class)
    parsers = {
        "reddit": (["i.redd.it", "reddit.com"], RedditParser),
        "imgur": (["imgur"], ImgurParser)
    }

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

        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)

    def download_images(self):
        """ Downloads images from the link provided in {image_link} using the parser specified in {source_map} """
        for k, (v, cls) in ImageParser.parsers.items():
            df = self.df[self.df["image_link"].str.contians("|".join(v))]
            cls(df, self.path).parse()
