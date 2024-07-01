import pandas as pd

from .helper_functions import filter_links
from .image_parser import ImageParser

from bs4 import BeautifulSoup
from collections import defaultdict
import urllib.error
import urllib.request


class RedditParser(ImageParser):

    def __init__(self, df: pd.DataFrame = None, *, filepath: str = None, path: str):
        """
        - df: dataframe containing only Reddit image links
        """
        super().__init__(df=df, filepath=filepath, path=path)

        patterns = ["i.redd.it", "reddit.com"]
        self.df = filter_links(self.df, patterns)

    def _parse_row(self, link):
        if "i.redd.it" in link and link.endswith(".jpg"):
            return link

        try:
            if "gallery" in link:
                response = urllib.request.urlopen(link)
                html = response.read()
                soup = BeautifulSoup(html, "html.parser")
                im_sources = [
                    im["src"]
                    for im in soup.find_all("img")
                    if ".jpg" in im.get("src", "")
                ]
                return im_sources[0]
        except Exception:
            pass
        return None
