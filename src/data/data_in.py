"""
parse.py parses the data provided by the Reddit torrent provided on Acamdemic Torrents.

Namely:
1) comments for each post are grouped and joined with the submissions
2) image hyperlinks are extracted from the post's body
3) post hashes are extracted from the permalink
"""
import logging
import os

import pandas as pd


class DataIngestor:

    def __init__(self, path: str, subreddits: [str],):
        """


        Parameters
        ----------
        path: the folder containing the uncompressed submissions and comments csv file
            requires no trailing forward/backslash
            eg: "./data/uncompressed"
        subreddits: the names of the subreddits being ingested
            ensure capitalization is correct
        """
        logging.basicConfig(filename="data_ingestor.log", encoding="utf-8", level=logging.INFO)

        self.path = path
        self.subreddits = subreddits
        self.df = pd.DataFrame()
        self.add_sources(subreddits)

    def add_sources(self, sources: [str]):
        """


        """
        for source in sources:
            self.add_source(source)

    def add_source(self, source: str):
        """

        Parameters
        ----------
        source:
        """
        submissions_df = self._parse_submissions(f"{self.path}/{source}_submissions.csv")
        comments_df = self._parse_comments(f"{self.path}/{source}_comments.csv")

        tmp_df = pd.merge(submissions_df, comments_df, on="post_id")
        tmp_df = self._extract_links(tmp_df)

        self.df = pd.concat([self.df, tmp_df], ignore_index=True)
        logging.info(f"Successfully added {source} to dataframe")

    def save(self, dir_: str, name: str):
        """

        Parameters
        ----------
        dir_: the path to the folder
        name: the name of the csv file
        """
        if not os.path.exists(dir_):
            os.mkdir(dir_)

        path = os.path.join(dir_, name)
        self.df.to_csv(path, index=False)

    @staticmethod
    def _parse_submissions(filepath: str):
        """

        Parameters
        ---------
        filepath:

        """
        df = pd.read_csv(filepath)
        df.columns = ["upvotes", "date", "title", "OP", "post_link", "selftext"]
        df["post_id"] = df["post_link"].str.extract(r"/comments/(\w+)")[0]
        return df

    @staticmethod
    def _parse_comments(filepath: str):
        """


        """
        df = pd.read_csv(filepath)
        df.columns = ["upvotes", "date", "OP", "comment_link", "body"]
        df["post_id"] = df["comment_link"].str.extract(r"/comments/(\w+)")[0]

        merged_df = df.groupby("post_id").agg({"body": list}).reset_index()
        return merged_df

    @staticmethod
    def _extract_links(df: pd.DataFrame):
        """

        Parameters
        ----------
        df:
        """
        image_pattern = r"https?://[^\s]+(?:\.(?:jpg|png|jpeg))"
        site_pattern = r"https?://[^\s]*(?:imgur|gallery)[^\s]*"
        combined_pattern = f"({image_pattern}|{site_pattern})"

        extracted_links = df["selftext"].str.extract(combined_pattern)

        df_out = df.merge(extracted_links, left_index=True, right_index=True, how="left")
        df_out.rename(columns={0: "image_link"}, inplace=True)
        df_out.dropna(subset="image_link", inplace=True)

        clean_pattern = r"([a-zA-Z0-9./:]+)"
        df_out["image_link"] = df_out["image_link"].str.extract(clean_pattern, expand=False)

        return df_out
