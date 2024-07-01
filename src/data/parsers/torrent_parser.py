"""
parse.py parses the data provided by the Reddit torrent provided on Acamdemic Torrents.

Namely:
1) comments for each post are grouped and joined with the submissions
2) image hyperlinks are extracted from the post's body
3) post hashes are extracted from the permalink
"""
import argparse
import os
import pandas as pd
import regex as re


SUBMISSION_COLUMNS = ["upvotes", "date", "title", "OP", "post_link", "selftext"]
COMMENT_COLUMNS = ["upvotes", "date", "OP", "comment_link", "body"]


def extract_links(df: pd.DataFrame):
    """
    Finds reddit and imgur links inside a block of text and extracts them.

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


class Subreddit:

    def __init__(self, path, subreddit):
        self.comments_path = os.path.join(path, f"{subreddit}_comments.csv")
        self.submissions_path = os.path.join(path, f"{subreddit}_submissions.csv")

        self.submissions = self.parse_submissions()
        self.comments = self.parse_comments()
        self.subreddit = self.parse()

    def parse(self):
        """ Adds {subreddit}_comments|submissions to the objects dataframe """
        tmp_df = pd.merge(self.submissions, self.comments, on="post_id")
        return extract_links(tmp_df)

    def parse_submissions(self):
        """ Adds a post_id column to the uncompressed submissions file """
        df = pd.read_csv(self.submissions_path, encoding="utf-8")
        df.columns = SUBMISSION_COLUMNS

        df["post_id"] = df["post_link"].str.extract(r"/comments/(\w+)")[0]

        return df

    def parse_comments(self):
        """ Aggregates comments relating to a single submission into a single row """

        df = pd.read_csv(self.comments_path, encoding="utf-8")
        df.columns = COMMENT_COLUMNS

        # Isolate the post_id from each of the comments
        df["post_id"] = df["comment_link"].str.extract(r"/comments/(\w+)")[0]
        # combine all the comments with the same post_id
        merged_df = df.groupby("post_id").agg({"body": list}).reset_index()

        return merged_df


class TorrentParser:

    def __init__(self, path: str):
        """
        path: the folder containing the uncompressed submissions and comments csv file
            requires both submission and comments file are formatted as "{subreddit}_submissions|comments.csv"
        """
        self.path = path.removesuffix("/")
        self.dfs = {}
        self.parse()

    def parse(self):
        """ Constructs Dataframe object consisting of all submissions and comments. """
        subreddits = set()
        for file in os.listdir(self.path):
            subreddit = re.search(r"([A-Za-z]+)(?=_(comments|submissions))", file)
            if subreddit:
                subreddits.add(subreddit.group(0))

        for k in subreddits:
            self.dfs[k] = Subreddit(self.path, k)

    def all(self):
        """ Concatenates all subreddits into a single df """
        return pd.concat([r.subreddit for r in self.dfs.values()])

    def save(self, filename="torrent_out.csv"):
        """
        Saves the df to the file {torrent_out.csv} in the same directory as the submission & comment csv files

        filename: saves the file as {filename}
            requires name end in ".csv"
        """
        path = os.path.join(self.path, filename)
        self.all().to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Aggregate academic torrent Reddit CSV files")
    parser.add_argument(
        "-d", "--directory",
        type=str,
        required=True,
        help="The path to the directory containing files to be parsed"
    )
    args = parser.parse_args()

    tp = TorrentParser(args.directory)
    tp.save()


if __name__ == "__main__":
    main()
