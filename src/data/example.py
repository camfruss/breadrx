import pandas as pd

from data_in import DataIngestor
from image_parser import ImageParser

def main():
    subreddits = ["Sourdough", "Breadit"]
    di = DataIngestor("./raw/uncompressed", subreddits)
    di.save("./subreddits", "from_scratch.csv")

    # TODO: update image_source.py parse functions

    # TODO: need to match final image to row in ./subreddits/from_scratch.csv
    # then drop all the unnecessary columns
    # then format columns into
    # then run batches of 77: (total // 50) + 1 => 50 calls to API
    # then save response in 4 separate columns, value, under/over/proper/unsure + scale
    # final csvs (drop unnecessary columns)

    ip = ImageParser("./filtered/crumb", "./final", di.df)
    # ip.download_images()
    ip.resize_dir()


if __name__ == "__main__":
    main()

"""
import os
from time import sleep

import pandas as pd
from alive_progress import alive_bar

def main():
    crumb_files = os.listdir("./filtered/crumb")
    no_crumb_files = os.listdir("./filtered/no_crumb")
    unknown_files = os.listdir("./filtered/unknown")

    crumb_file_names = set([os.path.splitext(file)[0] for file in crumb_files if os.path.isfile(os.path.join("./filtered/crumb", file))])
    no_crumb_file_names = set([os.path.splitext(file)[0] for file in no_crumb_files if os.path.isfile(os.path.join("./filtered/no_crumb", file))])
    unknown_file_names = set([os.path.splitext(file)[0] for file in unknown_files if os.path.isfile(os.path.join("./filtered/unknown", file))])

    still_unknown = list(unknown_file_names - no_crumb_file_names - crumb_file_names)

    # df = concat_subreddits(False)
    # df = extract_links(df, True)

    df = pd.read_csv("./subreddits23/extracted_links2.csv")
    df = df[df["post_id"].isin(still_unknown)]

    df = df[~df["links"].str.contains("imgur")]
    print(df, len(df))

    total = df.shape[0]
    with alive_bar(total, force_tty=True) as bar:
        for index, row in df.iterrows():
            filename = f"./images/{row['post_id']}"

            # if os.path.exists(filename):  # If the image is already saved, continue
            #     bar(); continue

            url = row["links"]
            if "imgur" in url:
                parse_imgur(url, filename)
            elif any(x in url for x in ["redd.it", "reddit"]):
                parse_reddit(url, filename)
            else:
                pass  # log error: extracted link didn't parse correctly
            bar(); sleep(1)
"""