from dotenv import load_dotenv
from itertools import product
import os
import praw
import pandas as pd
import re
"""
if gallery or ends in jpeg, 
a tag, href label contains

"preview.redd.it"

generate single csv with all posts and comments /data
extract images in a second step

go for 10,000 
"""

# Create Subreddit Instance
# load_dotenv()
# reddit = praw.Reddit(
#     client_id=os.getenv("CLIENT_ID"),
#     client_secret=os.getenv("CLIENT_SECRET"),
#     password=os.getenv("PASSWORD"),
#     user_agent=os.getenv("USERAGENT"),
#     username=os.getenv("USERNAME")
# )
# subreddits = reddit.subreddit("+".join(["Breadit", "Sourdough"]))
#
# # Construct search query
# prefixes = ["over", "under"]
# separators = ["-", " ", ""]
# suffixes = ["proof", "proofed", "fermented"]
#
# queries = list(product(prefixes, separators, suffixes))
# queries = [f"selftext:\"{''.join(q)}\"" for q in queries]
# query = " OR ".join(queries)

# Only take the comments that contain a query


# Load data

#
# demo = reddit.submission("1bp3esm")
# print(demo.url, demo.title, demo.selftext, demo.permalink, demo.name, demo.id, demo.comments, demo.is_self)
#
# query_result = subreddits.search(query, limit=None)
# for post in query_result:
#     if not post.is_self:
#         data = {
#             "id": post.id,
#             "create_utc": post.created_utc,
#             "permalink": post.permalink,
#             "title": post.title,
#             "body": post.selftext,
#             "url": post.url,
#             "comments": post.comments,
#         }
