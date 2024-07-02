import shutil

from alive_progress import alive_bar
from dotenv import load_dotenv
import json
import numpy as np
from openai import OpenAI
import os
import pandas as pd
import tiktoken
from time import sleep

# Filter rows with valid image in dataset
hf_path = "./huggingface"
images_path = os.path.join(hf_path, "images")

df = pd.read_csv("./raw/uncompressed/torrent_out.csv")
post_ids = set([f.split(".")[0] for f in os.listdir(images_path)])
df = df[df["post_id"].isin(post_ids)]

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

system_prompt = \
    """ You are looking at comments of posts from people asking about their bread. You will be provided the title of 
    each post and comments from other users as a json object in the following format:

    {
        title: string, // title of the post
        comments: string[]  // array, where each element is a separate comment from a user
    } 

    I want you to determine whether the person's bread is under-proofed, over-proofed, perfectly proofed, or if the
    comments are inconclusive. For each of these 4 categories, I want you to determine the probability it fits into
    each of the 4 categories. For example, if the comments are clear the bread is over-proofed, the "over-proofed" field
    should have a value close to 1. Use 2 significant digits. The json format of your response should be as follows:

    {
        over: float
        under: float
        perfect: float
        inconclusive: float
    } 
    """


def calculate_cost():
    """ Calculates estimated cost of generating all labels """
    API_COSTS = {
        "gpt-4o": 5.00,
        "gpt-3.5-turbo": 0.50
    }
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    token_count = 0
    for _, row in df.iterrows():
        text = row["title"] + row["body"]
        token_count += len(enc.encode(text))

    print(f"Token Count for {len(df):,} rows: {token_count:,}")
    print("Estimated cost based on model:")
    for k, v in API_COSTS.items():
        print(f"\t - {k}: ${token_count * v / 1_000_000:,.2f}")


def get_label(description):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        response_format={
            "type": "json_object"
        },
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": description
            }
        ]
    )
    return completion.choices[0].message.content


def move_images(split_df, split_name):
    path = os.path.join(hf_path, split_name)
    os.makedirs(path, exist_ok=True)
    for _, row in split_df.iterrows():
        src = os.path.join(images_path, row["image"])
        dst = os.path.join(path, row["image"])
        shutil.copy(src, dst)


def main():
    columns = ["image", "upvotes", "under_proof", "over_proof", "perfect_proof", "unsure_proof"]
    df_out = pd.DataFrame(columns=columns)

    with alive_bar(len(df)) as bar:
        for idx, row in df.iterrows():
            data = {"title": row["title"], "comments": row["body"]}
            result = get_label(json.dumps(data))
            response = json.loads(result)

            if response.get("under") is not None:
                df_out = df_out._append({
                    "image": f"{row['post_id']}.jpg",
                    "upvotes": row["upvotes"],
                    "under_proof": response.get("under"),
                    "over_proof": response.get("over"),
                    "perfect_proof": response.get("perfect"),
                    "unsure_proof": response.get("inconclusive")
                }, ignore_index=True)
            bar()
            sleep(0.50)

    # create splits
    train, validate, test = np.split(
        df_out.sample(frac=1, random_state=42),
        [
            int(0.80 * len(df_out)),
            int(0.90 * len(df_out))
        ]
    )

    df_out.loc[train.index, "image"] = "train/" + train["image"]
    df_out.loc[validate.index, "image"] = "validate/" + validate["image"]
    df_out.loc[test.index, "image"] = "test/" + test["image"]

    move_images(train, "train")
    move_images(validate, "validate")
    move_images(test, "test")

    df_out.to_csv("./huggingface/metadata.csv", index=False)


if __name__ == "__main__":
    main()
