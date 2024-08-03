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

GPT_MODEL = "gpt-4o-mini"

system_prompt = \
    """ You are looking at comments of posts from people asking about their bread. You will be provided the title of 
    each post and comments from other users as a json object in the following format:

    {
        title: string, // title of the post
        comments: string[]  // array, where each element is a separate comment from a user
    } 

    I want you to determine whether the person's bread is under-proofed, over-proofed, perfectly proofed, or if there
    is not enough information in the comments to decide. For example, if the comments are clear the bread is 
    over-proofed, the results field should be "over-proofed." The json format of your response should be as follows:

    {
        result: string
    } 
    """


def calculate_cost(model=GPT_MODEL):
    """ Calculates estimated cost of generating all labels """
    API_COSTS = {
        "gpt-4o": 5.00,
        "gpt-4o-mini": 0.150,
        "gpt-3.5-turbo": 0.50
    }
    enc = tiktoken.encoding_for_model(model)
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
        model=GPT_MODEL,
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
    new_path = os.path.join(hf_path, split_name)
    os.makedirs(new_path, exist_ok=True)
    for _, row in split_df.iterrows():
        src = os.path.join(images_path, row["file_name"])
        dst = os.path.join(new_path, row['file_name'])
        shutil.copy(src, dst)


def main():
    columns = ["file_name", "label"]
    df_out = pd.DataFrame(columns=columns)

    with alive_bar(len(df)) as bar:
        for idx, row in df.iterrows():
            data = {"title": row["title"], "comments": row["body"]}
            result = get_label(json.dumps(data))
            response = json.loads(result)

            if (label := response.get("result")) is not None and "proof" in label:
                df_out = df_out._append({
                    "file_name": f"{row['post_id']}.jpg",
                    "label": label
                }, ignore_index=True)
            bar()
            sleep(0.50)

    # create splits
    lower, upper = int(0.80 * len(df_out)), int(0.90 * len(df_out))
    train = df_out.iloc[:lower, :]
    validate = df_out.iloc[lower:upper, :]
    test = df_out.iloc[upper:, :]

    # reorganize file paths
    move_images(train, "train")
    move_images(validate, "validate")
    move_images(test, "test")

    # update image names with file paths
    df_out.loc[train.index, "file_name"] = "train/" + train["file_name"]
    df_out.loc[validate.index, "file_name"] = "validate/" + validate["file_name"]
    df_out.loc[test.index, "file_name"] = "test/" + test["file_name"]

    # vectorize
    label_map = {
        "under-proofed": 0,
        "over-proofed": 1,
        "perfectly proofed": 2
    }
    df_out["label"] = df_out["label"].map(label_map)
    df_out.to_csv("./huggingface/metadata.csv", index=False)


if __name__ == "__main__":
    main()
