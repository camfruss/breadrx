from dotenv import load_dotenv
import json
from openai import OpenAI
import os
import pandas as pd
import tiktoken


df = pd.read_csv("./subreddits/from_scratch.csv")
post_ids = set([f.split(".")[0] for f in os.listdir("./final")])
df = df[df["post_id"].isin(post_ids)]

# Create labels
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


def calculate_cost():  # Calculates estimated cost of generating all labels
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


def main():
    for idx, row in df[:1].iterrows():
        print(row)
        data = {"title": row["title"], "comments": row["body"]}
        # result = get_label(json.dumps(data))
        # response = json.loads(result)
        # print(type(response), response)
        {
            ""
        }


if __name__ == "__main__":
    main()
