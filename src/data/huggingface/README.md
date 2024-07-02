
## The Bread Proofing Dataset

### Dataset Summary

The Bread Proofing Dataset provides 3,801 images of sliced bread and the probability that each loaf is over-proofed, under-proofed, well-proofed, or 
if no determination can be made 

### Features
All data was collected and parsed from the /r/Breadit and /r/Sourdough subreddits, with raw data originating from 
[Academic Torrents](https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10), as Reddit does not natively provide all available posts. The data spans from the 
beginning of each subreddit's creation to 31 December 2023. 

To download the raw data yourself, follow the instructions provided [here](https://www.reddit.com/r/pushshift/comments/1akrhg3/).
The [script provided](https://github.com/Watchful1/PushshiftDumps/blob/master/scripts/filter_file.py) in the 
above instructions was used to remove comments that made no mention of proofing. For images contained in a gallery, only the
first image was ingested into the dataset for simplicity's sake. All non-crumb pictures were removed manually using 
[Narrative Select](https://narrative.so/select). Of the 12,558 posts that mentioned proofing, 10,027 contained a non-deleted
image, and 3,801 contained a usable crumb image. 

There are torrents available with more up-to-date posts, so more images could be collected in the future, but pursing 
alternative non-Reddit sources would likely yield better results. 

### Labels

OpenAI's gpt-3.5-turbo model was provided the post title, post description, and all relevant post comments to
determine the probability that a bread was under, over, perfectly, or inconclusively proofed. 

The total cost of labelling was only ~$1.00 USD. 

### Image Cleansing 

All images were cropped and scaled to 512x512 such that rectangular images
had their long edge cropped (rather than compressed).

### Dataset Structure

The data is structured with a pre-defined 80/10/10 train/validate/test split. 

```
├── test
│   ├── ...
│   └── 382nd jpg
├── train
│   ├── ...
│   └── 3041st jpg
└── validate
    ├── ... 
    └── 381st jpg
```

#### Dataset Instance

```
{
  "image": Image,
  "upvotes": int,  // proxy for engagement with post
  "under_proof": float,
  "over_proof": float,
  "perfect_proof": float,
  "unsure_proof": float
}
```

### Known Limitations

The comments provided are from anonymous and unverified bakers who are operating with onyl the limited
information provided by the original poster. As such, the data has a moderate amount of noise. 
Further, some of the images provided are not ideal crumb shots and have poor lighting. 