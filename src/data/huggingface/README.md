
## Dataset Card for Bread Proofing

### Dataset Summary


### Features
All data was collected and parsed from the /r/Breadit and /r/Sourdough subreddits, with raw data originating from 
[Academic Torrents](https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10), as Reddit does not natively provide all available posts. The data spans from the 
beginning of each subreddit's creation to 31 December 2023. 

To download the data yourself, follow the instructions provided [here](https://www.reddit.com/r/pushshift/comments/1akrhg3/).
The [script provided](https://github.com/Watchful1/PushshiftDumps/blob/master/scripts/filter_file.py) in the 
above instructions was used to remove comments that made no mention of proofing. For images contained in a gallery, only the
first image was ingested into the dataset for simplicity's sake. All non-crumb pictures were removed manually using 
[Narrative Select](https://narrative.so/select). Of the 12,558 posts that mentioned proofing, 10,027 contained a non-deleted
image, and 3,802 contained a usable crumb image. 

There are torrents available with more up-to-date posts, so more images could be collected in the future, but pursing 
alternative sources would likely yield better results. 

### Dataset Structure

#### Dataset Instance

```json
{
  "image": "",
  
  
}
```

#### Data Split

```
|    split    |   samples   |
-----------------------------
|    train    |     3800    |
|  validate   |      380    |
|    test     |      381    |
```

### Dataset Curation

#### Curation Rationale 

reason for creating dataset 

### Source Data

#### Data Collection 

All images were cropped and scaled to 512x512 such that rectangular images
had their long edge cropped (rather than compressed).

#### Data Labelling

OpenAI's API was provided the post title, post description, and all post comments to
determine a proofing label: "under proofed", "adequate", "over proofed," "inconclusive" and a signal strength, which ranged from [0, 1].

