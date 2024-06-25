# BreadRx

The proof is in the crumb? 

#### TODO
- [ ] Create labeled dataset of images # finish by Monday
  - [ ] refactor image parsing code
  - [ ] Connect to OpenAI API to determine whether over or under proofed + strength
  - [ ] Upload to HuggingFace

- [ ] Develop CNN architecture & train on Google colab Pro with CUDA. Just try smaller dataset first with PyTorch + MLFlow

- [ ] Add bias-variance tradeoff to determine whether more data would improve accuracy
- [ ] Evaluate model & create image heatmap to understand how model looks at image

- [ ] Build a simple website where user can upload image of their bread for the classifier
- [ ] Deploy model using Onnx Runtime 
- [ ] Host website on DigitalOcean
- [ ] Clean up GitHub repo
  - [ ] make from_scratch/data csv body comments proper json format then adjust openai_labels to match this change


## Overview

Building off Justin Kolpak's model Databricks 

## Data

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

#### Images

All images were cropped and scaled to 512x512 such that rectangular images
had their long edge cropped (rather than compressed).

### Labels
OpenAI's API was provided the post title, post description, and all post comments to
determine a proofing label: "under proofed", "adequate", "over proofed," "inconclusive" and a signal strength, which ranged from [0, 1].

All fully parsed and cleaned data can be found on huggingface.io here.

## CNN Model

### Architecture
PyTorch


### Training

Colab Pro + CUDA

### Evaluation

#### Accuracy
#### Bias-Variance Trade-Off
#### CNN Analysis


## Fine-tuned


## Further Questions
1. Other data sources?  
2. Adversarial Data Augmentation?

