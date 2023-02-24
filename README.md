# 🦮 Golden Retriever
<p align="center">
  <img src="https://user-images.githubusercontent.com/72010172/219724337-78234040-2f25-4620-86dd-e48e86963ebc.gif">
</p><br>
This is the repository of 'Golden Retriever' for AIKU team Project! Golden Retriever is the service that search most similar images given user text query. I think all of you've suffered searching images on your laptop or phone. Golden Retriever can pick up the images fast and correctly like this, Woof!

## Architecture
First, place your own images at folder. And then, (1) generate the caption of images to train the model that can align with images well. In this stage, you'll have image-caption pairs and (2) finetune the text-to-image retrieval model with pairs. Finally, (3) if you put a description of the images you want to search into the trained model, you will get the images you want!

## Demo

## Usage
If you want to train or inference the golden-retriever with your own images, please follow the next steps.
### 0. Basic Setting
```bash
# create conda env
conda create -n gr python=3.8

# activate conda env
conda activate gr
```
Our framework is based on BLIP model. BLIP is the Pretrained Visual-Language Model. Please refer official [repository](https://github.com/salesforce/BLIP) or [paper](https://arxiv.org/abs/2201.12086) for more details.
```bash
git clone https://github.com/salesforce/BLIP.git
```

## Collaborators
[@sylee0520](https://github.com/sylee0520) [@ONground-Korea](https://github.com/ONground-Korea) [@subin9](https://github.com/subin9) [@JeonSeongHu](https://github.com/JeonSeongHu) [@
KorBrodStat](https://github.com/KorBrodStat)

## Contact
If you have any questions, please contact me! ```sy-lee@korea.ac.kr```
