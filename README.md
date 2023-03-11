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
And place your own images in the `images` directory.
BLIP model basically supports the text-to-image and/or image-to-text retrieval only on `COCO`, `flickr30k`. Please add some files for retrieving on custom datasets.
```bash
# 1. Place the 'gr_config.yaml' to BLIP/configs
mv gr_config.yaml BLIP/configs/gr_config.yaml

# 2. Place the 'gr_dataset.py' to BLIP/data
mv gr_dataset.py BLIP/data/gr_dataset.py
```
Modify the BLIP/data/\_\_init\_\_.py file.
```bash
from data.gr_dataset import gr_train, gr_retrieval_eval

def create_dataset(dataset, config, min_scale=0.5):
        
    ...
    
    elif dataset=='retrieval_gr':          
        train_dataset = gr_train(transform_train, config['image_root'], config['ann_root'])
        val_dataset = gr_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val') 
        test_dataset = gr_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')          
        return train_dataset, val_dataset, test_dataset   
    ...
    
```
Modify the BLIP/data/gr_dataset.py file.
```bash
class gr_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        '''        
        
        with open(..., 'r') as f: # caption file path
        
        ...

class gr_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        
        with open(..., 'r') as f: # caption file path
        
        ...
 
```

### 1. Captioning
```bash
CUDA_VISIBLE_DEVICES=0 python caption.py \
--sample True \
--image_path {your image directory} \
--output_path {output directory}
```
`sample` means whether you use the nucleus sampling or beam search when captioning.
`image_path` means directory containing images that you want to caption.
`output_path` means output directory.

### 2. Training the retriever

Train the 🦮 retriever!
```bash
cd BLIP

python -m torch.distributed.run --nproc_per_node=2 train_retrieval.py \
--config ./configs/retrieval_gr.yaml \
--output_dir output/gr_retrieval
```
### 3. Test your 🦮 retriever!
Code will be updated soon!

## Collaborators
[@sylee0520](https://github.com/sylee0520) [@ONground-Korea](https://github.com/ONground-Korea) [@subin9](https://github.com/subin9) [@JeonSeongHu](https://github.com/JeonSeongHu) [@
KorBrodStat](https://github.com/KorBrodStat)

## Contact
If you have any questions, please contact me! ```sy-lee@korea.ac.kr```
