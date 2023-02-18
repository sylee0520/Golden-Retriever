import typer
import json
from pathlib import Path
from typing import Optional
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
from tqdm import tqdm
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_URL = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth'
image_size = 384

def main(sample: bool = True, image_path: str = typer.Option(...), output_path: str = typer.Option(...)):
    # annotation_path: BLIP format
    file_list = os.listdir(image_path)

    model = blip_decoder(pretrained=MODEL_URL, image_size=image_size, vit="base")
    model = model.eval()
    model = model.to(device)
    captions = []

    for f in tqdm(file_list):
        img_path = image_path + '/' + f
        try:
            image = load_image(img_path, image_size)
        except:
            print(f)
        
        with torch.no_grad():
        
            if sample:
                # nucleus sampling
                caption = model.generate(image, sample=True, top_p=0.9, max_length=30, min_length=5) 
            else:
                # beam search
                caption = model.generate(image, sample=False, num_beams=3, max_length=30, min_length=5) 
            for c in caption:
                captions.append({
                    'caption': c,
                    'image': f
                })
            
    
    with open(output_path, 'w') as f:
        json.dump(captions, f, indent="\t")


def load_image(image_path: Path, image_size: int):
    raw_image = Image.open(image_path).convert('RGB')   

    w,h = raw_image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image


if __name__ == "__main__":
    typer.run(main)