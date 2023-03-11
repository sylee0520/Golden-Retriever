import torch
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
from pathlib import Path
from torchvision.utils import save_image
from matplotlib.pyplot import imshow
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from models.blip_retrieval import blip_retrieval
import utils
from data import create_dataset, create_sampler, create_loader
import ruamel_yaml as yaml
from models.blip_retrieval import blip_retrieval
from tqdm import tqdm
from torchvision.transforms import transforms
import gradio as gr
from PIL import Image
import translation

def get_image(text):    
    image_embeds = np.load('/home/aiku/AIKU/golden-retriever-team/embedding.npy')
    image_embeds = torch.from_numpy(image_embeds).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    text=translation.main(text)
    distributed = True
    image_path = '/home/aiku/AIKU/golden-retriever-team/examples'
    config = yaml.load(open('/home/aiku/AIKU/golden-retriever-team/BLIP/configs/gr_config.yaml', 'r'), Loader=yaml.Loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config)  

    if distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                        batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                        num_workers=[4,4,4],
                                                        is_trains=[True, False, False], 
                                                        collate_fns=[None,None,None])
    
    
    model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                            vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                            queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])
    model = model.to(device)

    text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
    text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
    text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
    text_ids = text_input.input_ids
    text_ids[0] = model.tokenizer.enc_token_id
    
    sims_matrix = image_embeds @ text_embed.t()
    
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((1,len(val_loader.dataset.image)),-100.0).to(device)
    score_matrix_t2i = sims_matrix
    score_matrix_t2i = score_matrix_t2i.cpu().detach().numpy()
    score_matrix_t2i = np.argsort(score_matrix_t2i[0, :])[::-1]
    best_img_idx = score_matrix_t2i[:5]
    best_img = [val_loader.dataset.image[idx] for idx in best_img_idx]
    image = [Image.open(image_path + '/' + bimg) for bimg in best_img]
    return image

if __name__ == "__main__":
    demo = gr.Blocks()
    with demo:
        gr.Markdown(
            """
        # 🦮 Golden Retriever
        Type a description of the photo you want to find.
        """
        )
        inputs = gr.Textbox(label="Enter what you want to find")
        with gr.Row():
            output1 = [gr.Image(label="Top 1", type="pil").style(height=400, width=400)]
        with gr.Row():
            output2 = [gr.Image(label="Top 2", type="pil").style(height=200, width=200),
                gr.Image(label="Top 3", type="pil").style(height=200, width=200),
                gr.Image(label="Top 4", type="pil").style(height=200, width=200),
                gr.Image(label="Top 5", type="pil").style(height=200, width=200)]
        find_button=gr.Button("Find")
        find_button.click(get_image, inputs=inputs, outputs=output1+output2)

    demo.launch()