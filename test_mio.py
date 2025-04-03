import numpy as np
import os
from functools import partial
from transformers import CLIPModel, CLIPImageProcessor
from compute_pretrained_embeddings import get_embeddings
from dataloader import TarImageDataset, custom_collate_fn
from torch.utils.data import DataLoader
from my_utils import load_config

config = load_config("semdedup_configs.yaml")

model_name = "openai/clip-vit-base-patch32"
tar_files_directory = "/davinci-1/home/frbernardi/SemDeDup/data/raw"
batch_size = 4

model = CLIPModel.from_pretrained(model_name)
image_processor = CLIPImageProcessor.from_pretrained(model_name)
my_collate_fn = partial(custom_collate_fn, image_processor=image_processor)

dataset = TarImageDataset(tar_dir=tar_files_directory, transform=None)
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=my_collate_fn,
    num_workers=0)

for data_batch, paths_batch, batch_indices in dataloader:
    # data_batch = data_batch.to('cuda')
    break