import numpy as np
import os
from functools import partial
from transformers import CLIPModel, CLIPImageProcessor
from compute_pretrained_embeddings import get_embeddings
from dataloader import TarImageDataset, custom_collate_fn
from torch.utils.data import DataLoader
from my_utils import load_config

config = load_config("semdedup_configs.yaml")

model_name = config['model_name']
tar_files_directory = os.path.abspath("data/raw")
batch_size = config['batch_size']

model = CLIPModel.from_pretrained(model_name)
image_processor = CLIPImageProcessor.from_pretrained(model_name)

dataset = TarImageDataset(tar_dir=tar_files_directory, transform=None)
my_collate_fn = partial(custom_collate_fn, image_processor=image_processor)
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=my_collate_fn,
    num_workers=0
)

paths_str_type = config['paths_str_type']
embed_float_type = config['embed_float_type']
emb_memory_loc = config['embs_memory_loc']
paths_memory_loc = config['path_memory_loc']
dataset_size = dataset.__len__()
emb_size = config['emd_size']

emb_array = np.memmap(
    emb_memory_loc, 
    dtype=embed_float_type, 
    mode='w+', 
    shape=(dataset_size, emb_size)
)

path_array = np.memmap(
    paths_memory_loc, 
    dtype=paths_str_type,
    mode='w+', 
    shape=(dataset_size,)
)


if __name__ == "__main__":
    get_embeddings(
        model, 
        dataloader, 
        emb_array, 
        path_array
    )