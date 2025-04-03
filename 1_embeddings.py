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
batch_size = 1

model = CLIPModel.from_pretrained(model_name)
image_processor = CLIPImageProcessor.from_pretrained(model_name)

dataset = TarImageDataset(tar_dir=tar_files_directory, transform=None)
my_collate_fn = partial(custom_collate_fn, image_processor=image_processor)
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=my_collate_fn,
    num_workers=0)

path_str_type = 'float32'
emb_memory_loc = config['embs_memory_loc']
paths_memory_loc = ...
dataset_size = dataset.__len__()
emb_size = config['emd_size']

emb_array = np.memmap(
    emb_memory_loc,
    dtype='float32',
    mode='w+',
    shape=(dataset_size, emb_size)
)

path_array = np.memmap(
    emb_memory_loc, 
    dtype=path_str_type, 
    mode='w+', 
    shape=(dataset_size,)
)


if __name__ == "__main__":
    embeddings, paths = get_embeddings(model, dataloader, emb_array, path_array)

    # Create directories if they do not exist
    save_folder = config['save_folder']
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save embeddings
    embeddings_file = os.path.join(save_folder, "embeddings.npy")
    np.save(embeddings_file, embeddings)
    print(f"Embeddings saved to {embeddings_file}")

    # Save paths
    paths_file = os.path.join(save_folder, "paths.npy")
    np.save(paths_file, paths)
    print(f"Paths saved to {paths_file}")









# import torch
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image

# def get_image_embeddings(image_paths, model_name="openai/clip-vit-base-patch32", device=None):
#     # Set the device to GPU if available, otherwise CPU.
#     device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
# #     # Load the CLIP model and processor from Hugging Face.
# #     model = CLIPModel.from_pretrained(model_name)
# #     processor = CLIPProcessor.from_pretrained(model_name)
# #     model.to(device)
    
#     # Load and convert each image using PIL.
#     images = [Image.open(path).convert("RGB") for path in image_paths]
    
# #     # Process images to obtain pixel values required by the model.
# #     inputs = processor(images=images, return_tensors="pt", padding=True)
# #     pixel_values = inputs["pixel_values"].to(device)
    
#     # Compute image embeddings without gradient calculations.
#     with torch.no_grad():
#         outputs = model(pixel_values=pixel_values)
    
#     # The image embeddings are stored in the 'image_embeds' attribute.
#     image_embeddings = outputs.image_embeds
    
#     # Optionally, move embeddings to CPU and convert to numpy array.
#     return image_embeddings.cpu().numpy()

