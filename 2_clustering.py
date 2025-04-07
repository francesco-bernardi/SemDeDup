import yaml
import random
import numpy as np
import logging
from clustering.clustering import compute_centroids


logger = logging.getLogger(__name__) 
logger.addHandler(logging.StreamHandler())

confg_file = "clustering/configs/openclip/clustering_configs.yaml"

## -- Load kmeans clustering parameters from configs file
with open(confg_file, 'r') as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)

## -- Fix the seed
SEED = params['seed']
random.seed(SEED)
emb_memory_loc = params['emb_memory_loc'] 
paths_memory_loc = params['paths_memory_loc'] 
dataset_size = params['dataset_size'] 
emb_size = params['emb_size'] 
path_str_type = params['paths_str_type']

emb_memory = np.memmap(emb_memory_loc, dtype='float32', mode='r', shape=(dataset_size, emb_size))
paths_memory = np.memmap(paths_memory_loc, dtype=path_str_type, mode='r', shape=(dataset_size,))

if __name__ == "__main__":
    compute_centroids(
        data=emb_memory,
        ncentroids=params['ncentroids'],
        niter=params['niter'],
        seed=params['seed'],
        Kmeans_with_cos_dist=params['Kmeans_with_cos_dist'],
        save_folder=params['save_folder'],
        logger=logger,
        verbose=True,
    )