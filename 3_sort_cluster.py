import logging
import numpy as np

from clustering.sort_clusters import assign_and_sort_clusters
from my_utils import load_config

params = load_config("clustering/configs/openclip/clustering_configs.yaml")
config = load_config("semdedup_configs.yaml")
logger = logging.getLogger(__name__) 

paths_str_type = config['paths_str_type']
embed_float_type = config['embed_float_type']
emb_memory_loc = config['embs_memory_loc']
paths_memory_loc = config['path_memory_loc']
dataset_size = config['dataset_size']
emb_size = config['emd_size']

emb_memory = np.memmap(
    emb_memory_loc, 
    dtype='float32', 
    mode='r', 
    shape=(dataset_size, emb_size)
)

paths_memory = np.memmap(
    paths_memory_loc, 
    dtype=paths_str_type, 
    mode='r', 
    shape=(dataset_size,)
)

assign_and_sort_clusters(
    data=emb_memory,
    paths_list=paths_memory,
    sim_metric=params["sim_metric"],
    keep_hard=params["keep_hard"],
    kmeans_with_cos_dist=params["Kmeans_with_cos_dist"],
    save_folder=params["save_folder"],
    sorted_clusters_file_loc=params["sorted_clusters_file_loc"],
    cluster_ids=range(0, params["ncentroids"]),
    logger=logger,
)

