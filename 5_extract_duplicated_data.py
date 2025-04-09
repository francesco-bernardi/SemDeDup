from extract_dedup_data import extract_pruned_data
from my_utils import load_config

config = load_config("semdedup_configs.yaml")

output_txt_path = config['output_txt_path']
semdedup_pruning_tables_path = config['semdedup_pruning_tables_path']
sorted_clusters_path = config['sorted_clusters_path']
eps = config['eps']
num_clusters = config['num_clusters']

extract_pruned_data(
    sorted_clusters_path, 
    semdedup_pruning_tables_path, 
    eps, 
    num_clusters, 
    output_txt_path, 
    retreive_kept_samples=True
)