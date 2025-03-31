import yaml

def load_config(config_path):
    """
    Load a YAML configuration file and return its contents as a dictionary.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Parsed configuration.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return None