import yaml
from pathlib import Path


def load_config_yaml(path_to_config):
    with open(path_to_config, 'r') as f:
        path_config = yaml.safe_load(f)
    return {k: Path(v) for k, v in path_config.items()}