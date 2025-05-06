import yaml
from src.dataloader import MammaliaData
from pathlib import Path


def load_path_config(path_to_config):
    with open(path_to_config, 'r') as f:
        path_config = yaml.safe_load(f)
    return {k: Path(v) for k, v in path_config.items()}


paths = load_path_config('/cfs/earth/scratch/kraftjul/BA/hpc_submit/run_md_on_earth/path_config.yml')

path_to_dataset = paths['dataset']
path_labelfiles = paths['labels']
path_to_detector_output = paths['md_output']
detector_model = 'mdv5a'
mode = 'init'

dataset = MammaliaData(
    path_to_dataset=path_to_dataset,
    path_labelfiles=path_labelfiles,
    path_to_detector_output=path_to_detector_output,
    detector_model=detector_model,
    mode=mode,
    )