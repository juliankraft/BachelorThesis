import yaml
from pathlib import Path
from ba_dev.dataloader import MammaliaData


def load_path_config(path_to_config):
    with open(path_to_config, 'r') as f:
        path_config = yaml.safe_load(f)
    return {k: Path(v) for k, v in path_config.items()}


paths = load_path_config('../path_config.yml')

path_to_dataset = paths['dataset']
path_labelfiles = paths['testset']
path_to_detector_output = path_labelfiles / 'md_out'
detector_model = 'mdv5a'
mode = 'init'

dataset = MammaliaData(
    path_to_dataset=path_to_dataset,
    path_labelfiles=path_labelfiles,
    path_to_detector_output=path_to_detector_output,
    detector_model=detector_model,
    mode=mode,
    )