# run this script using:
# caffeinate -is conda run -n BA python run_md_on_osx.py 2>&1 | tee -a run_md.log

import datetime
from ba_stable.dataloader import MammaliaData
from ba_stable.utils import load_config_yaml

# Print log header
print("=" * 60)
print("Running MegaDetector Initialization Script")
print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

paths = load_config_yaml('path_config.yml')

path_to_dataset = paths['dataset']
path_labelfiles = paths['labels']
path_to_detector_output = paths['md_output']
detector_model = 'mdv5a'
mode = 'init'

print("=" * 60)
print("Configurations:")
print(f"Dataset: {path_to_dataset}")
print(f"Labels: {path_labelfiles}")
print(f"Detector output: {path_to_detector_output}")
print(f"Detector model: {detector_model}")
print(f"Mode: {mode}")
print("=" * 60)

dataset = MammaliaData(
    path_to_dataset=path_to_dataset,
    path_labelfiles=path_labelfiles,
    path_to_detector_output=path_to_detector_output,
    detector_model=detector_model,
    mode=mode,
    )
