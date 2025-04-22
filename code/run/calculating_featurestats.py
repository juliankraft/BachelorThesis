import torch

from torchvision.transforms import v2
from torch.utils.data import DataLoader

from ba_stable.dataset import MammaliaDataImage
from ba_stable.transform import ImagePipeline
from ba_stable.utils import load_config_yaml

paths = load_config_yaml('/cfs/earth/scratch/kraftjul/BA/code/path_config.yml')
target_path = paths['output'] / 'feature_stats.pt'

image_pipeline = ImagePipeline(
                pre_ops=[
                    ('to_rgb', {}),
                    ('crop_by_bb', {})
                ],
                transform=v2.Compose([
                                v2.ToImage(),
                                v2.ToDtype(torch.float32, scale=True),
                                ])
                )

dataset = MammaliaDataImage(
    path_labelfiles=paths['labels'],
    path_to_dataset=paths['dataset'],
    path_to_detector_output=paths['md_output'],
    detector_model=None,
    mode='init',
    image_pipeline=image_pipeline
)


def collate_fn(batch):
    return batch


loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=False 
    )


channel_sum = torch.zeros(3)
pixel_count = 0

print('Calculating mean...', flush=True)
for batch in loader:
    for item in batch:
        img = item['x']
        pixel_count += img.shape[1] * img.shape[2]
        for c in range(img.shape[0]):
            channel_sum[c] += img[c].sum()

mean = channel_sum / pixel_count

print('Calculating std...', flush=True)
channel_diff_squared_sum = torch.zeros(3)
for batch in loader:
    for item in batch:
        img = item['x']
        img_centered_squared = (img - mean[:, None, None]) ** 2
        for c in range(img_centered_squared.shape[0]):
            channel_diff_squared_sum[c] += img_centered_squared[c].sum()

std = torch.sqrt(channel_diff_squared_sum / pixel_count)

print("Mean:", mean, flush=True)
print("Std:", std, flush=True)

feature_stats = {'mean': mean, 'std': std}
torch.save(feature_stats, target_path)

print('Feature statistics saved to', target_path, flush=True)
