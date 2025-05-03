import yaml
import torch
from argparse import ArgumentParser
from torchvision.transforms import v2
from pathlib import Path
from typing import Sequence

from ba_dev.dataset import MammaliaDataImage
from ba_dev.datamodule import MammaliaDataModule
from ba_dev.transform import ImagePipeline
from ba_dev.model import LightningModelImage
from ba_dev.trainer import MammaliaTrainer


def print_banner(text, width=80, border_char='-'):
    inner_width = width - 4
    line = border_char * (width - 2)
    centered = text.center(inner_width)

    print(f"+{line}+")
    print(f"| {centered} |")
    print(f"+{line}+")


def read_config_yaml(config_path):
    try:
        with open(config_path) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file not found at {config_path}. Please provide a valid path."
            )
    except yaml.YAMLError as e:
        raise ValueError(
            f"Error parsing YAML file at {config_path}: {e}"
            )


def set_up_image_pipeline(cfg):
    pre_ops = []
    if cfg['to_rgb']:
        pre_ops.append(('to_rgb', {}))
    if cfg['crop_by_bb']:
        pre_ops.append(('crop_by_bb', {'crop_shape': cfg['crop_by_bb']}))

    ops = []
    if cfg['to_tensor']:
        ops.append([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
            ])

    resize = cfg['resize']
    if resize:
        if isinstance(resize, int):
            ops.append(v2.Resize((resize, resize)))
        elif isinstance(resize, Sequence) and len(resize) == 2:
            ops.append(v2.Resize((resize)))
        else:
            raise ValueError(
                f"Invalid resize value: {resize}. Must be int or Sequence of two ints."
                )

    norm = cfg['normalize']
    if norm:
        if isinstance(norm, dict):
            mean = norm['mean']
            std = norm['std']
        elif isinstance(norm, str):
            if norm.lower() == 'imagenet':
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            else:
                stats = torch.load(norm)
                mean = stats['mean']
                std = stats['std']
        ops.append(v2.Normalize(mean=mean, std=std))

    image_pipeline = ImagePipeline(
        pre_ops=pre_ops,
        transform=v2.Compose(ops)
        )

    augment = cfg['augmentation']
    if not augment:
        augmented_image_pipeline = None
    else:
        ops_aug = list(ops)

        for entry in augment:
            name, params = next(iter(entry.items()))
            Op = getattr(v2, name, None)
            if Op is None:
                raise ValueError(f"Unknown transform: {name!r}")
            ops_aug.append(Op(**(params or {})))

        augmented_image_pipeline = ImagePipeline(
                    pre_ops=pre_ops,
                    transform=v2.Compose(ops_aug)
                    )

    return image_pipeline, augmented_image_pipeline


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help='Path to the config yaml file.'
        )
    parser.add_argument(
        '--dev_run',
        action='store_true',
        help='Run a quick dev run to test the experiment setup.'
        )

    args = parser.parse_args()

    cfg = read_config_yaml(args.config_path)

    if args.dev_run:
        print_banner("!!!   Running in dev mode   !!!", width=80)

    # setting up image pipeline
    image_pipeline, augmented_image_pipeline = set_up_image_pipeline(cfg['image_pipeline'])



    dataset_kwargs = {
            'path_labelfiles': paths['test_labels'],
            'path_to_dataset': paths['dataset'],
            'path_to_detector_output': paths['md_output'],
            }

    datamodule = MammaliaDataModule(
                    dataset_cls=MammaliaDataImage,
                    dataset_kwargs=dataset_kwargs,
                    n_folds=5,
                    test_fold=0,
                    image_pipeline=image_pipeline,
                    augmented_image_pipeline=augmented_image_pipeline,
                    batch_size=32,
                    num_workers=1,
                    pin_memory=True,
                    )

    model = LightningModelImage(
                num_classes=datamodule.num_classes,
                class_weights=datamodule.class_weights,
                backbone_name='efficientnet_b0',
                backbone_pretrained=True,
                backbone_weights='DEFAULT',
                optimizer_name='AdamW',
                optimizer_kwargs={
                    'lr': 1e-3,
                    'weight_decay': 1e-5,
                    'amsgrad': False
                    },
                scheduler_name='CosineAnnealingLR',
                scheduler_kwargs={'T_max': 5},
                )

    log_dir = Path('/cfs/earth/scratch/kraftjul/BA/output/test')
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    trainer = MammaliaTrainer(
                log_dir=log_dir,
                accelerator='cpu',
                devices=1,
                patience=5,
                log_every_n_steps=1,
                trainer_kwargs={
                    'max_epochs': 1,
                    }
                )
