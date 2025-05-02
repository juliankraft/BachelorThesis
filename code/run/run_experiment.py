import yaml
import torch
from argparse import ArgumentParser
from torchvision.transforms import v2
from pathlib import Path

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

    if not Path(args.config_path).exists():
        raise FileNotFoundError(f"Config file {args.config_path} does not exist.")
    with open("config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.dev_run:
        print_banner("!!!   Running in dev mode   !!!", width=80)


    # setting up image pipeline

    if cfg['feature_stats'] is 


    stats = torch.load(paths['feature_stats'])

    image_pipeline = ImagePipeline(
            pre_ops=[
                if cfg['to_rgb']: ('to_rgb', {}),
                ('crop_by_bb', {'crop_shape': 1.0})
                ],
            transform=v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((224, 224)),
                v2.Normalize(
                    mean=stats['mean'],
                    std=stats['std']
                    )
                ])
            )

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
                    augmented_image_pipeline=None,
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
