import yaml
import shutil
import torch
from time import time
from argparse import ArgumentParser
from torchvision.transforms import v2
from pathlib import Path
from typing import Sequence

from ba_dev.dataset import MammaliaDataImage
from ba_dev.datamodule import MammaliaDataModule
from ba_dev.transform import ImagePipeline
from ba_dev.model import LightningModelImage
from ba_dev.trainer import MammaliaTrainer
from ba_dev.utils import count_parameters


def print_banner(text, width=89, border_char='-'):
    inner_width = width - 4
    line = border_char * (width - 2)
    centered = text.center(inner_width)

    print(f'+{line}+')
    print(f'| {centered} |')
    print(f'+{line}+')


def read_config_yaml(config_path):
    try:
        with open(config_path) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        raise FileNotFoundError(
            f'Config file not found at {config_path}. Please provide a valid path.'
            )
    except yaml.YAMLError as e:
        raise ValueError(
            f'Error parsing YAML file at {config_path}: {e}'
            )


def set_up_image_pipeline(cfg):
    pre_ops = []
    if cfg['to_rgb']:
        pre_ops.append(('to_rgb', {}))
    if cfg['crop_by_bb']:
        pre_ops.append(('crop_by_bb', {'crop_shape': cfg['crop_by_bb']}))

    ops = []
    if cfg['to_tensor']:
        ops.append(v2.ToImage())
        ops.append(v2.ToDtype(torch.float32, scale=True))

    resize = cfg['resize']
    if resize:
        if isinstance(resize, int):
            ops.append(v2.Resize((resize, resize)))
        elif isinstance(resize, Sequence) and len(resize) == 2:
            ops.append(v2.Resize((resize)))
        else:
            raise ValueError(
                f'Invalid resize value: {resize}. Must be int or Sequence of two ints.'
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
                raise ValueError(f'Unknown transform: {name!r}')
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

    args = parser.parse_args()
    config_path = Path(args.config_path)

    cfg = read_config_yaml(config_path)
    dev_run = cfg['dev_run']
    paths = {key: Path(value) for key, value in cfg['paths'].items()}

    if dev_run:
        print_banner('!!!   Running in dev mode   !!!', width=89)
        log_dir = paths['output_path'] / f'{cfg["experiment_name"]}_dev'
        if log_dir.exists():
            shutil.rmtree(log_dir)
    else:
        print_banner('!!!   Running Experiment   !!!', width=89)
        log_dir = paths['output_path'] / f'{cfg["experiment_name"]}'
        if log_dir.exists():
            raise FileExistsError(
                f'Output directory {log_dir} already exists. Please provide a different name.'
                )
    log_dir.mkdir(parents=True)

    experiment_info_path = log_dir / 'experiment_info.yaml'
    shutil.copy2(config_path, experiment_info_path)

    # setting up image pipeline
    image_pipeline, augmented_image_pipeline = set_up_image_pipeline(cfg['image_pipeline'])

    # setting up datamodule config
    label_key = 'test_labels' if dev_run else 'labels'
    dataset_raw = cfg.get('dataset') or {}
    dataset_kwargs = {
        'path_labelfiles': paths[label_key],
        'path_to_dataset': paths['dataset'],
        'path_to_detector_output': paths['md_output'],
        **dataset_raw
        }

    datamodule_raw = cfg.get('data_module') or {}
    datamodule_cfg = {
        'dataset_cls': MammaliaDataImage,
        'image_pipeline': image_pipeline,
        'augmented_image_pipeline': augmented_image_pipeline,
        **datamodule_raw
        }

    # setting up model config
    model_cfg = cfg['model']

    # setting up trainer config
    trainer_raw = cfg.get('trainer') or {}

    _not_dev_defaults = {
        'limit_train_batches': 1.0,
        'limit_val_batches': 1.0,
        'limit_test_batches': 1.0,
        'limit_predict_batches': 1.0,
        'max_epochs': -1,
        'log_every_n_steps': 10,
        }

    if dev_run:
        dev_run_args = {
            'limit_train_batches': 1,
            'limit_val_batches': 1,
            'limit_test_batches': 1,
            'limit_predict_batches': 1,
            'max_epochs': 2,
            'log_every_n_steps': 1
            }
    else:
        dev_run_args = (trainer_raw.get('not_dev') or {}).copy()
        for key, value in _not_dev_defaults.items():
            if key not in dev_run_args:
                dev_run_args[key] = value

    trainer_kwargs = {
        **(trainer_raw.get('trainer_kwargs') or {}),
        **dev_run_args
        }

    trainer_cfg = (trainer_raw.get('base_args') or {}).copy()
    trainer_cfg['trainer_kwargs'] = trainer_kwargs

    trainer_do_predict = trainer_raw['do_predict']

    # setting up folds or cross-validation
    cross_val = cfg['cross_val']['apply']
    n_folds = cfg['cross_val']['n_folds']
    test_fold = cfg['cross_val']['test_fold']

    if cross_val:
        if dev_run:
            folds = range(2)
        else:
            folds = range(n_folds)
    else:
        folds = [test_fold]

    run_params = {}
    if cross_val:
        run_params['folds'] = {}

    first_pass = True
    print('Configuration complete...')
    print('Configuration file copied to output directory.')

    # running the experiment
    for fold in folds:

        fold_params = {}

        if cross_val:
            trainer_log_dir = log_dir / f'fold_{fold}'
            print_statement = f'Running cross-validation fold {fold+1}/{len(folds)}'
            trainer_log_dir.mkdir(parents=True)
        else:
            trainer_log_dir = log_dir
            print_statement = f'Running Experiment with test fold = {test_fold}'

        print_banner(print_statement, width=89)

        dm_cfg = datamodule_cfg.copy()
        dm_cfg['dataset_kwargs'] = dataset_kwargs.copy()
        datamodule = MammaliaDataModule(
                        n_folds=5,
                        test_fold=fold,
                        **dm_cfg,
                        )

        m_cfg = model_cfg.copy()
        model = LightningModelImage(
                        num_classes=datamodule.num_classes,
                        class_weights=datamodule.class_weights,
                        **m_cfg
                        )

        t_cfg = trainer_cfg.copy()
        trainer = MammaliaTrainer(
                        log_dir=trainer_log_dir,
                        **t_cfg
                        )

        t0 = time()
        trainer.fit(
            model=model,
            datamodule=datamodule
            )
        fold_params['training_time'] = round(time() - t0, 0)
        fold_params['epochs_trained'] = trainer.current_epoch

        test_metrics = trainer.test(
            model=model,
            datamodule=datamodule
            )

        if trainer_do_predict:
            best_ckpt = trainer_log_dir / 'checkpoints' / 'best.ckpt'
            trainer.predict(
                model=model,
                datamodule=datamodule,
                ckpt_path=best_ckpt,
                return_predictions=False
                )

        fold_params['test_metrics'] = test_metrics[0]
        fold_params['class_weights'] = datamodule.class_weights.tolist()

        if first_pass:
            datamodule.dataframe.to_csv(log_dir / 'dataset.csv', index=False)
            run_params['model_parameters'] = count_parameters(model)
            run_params['label_decoder'] = datamodule.label_decoder
            run_params['num_classes'] = datamodule.num_classes
            first_pass = False

        if cross_val:
            fold_params['test_fold'] = fold
            run_params['folds'][fold] = fold_params
        else:
            run_params.update(fold_params)

        temp_info_path = log_dir / 'temp_experiment_info.yaml'
        with open(temp_info_path, "w") as f:
            yaml.dump(run_params, f, default_flow_style=False)

    run_output = {'output': run_params}
    with open(experiment_info_path, "a") as f:
        f.write("\n")
        yaml.dump(run_output, f, default_flow_style=False)
    
    temp_info_path.unlink(missing_ok=True)

    print_banner('Experiment completed!', width=89)
