from argparse import ArgumentParser
from ba_dev.utils import load_path_yaml


if __name__ == '__main__':

    paths = load_path_yaml('/cfs/earth/scratch/kraftjul/BA/data/path_config.yml')

    parser = ArgumentParser()

    parser.add_argument(
        '--experiment',
        type=str,
        default='default_experiment',
        help='Name of the experiment to run'
        )
    parser.add_argument(
        '--config',
        type=str,
        default='default_config',
        help='Name of the config to run'
        )