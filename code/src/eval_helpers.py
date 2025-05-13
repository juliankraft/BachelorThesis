
import ast
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import matplotlib.patches as patches

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from IPython.display import display
from PIL import Image
from pathlib import Path
from os import PathLike
from typing import Dict, Any

from ba_dev.dataset import MammaliaDataImage
from ba_dev.utils import BBox


def set_custom_plot_style():
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['font.size'] = 8
    plt.rcParams['legend.fontsize'] = 6

    plt.rcParams['axes.linewidth'] = 0.2
    plt.rcParams['xtick.major.width'] = 0.1
    plt.rcParams['ytick.major.width'] = 0.1
    plt.rcParams['xtick.minor.width'] = 0.05
    plt.rcParams['ytick.minor.width'] = 0.05


def plot_image_with_bbox(
        image: Image.Image,
        bbox: BBox,
        conf: float | None = None) -> Figure:

    """
    Plot an image with a bounding box.
    Args:
        image (Image.Image): The image to plot.
        bbox (BBox): The bounding box to plot, in the format [x, y, w, h].
        conf (float | None): The confidence score to display. If None, no score is displayed.
    Returns:
        Figure: The matplotlib figure object.
    """

    width, height = image.size

    x_abs = bbox[0] * width
    y_abs = bbox[1] * height
    w_abs = bbox[2] * width
    h_abs = bbox[3] * height

    fig, ax = plt.subplots()
    ax.imshow(image)

    rect = patches.Rectangle(
        (x_abs, y_abs), w_abs, h_abs,
        linewidth=1, edgecolor='red', facecolor='none'
    )

    ax.add_patch(rect)

    if conf is not None:
        ax.text(
            x_abs + 5, y_abs - 10,
            f"conf = {conf:.2f}",
            fontsize=8,
            color='white',
            bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', pad=1.5)
        )

    ax.axis('off')
    plt.tight_layout()

    plt.close(fig)

    return fig


def draw_bbox_on_ax(
        ax: Axes,
        image: Image.Image,
        bbox: BBox,
        conf: float | None = None
        ) -> None:
    """
    Draw an image with a bounding box onto an existing Axes.

    Args:
        ax (Axes): The axes to draw into.
        image (Image.Image): The image to plot.
        bbox (BBox): Bounding box [x, y, w, h] in relative coords (0-1).
        conf (float, optional): Confidence score to annotate.
    """
    width, height = image.size
    x_abs, y_abs = bbox[0] * width, bbox[1] * height
    w_abs, h_abs = bbox[2] * width, bbox[3] * height

    ax.imshow(image)
    rect = patches.Rectangle(
        (x_abs, y_abs), w_abs, h_abs,
        linewidth=1, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

    if conf is not None:
        ax.annotate(
            f"conf = {conf:.2f}",
            xy=(x_abs, y_abs),
            xytext=(2, 8),
            textcoords="offset points",
            ha="left", va="top",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="red", alpha=0.5, edgecolor="none", pad=1.5),
            clip_on=False
        )
    ax.axis('off')


def smooth_data(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot_model_metrics(
        metrics: PathLike | pd.DataFrame,
        title: str | None = None,
        type: str = 'valid',
        balanced_accuracy: bool = True,
        step: bool = False,
        image_size_cm: list[int] = [14, 7],
        window_size: int = 5
        ):

    """
    Plot the training and validation metrics of a run.

    Parameters:

        run (str): Run name.
        type (str): Type of metrics ('train', 'valid'). Default is 'valid'.
        step (bool): for type 'train', plot step instead of epoch. Default is False.
        balanced_accuracy (bool): If True, plot balanced accuracy. Default is True.
        image_size_cm (list): Size of the output image in cm.
        window_size (int): Size of the smoothing window. Default is 5.

    """
    if isinstance(metrics, PathLike):
        metrics = pd.read_csv(metrics)

    if balanced_accuracy:
        acc_type = 'bal_acc'
        acc_label = 'Balanced Accuracy'
    else:
        acc_type = 'acc'
        acc_label = 'Accuracy'

    if type == 'valid':
        loss = 'val_loss'
        acc = 'val_' + acc_type
        label = ['Validation Loss', 'Validation ' + acc_label]
    elif type == 'train':
        label = ['Training Loss', 'Training ' + acc_label]
        if step:
            loss = 'train_loss_step'
            acc = 'train_' + acc_type + '_step'
        else:
            loss = 'train_loss_epoch'
            acc = 'train_' + acc_type + '_epoch'

    validation_data = metrics[~metrics[loss].isnull()]

    # Smooth the data
    smoothed_val_loss = smooth_data(validation_data[loss], window_size=window_size)
    smoothed_val_acc = smooth_data(validation_data[acc], window_size=window_size)

    # Adjust the epoch range to match the length of the smoothed data
    epochs = validation_data['epoch'][len(validation_data['epoch']) - len(smoothed_val_loss):]

    fig, ax1 = plt.subplots(figsize=(image_size_cm[0]/2.54, image_size_cm[1]/2.54))

    if title:
        plt.title(title)

    # Plot unsmoothed valid_loss on the primary y-axis
    ax1.plot(validation_data['epoch'], validation_data[loss], color='#A6CEE3', label='', linewidth=1, alpha=0.5)
    # Plot smoothed valid_loss on the primary y-axis
    ax1.plot(epochs, smoothed_val_loss, color='#1F78B4', label=label[0], linewidth=0.5)

    # Adjust y-axis limits for loss
    loss_min = float(validation_data[loss].min())
    loss_max = float(validation_data[loss].max())
    ax1.set_ylim(loss_min * 0.87, loss_max * 1)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(label[0])
    ax1.tick_params(axis='y')

    # Create a secondary y-axis to plot valid_acc
    ax2 = ax1.twinx()
    # Plot unsmoothed valid_acc on the secondary y-axis
    ax2.plot(validation_data['epoch'], validation_data[acc], color='#FB9A99', label='', linewidth=1, alpha=0.5)
    # Plot smoothed valid_acc on the secondary y-axis
    ax2.plot(epochs, smoothed_val_acc, color='#E31A1C', label=label[1], linewidth=0.5)
    ax2.set_ylabel(label[1])
    ax2.tick_params(axis='y')

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower center', bbox_to_anchor=(0.5, 0.2), ncol=2)

    fig.tight_layout()

    return fig


class LoadRun:
    def __init__(
        self,
        log_path: str | PathLike
    ):
        self.log_path = Path(log_path)
        self.info = self.get_experiment_info()
        self.cross_val = self.info['cross_val']['apply']
        if self.cross_val:
            self.folds = range(2 if self.info.get('dev_run', False) else self.info['cross_val']['n_splits'])
        else:
            self.folds = [None]

        self.ds_path = Path(self.info['paths']['dataset'])
        self.run_dataset = self.get_run_dataset()

        self.decoder = self.info['output']['label_decoder']

    def show_sample(
            self,
            idx: int,
            info_to_print: str | list[str] | None = None,
            show_figures: bool = True
            ) -> Figure:

        sample = self.get_sample(idx=idx)

        if isinstance(info_to_print, str):
            info_to_print = [info_to_print]

        if info_to_print:
            for info in info_to_print:
                if info in sample:
                    print(f"{info}: {sample[info]}")
                else:
                    print(f"Info '{info}' not an available key.")

        figure = plot_image_with_bbox(
                image=Image.open(sample['path']),
                bbox=sample['bbox'],
                conf=sample['conf']
                )

        if show_figures:
            display(figure)

        return figure

    def show_all_bboxes_for_image(
            self,
            idx: int,
            first_n: int = -1,
            show_figures: bool = True
            ) -> list[Figure]:

        sample = self.get_sample(idx=idx)
        detections = self.get_bb_for_file(idx=idx)
        if not first_n == -1:
            detections = detections[:first_n]
        image = Image.open(sample['path'])

        figures = []

        for det in detections:
            figure = plot_image_with_bbox(
                    image=image,
                    bbox=det[0],
                    conf=det[1]
                    )

            if show_figures:
                display(figure)

            figures.append(figure)

        return figures

    def get_bb_for_file(
            self,
            idx: int
            ):

        seq_id = self.run_dataset.iloc[idx]['seq_id']
        file_name = Path(self.run_dataset.iloc[idx]['file_path']).name

        path_to_detection_results = Path(self.info['paths']['md_output']) / f"{seq_id}.json"
        with open(path_to_detection_results, 'r') as f:
            data = json.load(f)

        for entry in data:
            if entry['file'] == file_name:
                detections = entry['detections']

        pairs = [
                (det['bbox'], det['conf'])
                for det in detections
                if int(det['category']) == 1
                ]

        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

        return pairs_sorted

    def get_dataset(
            self,
            fold: int = 0,
            ) -> MammaliaDataImage:

        return MammaliaDataImage(
            path_labelfiles=self.info['paths']['labels'],
            path_to_dataset=self.info['paths']['dataset'],
            path_to_detector_output=self.info['paths']['md_output'],
            n_folds=self.info['cross_val']['n_folds'],
            test_fold=fold,
            mode='eval',
            image_pipeline=None,
            **self.info['dataset']
            )

    def calculate_metrics(
            self,
            metric: str,
            set_selection: list[str] | str = 'test',
            **kwargs
            ):

        func = getattr(skm, metric, None)
        if func is None:
            raise ValueError(f"Metric '{metric}' not found in sklearn.metrics")

        def compute(df_pred: pd.DataFrame):
            y_true = df_pred['class_id']
            y_pred = df_pred['pred_id']
            return func(y_true, y_pred, **kwargs)

        if self.cross_val:
            results = []
            for fold in self.folds:
                df_pred = self.get_predictions(fold=fold, set_selection=set_selection)
                results.append(compute(df_pred))
            return results
        else:
            df_pred = self.get_predictions(set_selection=set_selection)
            return compute(df_pred)

    def get_sample(
            self,
            idx: int
            ) -> Dict[str, Any]:

        return {
            'idx': idx,
            'class_label': self.run_dataset.iloc[idx]['class_label'],
            'class_id': self.run_dataset.iloc[idx]['class_id'],
            'seq_id': self.run_dataset.iloc[idx]['seq_id'],
            'path': self.ds_path / self.run_dataset.iloc[idx]['file_path'],
            'bbox': self.run_dataset.iloc[idx]['bbox'],
            'conf': self.run_dataset.iloc[idx]['conf']
            }

    def get_experiment_info(self) -> Dict:
        yaml_path = self.log_path / 'experiment_info.yaml'
        if not yaml_path.exists():
            raise FileNotFoundError(f"Experiment info file not found at {yaml_path}")
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def get_run_dataset(self) -> pd.DataFrame:
        csv_path = self.log_path / 'dataset.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {csv_path}")
        df = pd.read_csv(csv_path)
        return self._enforce_dtypes_and_idx(df)

    def get_predictions(
            self,
            fold: int | None = None,
            set_selection: list[str] | str | None = None,
            filter_by: str | None = None,
            sort: str | None = None
            ) -> pd.DataFrame:
        
        if set_selection:
            if isinstance(set_selection, str):
                set_selection = [set_selection]
            
            for set_sel in set_selection:
                if set_sel not in ['train', 'val', 'test']:
                    raise ValueError("set_selection must be 'train', 'val', or 'test'")

        prediction_path = self._handle_crossval_or_not('predictions', fold)

        df = pd.read_csv(prediction_path)

        df['correct'] = df['class_id'] == df['pred_id']

        df = self._enforce_dtypes_and_idx(df)
        
        if set_selection:
            mask = df['set'].isin(set_selection)
            df = df[mask]

        if 'probs' in df.columns:
            df['probs_max'] = [
                prob_list[pred]
                for prob_list, pred in zip(df['probs'], df['pred_id'])
            ]

        if filter_by:
            if filter_by == 'correct':
                df = df[df['correct']]
            elif filter_by == 'incorrect':
                df = df[~df['correct']]
            else:
                raise ValueError("filter_by must be either 'correct' or 'incorrect'")

        if sort:
            if sort == 'probs_max':
                df = df.sort_values(by='probs_max', ascending=False)

        return df

    def get_metrics(
            self,
            fold: int | None = None,
            ) -> pd.DataFrame:

        metrics_path = self._handle_crossval_or_not('metrics', fold)

        df = pd.read_csv(metrics_path)
        df = self._enforce_dtypes_and_idx(df)

        return df

    def _handle_crossval_or_not(
            self,
            type: None | str = None,
            fold: None | int = None,
            ) -> Path:

        options = {'metrics': 'metrics.csv', 'predictions': 'predictions.csv'}
        if type not in options:
            raise ValueError(f"Type must be one of {options.keys()}")

        if self.cross_val:
            if fold is None:
                raise ValueError("Fold number must be provided for cross-validation runs.")
            predictions_path = self.log_path / f'fold_{fold}' / options[type]
        else:
            predictions_path = self.log_path / options[type]

        if not predictions_path.exists():
            raise FileNotFoundError(f"{type} file not found at {predictions_path}")

        return predictions_path

    def _enforce_dtypes_and_idx(
            self,
            df: pd.DataFrame
            ) -> pd.DataFrame:
        df.insert(0, 'idx', df.index)
        cast_map = {
            'seq_id': 'int64',
            'class_id': 'int8',
            'fold': 'int8',
        }
        existing_casts = {k: v for k, v in cast_map.items() if k in df.columns}
        df = df.astype(existing_casts)

        def to_float_list(x):
            if isinstance(x, str):
                x = ast.literal_eval(x)
            return [float(i) for i in x]

        for col in ['bbox', 'probs']:
            if col in df.columns:
                df[col] = df[col].apply(to_float_list)

        return df
