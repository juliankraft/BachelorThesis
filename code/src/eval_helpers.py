
import ast
import os
import json
import yaml
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import matplotlib.patches as patches

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
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
        bbox: BBox | None = None,
        conf: float | None = None) -> Figure:

    """
    Plot an image with a bounding box if a BBox is provided.
    Args:
        image (Image.Image): The image to plot.
        bbox (BBox): The bounding box to plot, in the format [x, y, w, h].
        conf (float | None): The confidence score to display. If None, no score is displayed.
    Returns:
        Figure: The matplotlib figure object.
    """

    fig, ax = plt.subplots()
    ax.imshow(image)

    if bbox:
        width, height = image.size

        x_abs = bbox[0] * width
        y_abs = bbox[1] * height
        w_abs = bbox[2] * width
        h_abs = bbox[3] * height

        if w_abs > 0 and h_abs > 0:
            rect = patches.Rectangle(
                (x_abs, y_abs), w_abs, h_abs,
                linewidth=1, edgecolor='red', facecolor='none'
            )

        ax.add_patch(rect)

        if conf is not None and conf > 0:
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
        sample: Dict,
        annotation_type: str = 'detection',
        offset_margin_conf_annotation: int = 100
        ) -> None:

    img = sample['img']
    bbox = sample.get('bbox', None)

    ax.imshow(img)
    draw_bbox = False

    if bbox is not None and (bbox[2] > 0 and bbox[3] > 0):
        draw_bbox = True

    if draw_bbox and bbox is not None:
        width, height = img.size
        x_abs, y_abs = bbox[0] * width, bbox[1] * height
        w_abs, h_abs = bbox[2] * width, bbox[3] * height

        rect = patches.Rectangle(
            (x_abs, y_abs), w_abs, h_abs,
            linewidth=1, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        if annotation_type != 'none':

            label_up = 8

            if annotation_type == 'detection':
                try:
                    annotation_string = f"conf = {sample['conf']:.2f}"
                except KeyError:
                    print("'conf' not found in sample.")
                    annotation_string = None

            if annotation_type == 'classification':
                try:
                    annotation_string = f"{sample['pred_label']}: {sample['probs_max']:.2f}"
                except KeyError:
                    print("'pred_label' or 'probs_max' not found in sample.")
                    annotation_string = None
            if annotation_type == 'both':
                try:
                    annotation_string = (
                        f"detection: {sample['conf']:.2f}\n"
                        f"{sample['pred_label']}: {sample['probs_max']:.2f}"
                    )
                except KeyError:
                    print("'conf', 'pred_label' or 'probs_max' not found in sample.")
                    annotation_string = None

                label_up = 16

            if x_abs + offset_margin_conf_annotation > width:
                ha, offset = 'right', (-2, label_up)
                x = x_abs + w_abs
            else:
                ha, offset = 'left', (2, label_up)
                x = x_abs

            if annotation_string:
                ax.annotate(
                    annotation_string,
                    xy=(x, y_abs),
                    xytext=offset,
                    textcoords="offset points",
                    ha=ha, va="top",
                    fontsize=8, color="white",
                    bbox=dict(facecolor="red", alpha=0.5, edgecolor="none", pad=1.5),
                    clip_on=False
                    )

    ax.axis('off')


def plot_series_of_images(
        df: pd.DataFrame,
        dataset_path: PathLike | str,
        annotation_type: str = 'detection',
        ncols: int = 3,
        fig_width_cm: float = 24,
        offset_margin_conf_annotation: int = 100
        ) -> Figure:

    dataset_path = Path(dataset_path)

    if annotation_type not in ['detection', 'classification', 'both', 'none']:
        raise ValueError("Type must be 'detection', 'classification', 'both' or 'none'.")

    fig_width = fig_width_cm / 2.54
    nrows = (len(df) + ncols - 1) // ncols
    labels = list(string.ascii_lowercase)

    fig = plt.figure(figsize=(fig_width, fig_width * nrows / ncols * 3/4))
    gs = GridSpec(
        nrows=nrows,
        ncols=ncols,
        figure=fig,
        )

    for idx, (_, row) in enumerate(df.iterrows()):

        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])

        file_path = dataset_path / row['file_path']

        sample = row.to_dict()

        sample['img'] = Image.open(file_path)

        ax.annotate(
            f'({labels[idx]})',
            xy=(0.01, 0.98),
            xycoords=ax.transAxes,
            fontsize=10,
            color='red',
            ha='left',
            va='top',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1.5)
            )

        draw_bbox_on_ax(
            ax=ax,
            sample=sample,
            annotation_type=annotation_type,
            offset_margin_conf_annotation=offset_margin_conf_annotation
            )

    plt.tight_layout()

    plt.close(fig)

    return fig


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

    smoothed_val_loss = smooth_data(validation_data[loss], window_size=window_size)
    smoothed_val_acc = smooth_data(validation_data[acc], window_size=window_size)

    epochs = validation_data['epoch'][len(validation_data['epoch']) - len(smoothed_val_loss):]

    fig, ax1 = plt.subplots(figsize=(image_size_cm[0]/2.54, image_size_cm[1]/2.54))

    if title:
        plt.title(title)

    ax1.plot(validation_data['epoch'], validation_data[loss], color='#A6CEE3', label='', linewidth=1, alpha=0.5)
    ax1.plot(epochs, smoothed_val_loss, color='#1F78B4', label=label[0], linewidth=0.5)

    loss_min = float(validation_data[loss].min())
    loss_max = float(validation_data[loss].max())
    ax1.set_ylim(loss_min * 0.87, loss_max * 1)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(label[0])
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(validation_data['epoch'], validation_data[acc], color='#FB9A99', label='', linewidth=1, alpha=0.5)
    ax2.plot(epochs, smoothed_val_acc, color='#E31A1C', label=label[1], linewidth=0.5)
    ax2.set_ylabel(label[1])
    ax2.tick_params(axis='y')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower center', bbox_to_anchor=(0.5, 0.2), ncol=2)

    fig.tight_layout()

    return fig


def find_nested_diffs(dicts, parent_key=()):

    if not dicts:
        return {}

    if not all(isinstance(d, dict) for d in dicts):
        vals = [(i, d) for i, d in enumerate(dicts)]
        first_val = dicts[0]
        if any(d != first_val for d in dicts[1:]):
            return {parent_key: vals}
        else:
            return {}

    all_keys = set().union(*dicts)
    diffs = {}

    for key in all_keys:
        subset = [d.get(key) for d in dicts]
        branch = find_nested_diffs(subset, parent_key + (key,))
        diffs.update(branch)

    return diffs


def evaluate_all_runs(
        path_to_runs: PathLike,
        metrics: str | list[str] | dict[str, dict]
        ) -> pd.DataFrame:

    path_to_runs = Path(path_to_runs)
    run_paths = list(path_to_runs.glob('*/'))

    if isinstance(metrics, dict):
        metric_items = list(metrics.items())
    else:
        if isinstance(metrics, str):
            metrics = [metrics]
        metric_items = [(m, {}) for m in metrics]

    all_items = []

    for run_path in run_paths:

        model = LoadRun(log_path=run_path)

        for metric, m_kwargs in metric_items:

            img_scores = model.calculate_metrics(
                metric=metric,
                set_selection='test',
                scope='img',
                **m_kwargs
                )

            seq_scores = model.calculate_metrics(
                metric=metric,
                set_selection='test',
                scope='seq',
                **m_kwargs
                )

            if not isinstance(img_scores, list):
                img_scores = [img_scores]
                seq_scores = [seq_scores]

            for fold_idx, (img, seq) in enumerate(zip(img_scores, seq_scores)):
                all_items.append({
                    'model_name': model.info['model']['backbone_name'],
                    'pretrained': model.info['model']['backbone_pretrained'],
                    'experiment_name': model.info['experiment_name'],
                    'trainable_params': model.info['output']['model_parameters']['trainable'],
                    'metric': metric,
                    'fold': fold_idx,
                    'img_score': img,
                    'seq_score': seq,
                    })

    return pd.DataFrame(all_items)


def place_table(
        latex_table: str,
        center: bool = True,
        placement: str | None = None,
        ) -> str:

    lines = latex_table.splitlines()

    if placement:
        lines[0] = f'\\begin{{table}}[{placement}]'

    if center:
        lines.insert(1, r'\centering')

    return '\n'.join(lines)


def file_count(
        df: pd.DataFrame,
        threshold: float
        ):

    keys = df.columns.tolist()
    if 'max_conf' in keys:
        apply_to = 'max_conf'
    elif 'conf' in keys:
        apply_to = 'conf'
    else:
        raise ValueError("DataFrame must contain 'max_conf' or 'conf' column.")

    counts = (
        df[df[apply_to] >= threshold]
        .groupby('class_label')
        .size()
        .reset_index(name='count')
        )

    return counts


def count_vs_threshold(df, thresholds=(0.25, 0.5, 0.75), base_name="all"):

    total = f'{base_name}_avail'
    result = file_count(df, 0).rename(columns={'count': total})

    for t in thresholds:
        avail = file_count(df, t)['count']
        lost = result[total] - avail
        frac = (lost / result[total]).round(2)

        result[f'{base_name}_lost_{t}'] = lost
        result[f'{base_name}_frac_{t}'] = frac
        result[f'{base_name}_avail_{t}'] = avail

    return result


def get_md_info(
        seq_id: int,
        files_list: list,
        md_output_dir: Path
        ) -> tuple[list[float], list[list[float]], float]:

    json_path = os.path.join(md_output_dir, f"{seq_id}.json")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

    except FileNotFoundError:
        print(f"JSON file for sequence {seq_id} not found at {json_path}.")
        data = {}

    lookup = {item['file']: item.get('detections', []) for item in data}

    conf_values: list[float] = []
    bboxes: list[list[float]] = []

    for fname in files_list:
        dets = lookup.get(fname, [])

        if dets:
            best = max(dets, key=lambda d: d.get('conf', 0))
            conf = best.get('conf', 0.0)
            bbox = best.get('bbox', [0.0, 0.0, 0.0, 0.0])
        else:
            conf = 0.0
            bbox = [0.0, 0.0, 0.0, 0.0]

        conf_values.append(conf)
        bboxes.append(bbox)

    max_conf = max(conf_values) if conf_values else 0.0

    return conf_values, bboxes, max_conf


class LoadRun:
    def __init__(
            self,
            log_path: str | PathLike,
            paths: Dict | None = None
            ):

        self.log_path = Path(log_path)
        self.info = self.get_experiment_info()
        self.cross_val = self.info['cross_val']['apply']
        if self.cross_val:
            self.folds = range(2 if self.info.get('dev_run', False) else self.info['cross_val']['n_folds'])
        else:
            self.folds = [None]

        if paths:
            self.path_dataset = Path(paths['dataset'])
            self.path_md_output = Path(paths['md_output'])
            self.path_labels = Path(paths['labels'])
        else:
            self.path_dataset = Path(self.info['paths']['dataset'])
            self.path_md_output = Path(self.info['paths']['md_output'])
            self.path_labels = Path(self.info['paths']['labels'])

        self.decoder = self.info['output']['label_decoder']

        self.run_dataset = self.get_run_dataset()

        self.full_predicted_set = self.get_full_predicted_set()

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

        path_to_detection_results = self.path_md_output / f"{seq_id}.json"
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
            path_labelfiles=self.path_labels,
            path_to_dataset=self.path_dataset,
            path_to_detector_output=self.path_md_output,
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
            scope: str = 'img',
            **kwargs
            ):

        if scope == 'img':
            get_df = self.get_predictions
        elif scope == 'seq':
            get_df = self.get_predictions_for_seq
        else:
            raise ValueError("Scope must be either 'img' or 'seq'")

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
                df_pred = get_df(fold=fold, set_selection=set_selection)
                results.append(compute(df_pred))
            return results
        else:
            df_pred = get_df(fold=None, set_selection=set_selection)
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
            'path': self.path_dataset / self.run_dataset.iloc[idx]['file_path'],
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

    def get_full_predicted_set(
            self
            ) -> pd.DataFrame:

        if not self.folds:
            raise ValueError("This method is only applicable for cross-validation runs.")
        else:

            all_testsets = pd.DataFrame()

            for fold in self.folds:
                new_data = self.get_predictions(
                    set_selection='test',
                    fold=fold
                    )

                new_data['fold'] = fold

                all_testsets = pd.concat([all_testsets, new_data], ignore_index=True)

            common_cols = list(set(all_testsets.columns).intersection(self.run_dataset.columns) - {'idx'})
            all_testsets_pruned = all_testsets.drop(columns=common_cols)

            full_predicted_set = self.run_dataset.merge(all_testsets_pruned, on='idx')

            return full_predicted_set

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
        ds = self.get_run_dataset()
        df['seq_id'] = ds['seq_id']

        df['correct'] = df['class_id'] == df['pred_id']
        df['pred_label'] = df['pred_id'].map(self.decoder)

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

    def get_predictions_for_seq(
            self,
            fold: int | None = None,
            set_selection: list[str] | str | None = None,
            ) -> pd.DataFrame:

        df = self.get_predictions(
                fold=fold,
                set_selection=set_selection
                )

        def agg_probs(ps):
            summed = [sum(col) for col in zip(*ps)]
            total = sum(summed)
            return [v/total for v in summed]

        aggregated = (
            df
            .groupby('seq_id')
            .agg(
                class_id=('class_id', 'first'),
                set=('set', 'first'),
                count=('pred_id', 'size'),
                pred_id_majority=('pred_id', lambda x: x.mode()),
                probs=('probs',   agg_probs)
            )
            .reset_index()
        )

        aggregated['prob_max'] = aggregated['probs'].apply(max)
        aggregated['pred_id'] = aggregated['probs'].apply(lambda p: p.index(max(p)))

        return aggregated

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


class DFChunker:
    def __init__(
            self,
            df: pd.DataFrame,
            chunk_size: int
            ):
        """
        Class to slice a DataFrame into chunks of a specified size.
        Args:
        df: the DataFrame to slice into chunks
        chunk_size: number of rows per chunk
        """
        self.df = df.reset_index(drop=True)
        self.chunk_size = chunk_size
        self.n_chunks = (len(self.df) + chunk_size - 1) // chunk_size
        self.current = 0

        print(f"DataFrame has {len(self.df)} rows,")
        print(f"will be split into {self.n_chunks} chunks of size {self.chunk_size}.")

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> pd.DataFrame:
        if idx < 0:
            idx = self.n_chunks + idx
        if not (0 <= idx < self.n_chunks):
            raise IndexError(f"Chunk index {idx} out of range [0..{self.n_chunks-1}]")

        start = idx * self.chunk_size
        end = start + self.chunk_size
        return self.df.iloc[start:end]

    def get_current(self) -> pd.DataFrame:
        return self[self.current]

    def get_current_and_advance(self) -> pd.DataFrame:
        if self.current >= self.n_chunks:
            raise StopIteration("No more chunks available.")
        chunk = self[self.current]
        self.current += 1
        return chunk

    def reset(self) -> None:
        """Reset the cursor back to the first chunk."""
        self.current = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self) -> pd.DataFrame:
        return self.get_current_and_advance()

    def __repr__(self):
        return (
            f"<DFChunker chunks={self.n_chunks}, size={self.chunk_size}, cursor={self.current}>"
        )
