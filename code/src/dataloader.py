import os
import warnings
import csv
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, Sequence, List, Tuple

from megadetector.detection.run_detector import model_string_to_model_version

from os import PathLike
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold

from ba_dev.runner import MegaDetectorRunner


class ImagePipeline:
    def __init__(
            self,
            path: str | PathLike,
            ):

        self.img: Image.Image = Image.open(path)

    def load(
            self,
            path: str | Path
            ):
        self.img = Image.open(path)
        return self

    def to_rgb(self):
        self.img = self.img.convert("RGB")
        return self

    def crop_by_bbox(
            self,
            bbox: list[float] | tuple[float, float, float, float]
            ):

        width, height = self.img.size

        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int((bbox[0] + bbox[2]) * width)
        y2 = int((bbox[1] + bbox[3]) * height)

        self.img = self.img.crop((x1, y1, x2, y2))
        self.img = self.img.crop((x1, y1, x2, y2))

    def crop_center_sample(
            self,
            bbox: list[float] | tuple[float, float, float, float],
            sample_size: int | Sequence[int] = (50, 50)
            ):

        if isinstance(sample_size, int):
            target_width = target_height = sample_size
        elif isinstance(sample_size, Sequence):
            if len(sample_size) == 1:
                target_width = target_height = sample_size[0]
            elif len(sample_size) == 2:
                target_width, target_height = sample_size
            else:
                raise ValueError("sample_size must be an int or a sequence of maximum two integers.")

        width, height = self.img.size

        center_x = int(bbox[0] * width) + int(bbox[2] * width / 2)
        center_y = int(bbox[1] * height) + int(bbox[3] * height / 2)

        x1 = center_x - (target_width // 2)
        y1 = center_y - (target_height // 2)
        x2 = x1 + target_width
        y2 = y1 + target_height

        self.img = self.img.crop((x1, y1, x2, y2))
        return self

    def resize(
            self,
            size: Sequence[int] | int,
            ):
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, Sequence):
            if len(size) == 1:
                size = (size[0], size[0])
            elif len(size) == 2:
                size = (size[0], size[1])
            else:
                raise ValueError("size must be an int or a sequence of maximum two integers.")
        self.img = self.img.resize(size)
        return self

    def get(self) -> Image.Image:
        return self.img


class MammaliaData(Dataset):
    """
    A class to load and process the Mammalia dataset. It can be used for the initial detection of the images
    utilizing the MegaDetector model, or for training a custom model for classification on the detected images.
    The dataset is divided into training and testing sets based on the sequence IDs.

    Parameters
    ----------
    path_labelfiles : str | PathLike
        Path to the directory containing the label files.
    path_to_dataset : str | PathLike
        Path to the main directory of the dataset, referenced in the labelfiles.
    path_to_detector_output : str | PathLike
        Path to the directory where the detector output is available for training or where the output will be saved
        if detection is applied.
    categories_to_drop : list[str], optional
        All empty labels are excluded anyways. Per default the categories 'other' and 'glis_glis' are excluded
        as well. This argument could change this behavior.
    detector_model : str
        If a detector model is provided, the detection will be applied to the whole dataset and stored for training.
        The model must be one of the available models in the MegaDetector repository.
        The default is None. A valid detection output must be available at the path_to_detector_output.
    applied_detection_confidence : float
        The detection is done with a confidence of 0.25 by default to provide some flexibility
        with the training. The confidence can be set to a higher value to reduce the number of detections used from
        the output. The default is 0.25.
    random_seed : int
        The seed used for the random number generator. The default is 55.
    test_size : float
        The proportion of the dataset to include in the test split. The default is 0.2.
    n_folds : int
        The number of folds to use for cross-validation. The default is 5.
    val_fold : int
        The fold index to use for validation. The default is 0.
    available_detection_confidence : float
        If the MD is applied, this is the minimal confidence to storred the output. If MD is not applied, this Value
        must be set to the value used for the detection. The default is 0.25.
    sample_length : int
        For training this parameter specifies the range (1 - sample_length) of randomly selected samples per sequence.
        For testing this parameter specifies the maximum number of samples per sequence.
        The default is 10.
    sample_img_size : [int, int]
        The size to which the detected areas are resized. The default is [224, 224].
    mode : str
        The mode in which the dataset is used. Can be either 'train', 'test', 'val' or 'init' defining which data will
        be sampled and adjusting how it is sampled. The default is 'train'.
    """

    def __init__(
            self,
            path_labelfiles: str | PathLike,
            path_to_dataset: str | PathLike,
            path_to_detector_output: str | PathLike,
            categories_to_drop: list[str] | None = ['other', 'glis_glis'],
            detector_model: str | None = None,
            applied_detection_confidence: float = 0.25,
            available_detection_confidence: float = 0.25,
            random_seed: int = 55,
            test_size: float = 0.2,
            n_folds: int = 5,
            val_fold: int = 0,
            sample_length: int = 10,
            sample_img_size: list[int] = [224, 224],
            mode: str = 'train',
            ):
        super().__init__()

        self.random_seed = random_seed
        self.test_size = test_size
        if n_folds <= val_fold:
            raise ValueError("The val_fold must be smaller than n_folds.")
        self.n_folds = n_folds
        self.val_fold = val_fold

        mode_available = ['train', 'test', 'val', 'init']
        if mode in mode_available:
            self.mode = mode
        else:
            raise ValueError(f'Please choose a mode from {mode_available}.')

        if applied_detection_confidence < available_detection_confidence:
            raise ValueError("The applied detection confidence can not be lower than the available one.")
        self.applied_detection_confidence = applied_detection_confidence
        self.available_detection_confidence = available_detection_confidence

        self.categories_to_drop = categories_to_drop if categories_to_drop is not None else []
        self.sample_length = sample_length
        self.sample_img_size = sample_img_size

        self.path_labelfiles = Path(path_labelfiles)
        if not self.path_labelfiles.exists():
            raise ValueError("The path to the label files does not exist.")

        self.path_to_dataset = Path(path_to_dataset)
        if not self.path_to_dataset.exists():
            raise ValueError("The path to the dataset does not exist.")

        self.path_to_detector_output = Path(path_to_detector_output)
        self.detector_model = detector_model

        if self.mode == 'init':
            if self.detector_model is not None:
                self.run_detector()
            else:
                if not any(self.path_to_detector_output.glob("*.json")):
                    raise ValueError('A valid detection output must be available at the path_to_detector_output.')

        self.ds_full = self.get_ds_full()
        self.ds_filtered = self.get_ds_filtered()

        self.trainval_seq_ids, self.test_seq_ids = self.split_dataset(
                                            seed=self.random_seed,
                                            test_size=self.test_size,
                                            )

        if self.__class__ == MammaliaData:

            trainval_set = self.ds_filtered[self.ds_filtered['seq_id'].isin(self.trainval_seq_ids)]

            self.trainval_folds = self.create_folds(
                                                seed=self.random_seed,
                                                n_folds=self.n_folds,
                                                ds=trainval_set
                                                )

            self.val_seq_ids = self.trainval_folds[self.val_fold]
            self.train_seq_ids = [
                    seq_id
                    for i, fold in enumerate(self.trainval_folds)
                    if i != self.val_fold
                    for seq_id in fold
                ]

            dataset = self.ds_filtered

            if self.mode == 'test':
                self.ds = dataset[dataset['seq_id'].isin(self.test_seq_ids)]
            elif self.mode == 'train':
                self.ds = dataset[dataset['seq_id'].isin(self.train_seq_ids)]
            elif self.mode == 'val':
                self.ds = dataset[dataset['seq_id'].isin(self.val_seq_ids)]
            elif self.mode == 'init':
                self.ds = dataset[dataset['seq_id'].isin(self.trainval_seq_ids)]

            self.seq_id_map = self.ds['seq_id'].tolist()

    def get_ds_full(
            self,
            ) -> pd.DataFrame:

        label_files = self.path_labelfiles.glob("*.csv")

        ds_full = pd.DataFrame()

        for file in label_files:
            ds_full = pd.concat([ds_full, pd.read_csv(file)], ignore_index=True)

        if ds_full['seq_id'].duplicated().any():
            duplicates = ds_full['seq_id'][ds_full['seq_id'].duplicated()].unique()
            raise ValueError(f"Duplicate seq_id(s) found in metadata: {duplicates[:5]} ...")

        return ds_full

    def get_ds_filtered(
            self,
            categories_to_drop: list[str] | None = None,  # if not provided it will use self.categories_to_drop
            drop_label2_nan: bool = True,
            exclude_no_detections_sequences: bool = True
            ) -> pd.DataFrame:

        if categories_to_drop is None:
            categories_to_drop = self.categories_to_drop

        ds_filtered = self.ds_full.copy()

        if drop_label2_nan:
            ds_filtered = ds_filtered.dropna(subset=['label2'])

        ds_filtered = ds_filtered[~ds_filtered['label2'].isin(categories_to_drop)]

        if exclude_no_detections_sequences:
            detect_seq_ids, no_detect_seq_ids = self.check_seq_for_detections(
                sequences_to_filter=ds_filtered['seq_id'].tolist(),
                detection_confidence=self.applied_detection_confidence
                )

            if len(no_detect_seq_ids) > 0:
                suffix = "" if len(no_detect_seq_ids) <= 10 else " ..."
                warnings.warn(
                    f"With the detection confidence of {self.applied_detection_confidence},\n"
                    f"{len(no_detect_seq_ids)} sequences had no detections and will be excluded.\n"
                    f"Excluded sequences: {no_detect_seq_ids[:10]}{suffix}",
                    UserWarning
                )

            ds_filtered = ds_filtered[ds_filtered['seq_id'].isin(detect_seq_ids)]

        return ds_filtered

    def split_dataset(
            self,
            seed: int,
            test_size: float
                ) -> Tuple[List[int], List[int]]:

        ds = self.ds_filtered

        trainval_seq_ids, test_seq_ids = train_test_split(
                                            ds['seq_id'].tolist(),
                                            test_size=test_size,
                                            random_state=seed,
                                            stratify=ds['label2'].tolist()
                                            )

        return trainval_seq_ids, test_seq_ids

    def create_folds(
            self,
            seed: int,
            n_folds: int,
            ds: pd.DataFrame | None = None
            ) -> List[List[int]]:

        if ds is None:
            ds = self.ds_filtered

        ds_trainval = ds[ds['seq_id'].isin(self.trainval_seq_ids)]

        seq_label_df = (
            ds_trainval.groupby('seq_id')
            .first()[['label2']]
            .reset_index()
        )

        seq_ids = seq_label_df['seq_id'].tolist()
        labels = seq_label_df['label2'].tolist()

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        folds: List[List[int]] = []
        for _, val_idx in skf.split(seq_ids, labels):
            val_fold_ids = [seq_ids[i] for i in val_idx]
            folds.append(val_fold_ids)

        return folds

    def get_all_files_of_type(
            self,
            path: str | PathLike,
            file_type: str,
            get_full_path: bool = True
            ) -> list[str | PathLike]:

        path = Path(path)
        files = []
        for file in os.listdir(path):
            if file.endswith(file_type):
                if get_full_path:
                    files.append(path / file)
                else:
                    files.append(file)
        return files

    def reading_all_metadata(
            self,
            list_of_files: list[PathLike | str],
            categories_to_drop: list[str] | None = None,
            ) -> pd.DataFrame:

        if categories_to_drop is None:
            categories_to_drop = []

        metadata = pd.DataFrame()
        for file in list_of_files:
            metadata = pd.concat([metadata, pd.read_csv(file)], ignore_index=True)
            metadata = metadata.dropna(subset=['label2'])
            metadata = metadata[~metadata['label2'].isin(categories_to_drop)]
        return metadata

    def check_seq_for_detections(
            self,
            sequences_to_filter: list[int],
            detection_confidence: float,
            ) -> Tuple[List[int], List[int]]:

        detection_summary = self.get_detection_summary(
            usecols=["seq_id", "max_conf"]
            )

        seq_ids_to_exclude_set = set(
            detection_summary[detection_summary["max_conf"] < detection_confidence]["seq_id"].tolist()
            )
        seq_ids_to_filter_set = set(sequences_to_filter)

        no_detect_seq_ids = list(seq_ids_to_filter_set & seq_ids_to_exclude_set)

        detect_seq_ids = list(seq_ids_to_filter_set - seq_ids_to_exclude_set)

        return detect_seq_ids, no_detect_seq_ids

    def get_detection_summary(
            self,
            usecols: list[str] | None = None,
            ) -> pd.DataFrame:

        return pd.read_csv(
                self.path_to_detector_output / "detection_summary.csv",
                usecols=usecols
                )

    def get_class_weight(                                        # still to be implemented
            self
            ) -> torch.Tensor:

        class_weights = torch.Tensor()

        return class_weights

    def get_all_images_of_sequence(
            self,
            seq_id: int,
            dataframe: pd.DataFrame | None = None,
            ) -> dict[str, PathLike]:

        if dataframe is None:
            dataframe = self.get_ds_full()

        try:
            row = dataframe.loc[dataframe['seq_id'] == seq_id].iloc[0]
        except IndexError:
            raise ValueError(f"No sequence with seq_id={seq_id} found.")

        seq_path = Path(row['Directory'])
        all_files = row['all_files'].split(",")

        return {
            file_name: self.path_to_dataset / seq_path / file_name
            for file_name in all_files
            }

    def run_detector(
            self,
            ) -> None:

        if self.detector_model is None:
            raise ValueError('Method not available - No detector model provided.')
        elif self.detector_model not in model_string_to_model_version.keys():
            raise ValueError(
                f"The model {self.detector_model} is not supported. "
                f"Please choose from {model_string_to_model_version.keys()}."
            )
        elif not self.path_to_detector_output.exists():
            os.makedirs(self.path_to_detector_output)
        elif any(self.path_to_detector_output.iterdir()):
            raise ValueError("The path to the detector output contains files. Please clear or choose a different path.")

        runner = MegaDetectorRunner(
            model_path=self.detector_model,
            confidence=0.25
            )

        metadata = self.get_ds_full()

        sequences = metadata['seq_id'].tolist()

        detection_rows = []

        for seq_id in sequences:
            seq_images = list(self.get_all_images_of_sequence(seq_id, dataframe=metadata).values())
            output_file_path = self.path_to_detector_output / f"{seq_id}.json"
            detections = runner.run_on_images(
                images=seq_images,
                output_file_path=output_file_path
                )

            detection_row = {
                    "seq_id": seq_id,
                    "max_conf": max(detections) if len(detections) > 0 else 0,
                    "n_detections": len(detections),
                    "conf_list": json.dumps(detections)
                }

            detection_rows.append(detection_row)

        all_detections = pd.DataFrame(detection_rows, columns=["seq_id", "max_conf", "n_detections", "conf_list"])

        all_detections.to_csv(
            self.path_to_detector_output / "detection_summary.csv",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC
            )

    def getting_bb_list_for_seq(
            self,
            seq_id: int,
            confidence: float | None = None,
            ) -> list[dict]:

        if confidence is None:
            confidence = self.applied_detection_confidence

        path_to_detection_results = self.path_to_detector_output / f"{seq_id}.json"
        with open(path_to_detection_results, 'r') as f:
            data = json.load(f)

        bb_list = []

        for entry in data:
            file_name = entry['file']
            detections = entry.get('detections', [])

            for det in detections:
                if det['category'] == "1" and det['conf'] >= confidence:
                    bb_list.append({
                        'file': file_name,
                        'conf': det['conf'],
                        'bbox': det['bbox']
                    })

        bb_list = sorted(bb_list, key=lambda x: x['conf'], reverse=True)

        return bb_list

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Any:

        seq_id = self.seq_id_map[index]
        row = self.ds[self.ds['seq_id'] == seq_id].squeeze()

        return row


class MammaliaDataImage(MammaliaData):
    def __init__(
            self,
            path_labelfiles: str | PathLike,
            path_to_dataset: str | PathLike,
            path_to_detector_output: str | PathLike,
            detector_model: str | None = None,
            applied_detection_confidence: float = 0.25,
            available_detection_confidence: float = 0.25,
            random_seed: int = 55,
            test_size: float = 0.2,
            n_folds: int = 5,
            val_fold: int = 0,
            mode: str = 'train',
            ):
        super().__init__(
            path_labelfiles=path_labelfiles,
            path_to_dataset=path_to_dataset,
            path_to_detector_output=path_to_detector_output,
            detector_model=detector_model,
            applied_detection_confidence=applied_detection_confidence,
            available_detection_confidence=available_detection_confidence,
            random_seed=random_seed,
            test_size=test_size,
            n_folds=n_folds,
            val_fold=val_fold,
            mode=mode,
            )

        self.ds_exploded = self.explode_df(
                in_df=self.ds_filtered,
                only_one_bb_per_image=True,
                )

        trainval_set = self.ds_exploded[self.ds_exploded['seq_id'].isin(self.trainval_seq_ids)]
        test_set = self.ds_exploded[self.ds_exploded['seq_id'].isin(self.test_seq_ids)]

        self.trainval_folds = self.create_folds(
                                        seed=self.random_seed,
                                        n_folds=self.n_folds,
                                        ds=trainval_set.reset_index(drop=True)
                                        )

        self.val_indices = self.trainval_folds[self.val_fold]
        self.train_indices = [
            idx
            for i, fold in enumerate(self.trainval_folds)
            if i != self.val_fold
            for idx in fold
            ]

        dataset = self.ds_exploded
        if self.mode == 'test':
            self.ds = dataset[dataset['seq_id'].isin(self.test_seq_ids)]
        elif self.mode == 'train':
            self.ds = trainval_set.iloc[self.train_indices]
        elif self.mode == 'val':
            self.ds = trainval_set.iloc[self.val_indices]
        elif self.mode == 'init':
            self.ds = trainval_set

        if self.mode == 'test':
            self.ds = test_set.reset_index(drop=True)
        elif self.mode == 'train':
            self.ds = trainval_set.iloc[self.train_indices].reset_index(drop=True)
        elif self.mode == 'val':
            self.ds = trainval_set.iloc[self.val_indices].reset_index(drop=True)
        elif self.mode == 'init':
            self.ds = trainval_set.reset_index(drop=True)

        self.row_map = self.ds.index.tolist()

    def create_folds(
            self,
            seed: int,
            n_folds: int,
            ds: pd.DataFrame | None = None
            ) -> List[List[int]]:

        if ds is None:
            ds = self.ds_exploded

        labels = ds['label2'].tolist()
        indices = np.arange(len(ds))

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        folds: List[List[int]] = []
        for _, val_idx in skf.split(indices, labels):
            folds.append(val_idx.tolist())

        return folds

    def explode_df(
            self,
            in_df: pd.DataFrame,
            only_one_bb_per_image: bool = True,
            ) -> pd.DataFrame:

        original_keys_to_keep = ['seq_id', 'label2', 'SerialNumber']

        out_rows = []

        for i, row in in_df.iterrows():

            used_files = set()

            bb_list = self.getting_bb_list_for_seq(
                        seq_id=row['seq_id'],
                        confidence=self.applied_detection_confidence,
                        )

            row_info_to_add = {key: row[key] for key in original_keys_to_keep}

            for item in bb_list:

                file_name = item['file']

                if only_one_bb_per_image and file_name in used_files:
                    continue

                used_files.add(file_name)

                new_row = row_info_to_add.copy()

                new_row['file_path'] = Path(row['Directory']) / file_name
                new_row['bbox'] = item['bbox']
                new_row['conf'] = item['conf']

                out_rows.append(new_row)

        return pd.DataFrame(out_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: int) -> Any:
        row_index = self.row_map[index]
        row = self.ds.iloc[row_index]

        return row
