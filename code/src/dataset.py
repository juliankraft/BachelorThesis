import os
import warnings
import csv
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, List, Tuple, Sequence

from megadetector.detection.run_detector import model_string_to_model_version

from os import PathLike
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from torchvision.transforms import v2

from ba_dev.runner import MegaDetectorRunner
from ba_dev.transform import ImagePipeline
from ba_dev.utils import best_weighted_split


class MammaliaData(Dataset):

    """

    A parent class to define the common methods and attributes for the different versions of Mammalia dataset.

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
            mode: str = 'train',
            ):

        if type(self) is MammaliaData:
            raise TypeError("MammaliaData is abstract and can't be instantiated directly.")

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

        labels = ['apodemus_sp', 'mustela_erminea', 'cricetidae', 'soricidae', 'other', 'glis_glis']
        self.categories_to_drop = categories_to_drop if categories_to_drop is not None else []
        if any(label not in labels for label in self.categories_to_drop):
            raise ValueError(f"Invalid categories to drop. Available categories: {labels}")
        self.class_labels = sorted([label for label in labels if label not in self.categories_to_drop])
        self.label_encoder = {label: idx for idx, label in enumerate(self.class_labels)}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}

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

        self.test_seq_ids, self.folds = self.custom_split_dataset(
                                            ds=self.ds_filtered,
                                            test_size=self.test_size,
                                            n_folds=self.n_folds,
                                            seed=self.random_seed
                                            )

        self.val_seq_ids = self.folds[self.val_fold]

        self.train_seq_ids = [
            seq_id
            for i, fold in enumerate(self.folds)
            if i != self.val_fold
            for seq_id in fold
            ]
        self.trainval_seq_ids = [
            seq_id
            for i, fold in enumerate(self.folds)
            for seq_id in fold
            ]

        if self.mode == 'train':
            self.ds = self.ds_filtered[self.ds_filtered['seq_id'].isin(self.train_seq_ids)].reset_index(drop=True)
        elif self.mode == 'val':
            self.ds = self.ds_filtered[self.ds_filtered['seq_id'].isin(self.val_seq_ids)].reset_index(drop=True)
        elif self.mode == 'test':
            self.ds = self.ds_filtered[self.ds_filtered['seq_id'].isin(self.test_seq_ids)].reset_index(drop=True)
        elif self.mode == 'init':
            self.ds = self.ds_filtered[self.ds_filtered['seq_id'].isin(self.trainval_seq_ids)].reset_index(drop=True)

    def custom_split_dataset(
            self,
            ds: pd.DataFrame,
            test_size: float,
            n_folds: int,
            seed: int,
            ) -> tuple[list[int], list[list[int]]]:

        rng = np.random.default_rng(seed)

        test_seq_ids = []
        fold_seq_ids = [[] for _ in range(n_folds)]

        for value in ds['class'].unique():
            ds_selected = ds[ds['class'] == value]
            length = ds_selected.shape[0]
            indices = rng.permutation(length)

            seq_ids = ds_selected['seq_id'].to_numpy()[indices]
            seq_lengths = ds_selected['n_files'].to_numpy()[indices]

            train_images = int(seq_lengths.sum() * test_size)
            fold_images = int(seq_lengths.sum() * (1 - test_size)) // n_folds
            split_sizes = [train_images] + [fold_images] * (n_folds)

            cut_idx_list = []
            seq_lengths_avail = seq_lengths.copy()

            for split_size in split_sizes:
                relative_cut_idx = best_weighted_split(seq_lengths_avail, split_size)

                seq_lengths_avail = seq_lengths_avail[relative_cut_idx:]

                used_idx = sum(cut_idx_list)
                absolute_cut_idx = relative_cut_idx + used_idx
                cut_idx_list.append(absolute_cut_idx)

            split_seq_ids = np.split(seq_ids, cut_idx_list)

            test_seq_ids.extend(split_seq_ids[0])

            for fold in range(n_folds):
                for fold in range(n_folds):
                    fold_seq_ids[fold].extend(split_seq_ids[fold + 1])

        return test_seq_ids, fold_seq_ids

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

        ds_full['class'] = ds_full['label2'].map(self.label_encoder)

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

    def explode_df(
            self,
            in_df: pd.DataFrame,
            only_one_bb_per_image: bool = True,
            ) -> pd.DataFrame:

        original_keys_to_keep = ['seq_id', 'class', 'label2', 'SerialNumber']

        out_rows = []

        for _, row in in_df.iterrows():

            used_files = set()

            bb_data = self.get_bb_list_for_seq(
                        seq_id=row['seq_id'],
                        confidence=self.applied_detection_confidence,
                        )

            row_info = {key: row[key] for key in original_keys_to_keep}
            directory = Path(row['Directory'])

            for file_name, bbox, conf in zip(bb_data['file'], bb_data['bbox'], bb_data['conf']):

                if only_one_bb_per_image and file_name in used_files:
                    continue

                used_files.add(file_name)

                new_row = row_info.copy()
                new_row['file_path'] = directory / file_name
                new_row['bbox'] = bbox
                new_row['conf'] = conf

                out_rows.append(new_row)

        return pd.DataFrame(out_rows).reset_index(drop=True)

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

    def get_class_weight(
            self,
            ) -> torch.Tensor:

        if self.mode != 'init':
            raise ValueError('Class weights can only be computed in init mode.')

        encoded_labels = self.ds['class'].to_numpy()
        classes = np.array(encoded_labels)

        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=encoded_labels
        )

        class_weights = torch.tensor(weights, dtype=torch.float32)

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

    def get_bb_list_for_seq(
            self,
            seq_id: int,
            confidence: float | None = None,
            ) -> dict[str, Sequence[Any]]:

        if confidence is None:
            confidence = self.applied_detection_confidence

        path_to_detection_results = self.path_to_detector_output / f"{seq_id}.json"
        with open(path_to_detection_results, 'r') as f:
            data = json.load(f)

        img_list = []
        bbox_list = []
        conf_list = []

        for entry in data:
            file_name = entry['file']
            detections = entry.get('detections', [])

            for det in detections:
                if det['category'] == "1" and det['conf'] >= confidence:
                    img_list.append(file_name)
                    bbox_list.append(det['bbox'])
                    conf_list.append(det['conf'])

        combined = list(zip(img_list, bbox_list, conf_list))
        combined_sorted = sorted(combined, key=lambda x: x[2], reverse=True)
        img_list, bbox_list, conf_list = zip(*combined_sorted)

        return {'file': img_list, 'bbox': bbox_list, 'conf': conf_list}


class MammaliaDataSequence(MammaliaData):

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
    mode : str
        The mode in which the dataset is used. Can be either 'train', 'test', 'val' or 'init' defining which data will
        be sampled and adjusting how it is sampled. The default is 'train'.

    """

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

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Any:

        # seq_id = self.seq_id_map[index]
        # row = self.ds[self.ds['seq_id'] == seq_id].squeeze()
        # detections = self.get_bb_list_for_seq(
        #     seq_id=seq_id,
        #     confidence=self.applied_detection_confidence
        #     )

        # # image_path_list = [row['Directory'] / name for name in detections['file']]
        # # bbox_list = detections['bbox']

        # x = 'not defined'
        # y = self.label_encoder[row['label2']]
        # return x, y

        pass


class MammaliaDataImage(MammaliaData):

    """

    A subclass of MammaliaData that is used to handle the data on a image level instead of a sequence level.

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
    mode : str
        The mode in which the dataset is used. Can be either 'train', 'test', 'val' or 'init' defining which data will
        be sampled and adjusting how it is sampled. The default is 'train'.
    transform : ImagePipeline (custom class)
        The transform to be applied to the images.

    """

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
            transform: ImagePipeline | None = None
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

        if transform is None:
            self.transform = ImagePipeline(
                pre_ops=[]
                )
        else:
            self.transform = transform
        if self.transform.path_to_dataset is None:
            self.transform.path_to_dataset = self.path_to_dataset

        self.ds = self.explode_df(
                in_df=self.ds,
                only_one_bb_per_image=True,
                )

        self.row_map = self.ds.index.tolist()

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index):
        row_index = self.row_map[index]
        row = self.ds.iloc[row_index]

        image_path = row['file_path']
        bbox = row['bbox']

        return self.transform(image_path, bbox)


class MammaliaDataFeatureStats(MammaliaDataImage):
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
            mode: str = 'init',
            transform: ImagePipeline | None = None
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
            transform=transform
        )

        self.image_pipline = ImagePipeline(
                path_to_dataset=self.path_to_dataset,
                pre_ops=[
                    ('to_rgb', {}),
                    ('crop_by_bb', {})
                ],
                transform=v2.Compose([
                                v2.ToImage(),
                                v2.ToDtype(torch.float32, scale=True),
                                ])
                )

    def __getitem__(self, index):
        row_index = self.row_map[index]
        row = self.ds.iloc[row_index]

        image_path = row['file_path']
        bbox = row['bbox']

        return self.image_pipline(image_path, bbox)
