import os
import warnings
import csv
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, List, Tuple, Sequence, TypedDict

from megadetector.detection.run_detector import model_string_to_model_version

from os import PathLike
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

from ba_dev.runner import MegaDetectorRunner
from ba_dev.transform import ImagePipeline, BatchImagePipeline
from ba_dev.utils import best_weighted_split, BBox


class DetectionResult(TypedDict):
    file: list[str]
    bbox: list[Sequence[float]]
    conf: list[float]


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
            extra_test_set: float | None = None,
            n_folds: int = 5,
            test_fold: int = 0,
            mode: str = 'train',
            image_pipeline: ImagePipeline | None = None,
            ):

        if type(self) is MammaliaData:
            raise TypeError("MammaliaData is abstract and can't be instantiated directly.")

        super().__init__()

        self.image_pipeline = image_pipeline

        self.random_seed = random_seed
        self.extra_test_set = extra_test_set
        if n_folds <= test_fold:
            raise ValueError("The test_fold must be smaller than n_folds.")
        self.n_folds = n_folds
        self.test_fold = test_fold
        self.val_fold = (test_fold + 1) % n_folds

        mode_available = ['train', 'test', 'val', 'pred', 'init', 'eval']
        if mode in mode_available:
            self.mode = mode
        else:
            raise ValueError(f'Please choose a mode from {mode_available}.')

        if applied_detection_confidence < available_detection_confidence:
            raise ValueError("The applied detection confidence can not be lower than the available one.")
        self.applied_detection_confidence = applied_detection_confidence
        self.available_detection_confidence = available_detection_confidence

        labels = ['apodemus_sp', 'mustela_erminea', 'cricetidae', 'soricidae', 'glis_glis', 'other']
        self.categories_to_drop = categories_to_drop if categories_to_drop is not None else []
        if any(label not in labels for label in self.categories_to_drop):
            raise ValueError(f"Invalid categories to drop. Available categories: {labels}")

        kept_labels = [label for label in labels if label not in self.categories_to_drop]
        dropped_labels = [label for label in labels if label in self.categories_to_drop]

        all_labels = kept_labels + dropped_labels

        self.class_labels = kept_labels
        self.num_classes = len(kept_labels)

        self.complete_label_encoder = {label: idx for idx, label in enumerate(all_labels)}

        self.label_encoder = {label: idx for idx, label in enumerate(kept_labels)}
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
        self.ds_filtered, self.no_detect_seq_ids = self.get_ds_filtered()

        self.separate_test_seq_ids, self.folds = self.custom_split(
                                            ds=self.ds_filtered,
                                            test_size=self.extra_test_set,
                                            n_folds=self.n_folds,
                                            seed=self.random_seed
                                            )

        self.val_seq_ids = self.folds[self.val_fold]
        self.test_seq_ids = self.folds[self.test_fold]

        self.train_seq_ids = [
            seq_id
            for i, fold in enumerate(self.folds)
            if i != self.test_fold and i != self.val_fold
            for seq_id in fold
            ]

        self.trainval_seq_ids = [
            seq_id
            for i, fold in enumerate(self.folds)
            if i != self.test_fold
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
        elif self.mode in ['eval', 'pred']:
            self.ds = self.ds_filtered

    def custom_split(
            self,
            ds: pd.DataFrame,
            test_size: float | None,
            n_folds: int,
            seed: int,
            ) -> tuple[list[int], list[list[int]]]:

        rng = np.random.default_rng(seed)

        if test_size is not None and not (0.0 < test_size < 1.0):
            raise ValueError("test_size must be between 0 and 1, or None.")

        test_seq_ids: list[int] = []
        fold_seq_ids: list[list[int]] = [[] for _ in range(n_folds)]

        for selected_class in ds['class_id'].unique():
            ds_selected = ds[ds['class_id'] == selected_class]
            permutation = rng.permutation(len(ds_selected))
            seq_ids = ds_selected['seq_id'].to_numpy()[permutation]
            seq_lengths = ds_selected['n_files'].to_numpy()[permutation]

            if test_size is not None:
                n_test_images = int(seq_lengths.sum() * test_size)
                cut_idx = best_weighted_split(seq_lengths, n_test_images)
                if cut_idx == 0:
                    raise ValueError(f'The test set is not containing images for the class {selected_class}.')
                test_seq_ids.extend(seq_ids[:cut_idx])
                seq_ids = seq_ids[cut_idx:]
                seq_lengths = seq_lengths[cut_idx:]

            fold_size = seq_lengths.sum() / n_folds

            avail_ids = seq_ids
            avail_lengths = seq_lengths

            for i in range(n_folds - 1):
                cut_idx = best_weighted_split(avail_lengths, fold_size)
                if cut_idx == 0 or cut_idx == len(avail_lengths):
                    raise ValueError(f'Not all folds will contain samples of class_id: {selected_class}.')
                fold_seq_ids[i].extend(avail_ids[:cut_idx])
                avail_ids = avail_ids[cut_idx:]
                avail_lengths = avail_lengths[cut_idx:]

            fold_seq_ids[-1].extend(avail_ids)

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

        ds_full['class_label'] = ds_full['label2']
        ds_full['class_id'] = ds_full['class_label'].map(self.complete_label_encoder).fillna(-1).astype(int)

        ds_full = ds_full.drop(columns=['label', 'label2', 'duplicate_label'], errors='ignore')

        return ds_full

    def get_ds_filtered(
            self,
            categories_to_drop: list[str] | None = None,  # if not provided it will use self.categories_to_drop
            drop_nan: bool = True,
            exclude_no_detections_sequences: bool = True
            ) -> tuple[pd.DataFrame, list[int]]:

        if categories_to_drop is None:
            categories_to_drop = self.categories_to_drop

        ds_filtered = self.ds_full.copy()

        if drop_nan:
            ds_filtered = ds_filtered.dropna(subset=['class_label'])

        mask = ~ds_filtered['class_label'].isin(categories_to_drop)
        ds_filtered = ds_filtered.loc[mask]

        if exclude_no_detections_sequences:
            detect_seq_ids, no_detect_seq_ids = self.check_seq_for_detections(
                sequences_to_filter=ds_filtered['seq_id'].tolist(),
                detection_confidence=self.applied_detection_confidence
                )

            if len(no_detect_seq_ids) > 0 and self.mode in ['init', 'eval']:
                suffix = "" if len(no_detect_seq_ids) <= 10 else " ..."
                warnings.warn(
                    f"With the detection confidence of {self.applied_detection_confidence},\n"
                    f"{len(no_detect_seq_ids)} sequences had no detections and will be excluded.\n"
                    f"Excluded sequences: {no_detect_seq_ids[:10]}{suffix}",
                    UserWarning
                )

            ds_filtered = ds_filtered[ds_filtered['seq_id'].isin(detect_seq_ids)]

        return ds_filtered, no_detect_seq_ids

    def explode_df(
            self,
            in_df: pd.DataFrame,
            only_one_bb_per_image: bool = True,
            ) -> pd.DataFrame:

        original_keys_to_keep = ['seq_id', 'class_id', 'class_label', 'SerialNumber', 'n_files']

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

    def get_ds_with_folds(self) -> pd.DataFrame:
        """
        Return a copy of self.ds with an extra column 'fold':
         - = -1 for seq_ids in the separate_test_seq_ids
         - = 0,1,… for the other folds as per self.folds
        """
        fold_map: dict[int, int] = {
            seq_id: -1 for seq_id in self.separate_test_seq_ids
        }
        for fold_idx, seq_ids in enumerate(self.folds):
            for seq_id in seq_ids:
                if seq_id not in fold_map:
                    fold_map[seq_id] = fold_idx
        df = self.ds.copy()
        df['fold'] = df['seq_id'].map(fold_map).fillna(-1).astype(int)
        return df

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

    def get_class_weights(
            self,
            ) -> torch.Tensor:

        ds = self.ds_filtered[self.ds_filtered['seq_id'].isin(self.train_seq_ids)].reset_index(drop=True)
        y = ds['class_id'].to_numpy()
        classes = np.unique(y)

        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y
        )

        class_weights = torch.tensor(weights, dtype=torch.float32)

        return class_weights

    def get_num_classes(self) -> int:
        return self.num_classes

    def get_label_encoder(self) -> dict[str, int]:
        return self.label_encoder

    def get_label_decoder(self) -> dict[int, str]:
        return self.label_decoder

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
            only_one_bb_per_image: bool = False,
            ) -> DetectionResult:

        if confidence is None:
            confidence = self.applied_detection_confidence

        path_to_detection_results = self.path_to_detector_output / f"{seq_id}.json"
        with open(path_to_detection_results, 'r') as f:
            data = json.load(f)

        combined: list[tuple[str, BBox, float]] = []
        for entry in data:
            file_name = entry['file']
            for det in entry.get('detections', []):
                if det['category'] == "1" and det['conf'] >= confidence:
                    combined.append((file_name, det['bbox'], det['conf']))

        # If requested, keep only the best (highest-conf) det per image:
        if only_one_bb_per_image:
            best_per_file: dict[str, tuple[BBox, float]] = {}
            for file_name, bbox, conf in combined:
                prev = best_per_file.get(file_name)
                if prev is None or conf > prev[1]:
                    best_per_file[file_name] = (bbox, conf)
            # rebuild combined from best_per_file
            combined = [(fn, bc[0], bc[1]) for fn, bc in best_per_file.items()]

        # sort all remaining detections by confidence descending
        combined_sorted = sorted(combined, key=lambda x: x[2], reverse=True)

        # unzip (guard against empty)
        if combined_sorted:
            img_list, bbox_list, conf_list = map(list, zip(*combined_sorted))
        else:
            img_list, bbox_list, conf_list = [], [], []

        return {
            'file': img_list,
            'bbox': bbox_list,
            'conf': conf_list
        }


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
    detector_model : str | None
        If a detector model is provided, the detection will be applied to the whole dataset and stored for training.
        The model must be one of the available models in the MegaDetector repository.
        The default is None. A valid detection output must be available at the path_to_detector_output.
    applied_detection_confidence : float
        The detection is done with a confidence of 0.25 by default to provide some flexibility
        with the training. The confidence can be set to a higher value to reduce the number of detections used from
        the output. The default is 0.25.
    random_seed : int
        The seed used for the random number generator. The default is 55.
    extra_test_set : float | None
        If a extra test set needs to be created not included in the folds for additional later experiments.
        The value defines the proportion of the dataset to include in the separate test split. The default is None.
    n_folds : int
        The number of folds to use for cross-validation. The default is 5.
    test_fold : int
        Index of the fold used for testing; val_fold is (test_fold+1)%n_folds.
    available_detection_confidence : float
        If the MD is applied, this is the minimal confidence to storred the output. If MD is not applied, this Value
        must be set to the value used for the detection. The default is 0.25.
    mode : str
        The mode in which the dataset is used. Available: 'train', 'test', 'val', 'pred' 'init', 'eval'
        The default is 'train'.
    image_pipeline : BatchImagePipeline (custom class)
        The image_pipeline to be applied to the images.
    sample_size : int
        The limit of samples to be used from each sequence. In train mode this means, between 1 and sample_size samples
        will be randomly chosen with replacement. In test and val mode the images with the highest detection
        confidence scores will be used and it is limited to sample_size samples. For init and eval mode, all
        images will be used. The default is None meaning in all modes all images will be used.
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
            extra_test_set: float | None = None,
            n_folds: int = 5,
            test_fold: int = 0,
            mode: str = 'train',
            image_pipeline: BatchImagePipeline | None = None,
            sample_size: int | None = 10
            ):
        super().__init__(
            path_labelfiles=path_labelfiles,
            path_to_dataset=path_to_dataset,
            path_to_detector_output=path_to_detector_output,
            detector_model=detector_model,
            categories_to_drop=categories_to_drop,
            applied_detection_confidence=applied_detection_confidence,
            available_detection_confidence=available_detection_confidence,
            random_seed=random_seed,
            extra_test_set=extra_test_set,
            n_folds=n_folds,
            test_fold=test_fold,
            mode=mode,
            )

        self.sample_size = sample_size

        if image_pipeline is None:
            self.image_pipeline = BatchImagePipeline(
                pre_ops=[]
                )
        else:
            self.image_pipeline = image_pipeline

        if self.image_pipeline.path_to_dataset is None:
            self.image_pipeline.path_to_dataset = self.path_to_dataset

    def get_index_by_seq_id(
            self,
            seq_id: int) -> int:

        mask = self.ds['seq_id'] == seq_id
        if not mask.any():
            raise ValueError(f"seq_id {seq_id} not found in dataset.")

        pos = int(np.flatnonzero(mask.to_numpy())[0])
        return pos

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Any:

        row = self.ds.iloc[index]

        class_id = row['class_id']
        class_label = row['class_label']
        seq_id = row['seq_id']
        base_image_path = Path(row['Directory'])

        sequence = self.get_bb_list_for_seq(seq_id, only_one_bb_per_image=True)

        image_path: list[str | PathLike] = [base_image_path / file_name for file_name in sequence['file']]
        bbox = sequence['bbox']
        conf = sequence['conf']

        if self.sample_size is not None and self.mode in ['train', 'val', 'test', 'pred', 'eval']:
            sequence_length = len(sequence['file'])

            if self.mode == 'train':
                actual_sample_size = np.random.randint(1, self.sample_size + 1)
                sample_indices = np.random.choice(
                    sequence_length,
                    size=actual_sample_size,
                    replace=True
                    )
            else:
                sample_indices = np.arange(min(sequence_length, self.sample_size))

            image_path = [image_path[i] for i in sample_indices]
            bbox = [bbox[i] for i in sample_indices]
            conf = [conf[i] for i in sample_indices]

        sample = self.image_pipeline(image_path, bbox)

        item = {
            'sample': sample,
            'class_id': class_id,
            'bbox': bbox,
            'conf': conf,
            'seq_id': seq_id,
            }

        if self.mode in ['eval', 'pred']:
            if row['seq_id'] in self.val_seq_ids:
                item['set'] = 'val'
            elif row['seq_id'] in self.test_seq_ids:
                item['set'] = 'test'
            else:
                item['set'] = 'train'

        if self.mode == 'eval':
            item['file_path'] = image_path
            item['class_label'] = class_label

        return item


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
    detector_model : str | None
        If a detector model is provided, the detection will be applied to the whole dataset and stored for training.
        The model must be one of the available models in the MegaDetector repository.
        The default is None. A valid detection output must be available at the path_to_detector_output.
    applied_detection_confidence : float
        The detection is done with a confidence of 0.25 by default to provide some flexibility
        with the training. The confidence can be set to a higher value to reduce the number of detections used from
        the output. The default is 0.25.
    random_seed : int
        The seed used for the random number generator. The default is 55.
    extra_test_set : float | None
        If a extra test set needs to be created not included in the folds for additional later experiments.
        The value defines the proportion of the dataset to include in the separate test split. The default is None.
    n_folds : int
        The number of folds to use for cross-validation. The default is 5.
    test_fold : int
        Index of the fold used for testing; val_fold is (test_fold+1)%n_folds.
    available_detection_confidence : float
        If the MD is applied, this is the minimal confidence to storred the output. If MD is not applied, this Value
        must be set to the value used for the detection. The default is 0.25.
    mode : str
        The mode in which the dataset is used. Can be either 'train', 'test', 'val' or 'init' defining which data will
        be sampled and adjusting how it is sampled. The default is 'train'.
    image_pipeline : ImagePipeline (custom class)
        The image_pipeline to be applied to the images.

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
            extra_test_set: float | None = None,
            n_folds: int = 5,
            test_fold: int = 0,
            mode: str = 'train',
            image_pipeline: ImagePipeline | None = None
            ):
        super().__init__(
            path_labelfiles=path_labelfiles,
            path_to_dataset=path_to_dataset,
            path_to_detector_output=path_to_detector_output,
            categories_to_drop=categories_to_drop,
            detector_model=detector_model,
            applied_detection_confidence=applied_detection_confidence,
            available_detection_confidence=available_detection_confidence,
            random_seed=random_seed,
            extra_test_set=extra_test_set,
            n_folds=n_folds,
            test_fold=test_fold,
            mode=mode,
            )

        if image_pipeline is None:
            self.image_pipeline = ImagePipeline(
                pre_ops=[]
                )
        else:
            self.image_pipeline = image_pipeline
        if self.image_pipeline.path_to_dataset is None:
            self.image_pipeline.path_to_dataset = self.path_to_dataset

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
        sample = self.image_pipeline(image_path, bbox)

        item = {
            'sample': sample,
            'class_id': row['class_id'],
            'bbox': bbox,
            'conf': row['conf'],
            'seq_id': row['seq_id'],
        }

        if self.mode in ['eval', 'pred']:
            if row['seq_id'] in self.val_seq_ids:
                item['set'] = 'val'
            elif row['seq_id'] in self.test_seq_ids:
                item['set'] = 'test'
            else:
                item['set'] = 'train'

        if self.mode == 'eval':
            item['file_path'] = image_path
            item['class_label'] = row['class_label']

        if self.mode == 'pred':
            item['file'] = image_path.name

        return item
