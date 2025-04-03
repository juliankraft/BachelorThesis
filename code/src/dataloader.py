import os
import warnings
import csv
import json
import torch
import pandas as pd
from pathlib import Path
from typing import Any

from megadetector.detection.run_detector import model_string_to_model_version

from os import PathLike
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src.runner import MegaDetectorRunner


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
        By default all non-empty labels are used. To drop certain labels from the dataset, provide a list of labels
        to drop. In detect mode, this parameter is ignored.
    detector_model : str
        If a detector model is provided, the detection will be applied to the whole dataset and stored for training.
        The model must be one of the available models in the MegaDetector repository.
        The default is None. A valid detection output must be available at the path_to_detector_output.
    detection_confidence : float
        The detection is done with a confidence of 0.25 by default to provide some flexibility
        with the training. The confidence can be set to a higher value to reduce the number of detections used from
        the output. The default is 0.25.
    sample_length : int
        For training this parameter specifies the range (1 - sample_length) of randomly selected samples per sequence.
        For testing this parameter specifies the maximum number of samples per sequence.
        The default is 10.
    sample_img_size : [int, int]
        The size to which the detected areas are resized. The default is [224, 224].
    mode : str
        The mode in which the dataset is used. Can be either 'train', 'test' or 'init' defining which data will be
        sampled and adjusting how it is sampled. The default is 'train'.
    """

    def __init__(
            self,
            path_labelfiles: str | PathLike,
            path_to_dataset: str | PathLike,
            path_to_detector_output: str | PathLike,
            categories_to_drop: list[str] | None = None,
            detector_model: str | None = None,
            detection_confidence: float = 0.25,
            sample_length: int = 10,
            sample_img_size: list[int] = [224, 224],
            mode: str = 'train',
            ):
        super().__init__()

        if mode in ['train', 'test', 'init']:
            self.mode = mode
        else:
            raise ValueError("Please choose a mode from ['train', 'test'].")

        if detection_confidence < 0.25:
            raise ValueError("Detection confidence must be at least 0.25.")

        self.categories_to_drop = categories_to_drop if categories_to_drop is not None else []
        self.detection_confidence = detection_confidence
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

        self.ds_full = self.reading_all_metadata(
                    list_of_files=self.getting_all_files_of_type(self.path_labelfiles, file_type='.csv'),
                    categories_to_drop=self.categories_to_drop
                    )

        if self.mode == 'init':
            if self.detector_model is not None:
                self.run_detector()
            else:
                if not any(self.path_to_detector_output.glob("*.json")):
                    raise ValueError('A valid detection output must be available at the path_to_detector_output.')

        if self.ds_full['seq_id'].duplicated().any():
            duplicates = self.ds_full['seq_id'][self.ds_full['seq_id'].duplicated()].unique()
            raise ValueError(f"Duplicate seq_id(s) found in metadata: {duplicates[:5]} ...")

        train_seq_ids, test_seq_ids = train_test_split(
                                            self.ds_full['seq_id'],
                                            test_size=0.2,
                                            random_state=55,
                                            stratify=self.ds_full['label2']
                                            )

        filtered_train_seq_ids = self.exclude_ids_with_no_detections(
            set_type='train',
            sequences_to_filter=train_seq_ids
        )

        filtered_test_seq_ids = self.exclude_ids_with_no_detections(
            set_type='test',
            sequences_to_filter=test_seq_ids
        )

        if self.mode in ['train', 'init']:
            active_seq_ids = filtered_train_seq_ids
        elif self.mode == 'test':
            active_seq_ids = filtered_test_seq_ids

        self.ds = self.ds_full[self.ds_full['seq_id'].isin(active_seq_ids)]
        self.seq_ids = self.ds['seq_id'].tolist()

    def getting_all_files_of_type(
            self,
            path: str | PathLike,
            file_type: str | None = None,
            get_full_path: bool = True
            ) -> list[str | PathLike]:

        path = Path(path)
        files = []
        for file in os.listdir(path):
            if file_type is None or file.endswith(file_type):
                if get_full_path:
                    files.append(path / file)
                else:
                    files.append(file)
        return files

    def reading_all_metadata(
            self,
            list_of_files: list[PathLike | str],
            categories_to_drop: list[str] = []
            ) -> pd.DataFrame:

        metadata = pd.DataFrame()
        for file in list_of_files:
            metadata = pd.concat([metadata, pd.read_csv(file)], ignore_index=True)
            metadata = metadata.dropna(subset=['label2'])
            metadata = metadata[~metadata['label2'].isin(categories_to_drop)]
        return metadata

    def exclude_ids_with_no_detections(
            self,
            set_type: str,
            sequences_to_filter: list[int],
            ) -> list[int]:

        detection_summary = self.get_detection_summary(
            usecols=["seq_id", "max_conf"]
            )

        seq_ids_to_exclude_set = set(
            detection_summary[detection_summary["max_conf"] < self.detection_confidence]["seq_id"].tolist()
            )
        seq_ids_to_filter_set = set(sequences_to_filter)

        excluded_seq_ids = list(seq_ids_to_filter_set & seq_ids_to_exclude_set)

        if excluded_seq_ids:
            suffix = "" if len(excluded_seq_ids) <= 10 else " ..."
            warnings.warn(
                f"With the current detection confidence of {self.detection_confidence},\n"
                f"{len(excluded_seq_ids)} sequences of the {set_type} set had no detections and will be excluded.\n"
                f"Excluded sequences: {excluded_seq_ids[:10]}{suffix}",
                UserWarning
            )

        return list(seq_ids_to_filter_set - seq_ids_to_exclude_set)

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

        if self.mode == 'test':
            raise ValueError("Class weights are not available in test mode.")

        class_weights = torch.Tensor([1.0] * len(self.ds_full['label2'].unique()))

        return class_weights

    def get_all_images_of_sequence(
            self,
            seq_id: int,
            dataframe: pd.DataFrame | None = None,
            ) -> dict[str, PathLike]:

        if dataframe is None:
            dataframe = self.ds_full

        image_dict = {}
        row = dataframe.loc[dataframe['seq_id'] == seq_id].squeeze()
        seq_path = Path(row['Directory'])
        all_files = row['all_files'].split(',')
        for file in all_files:
            image_dict[file] = self.path_to_dataset / seq_path / file
        return image_dict

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

        metadata = self.reading_all_metadata(
                    list_of_files=self.getting_all_files_of_type(self.path_labelfiles, file_type='.csv'),
                    )

        sequences = metadata['seq_id'].unique().tolist()

        detection_rows = []

        for seq_id in sequences:
            seq_images = list(self.get_all_images_of_sequence(seq_id).values())
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

        if self.mode != 'detect':
            raise ValueError("Only available if dataset is in detect mode.")

        if confidence is None:
            confidence = self.detection_confidence

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

    def __getitem__(self, index: int) -> Any:               # still to be implemented
        # seq_id = self.seq_ids[index]

        # images = self.get_all_images_of_sequence(seq_id)
        # bounding_boxes = self.getting_bb_list_for_seq(seq_id)

        print("Methode changed")
        return None
