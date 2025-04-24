import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader
from ba_dev.dataset import MammaliaData
from ba_dev.transform import ImagePipeline

from typing import Type, Any


class MammaliaDataModule(L.LightningDataModule):

    """
    LightningDataModule for Mammalia datasets, handling train/val/test splits and dataloaders.

    This DataModule abstracts away fold-based splitting, data loading, and augmentation pipelines
    for both sequence-level (MammaliaDataSequence) and image-level (MammaliaDataImage) datasets.

    Parameters
    ----------
    dataset_cls : Type[MammaliaData]
        Subclass of MammaliaData to use (e.g., MammaliaDataSequence or MammaliaDataImage).
    dataset_kwargs : dict
        Keyword arguments to pass to the dataset constructor.
        Must include keys:
            'path_labelfiles', 'path_to_dataset', 'path_to_detector_output'
        Possible keys:
            'categories_to_drop', 'detector_model', 'applied_detection_confidence',
            'available_detection_confidence', 'random_seed', 'extra_test_set',
            'image_pipeline', 'sample_size'
        Forbidden keys:
            'n_folds', 'test_fold', 'image_pipeline', 'mode'
    n_folds : int, default=5
        Number of stratified folds for cross-validation.
    test_fold : int, default=0
        Index of the fold used for testing; val_fold is (test_fold+1)%n_folds.
    image_pipeline : ImagePipeline | None, default=None
        Pipeline for loading and preprocessing images.
    augmented_image_pipeline : ImagePipeline | None, default=None
        Optional image pipeline only for training if augmentation is needed.
        if not provided image_pipeline is used.
    batch_size : int, default=32
        Batch size for dataloader (doubled if mode is not 'train').
    num_workers : int, default=4
        Number of subprocesses to use for data loading.
    pin_memory : bool, default=True
        Whether to pin memory in DataLoader for faster GPU transfer.

    Attributes
    ----------
    class_weights : torch.Tensor
        Tensor of balanced class weights computed on the combined train+val set.
    """

    def __init__(
            self,
            dataset_cls: Type[MammaliaData],
            dataset_kwargs: dict,
            n_folds: int = 5,
            test_fold: int = 0,
            image_pipeline: ImagePipeline | None = None,
            augmented_image_pipeline: ImagePipeline | None = None,

            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = True,
            ) -> None:

        super().__init__()

        self.dataset_cls = dataset_cls
        required_keys = ['path_labelfiles', 'path_to_dataset', 'path_to_detector_output']
        for key in required_keys:
            if key not in dataset_kwargs:
                raise ValueError(f"Missing required key '{key}' in dataset_kwargs.")

        module_keys = ['n_folds', 'test_fold', 'image_pipeline', 'mode']
        for key in module_keys:
            if key in dataset_kwargs:
                raise ValueError(f"Key '{key}' should not be provided in dataset_kwargs. It is defined internally.")

        self.n_folds = n_folds
        dataset_kwargs['n_folds'] = n_folds
        self.test_fold = test_fold
        dataset_kwargs['test_fold'] = test_fold

        self.dataset_kwargs = dataset_kwargs.copy()

        self.image_pipeline = image_pipeline

        if augmented_image_pipeline is None:
            self.augmented_image_pipeline = image_pipeline
        else:
            self.augmented_image_pipeline = augmented_image_pipeline

        init_dataset = self.get_dataset(mode='init')

        self.class_weights = init_dataset.get_class_weights()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.dataloader_kwargs: dict[str, Any] = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory
            }

    def get_class_weights(self) -> torch.Tensor:
        return self.class_weights

    def get_dataset(
            self,
            mode: str,
            ) -> MammaliaData:

        if mode == 'train':
            image_pipeline = self.augmented_image_pipeline
        else:
            image_pipeline = self.image_pipeline

        dataset = self.dataset_cls(
            image_pipeline=image_pipeline,
            mode=mode,
            **self.dataset_kwargs
            )

        return dataset

    def train_dataloader(self) -> DataLoader:
        dataset = self.get_dataset(mode='train')
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **self.dataloader_kwargs
            )

    def val_dataloader(self) -> DataLoader:
        dataset = self.get_dataset(mode='valid')
        return DataLoader(
            dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            **self.dataloader_kwargs
        )

    def test_dataloader(self) -> DataLoader:
        dataset = self.get_dataset(mode='test')
        return DataLoader(
            dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            **self.dataloader_kwargs
        )

    def predict_dataloader(self) -> DataLoader:
        dataset = self.get_dataset(mode='eval')
        return DataLoader(
            dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            **self.dataloader_kwargs
        )
