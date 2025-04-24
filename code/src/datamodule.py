import torch
from pytorch_lightning import LightningDataModule
from ba_dev.dataset import MammaliaData, MammaliaDataSequence, MammaliaDataImage

from typing import Type


class MammaliaDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_cls: Type[MammaliaData],
            dataset_kwargs: dict,
            n_folds: int = 5,
            val_fold: int = 0,
            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = True,
            ) -> None:
        super().__init__()

        self.dataset_cls = dataset_cls
        self.dataset_kwargs = dataset_kwargs.copy()
        self.n_folds = n_folds
        self.val_fold = val_fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset: MammaliaData | None = None
        self.val_dataset: MammaliaData | None = None
        self.test_dataset: MammaliaData | None = None

        self.class_weights: torch.Tensor | None = None

    def setup(self, stage: str | None = None) -> None:

        init_kwargs = {
            **self.dataset_kwargs,
            'n_folds': self.n_folds,
            'val_fold': self.val_fold,
            'mode': 'init'
        }

        master = self.dataset_cls(**init_kwargs)

        self.class_weights = master.get_class_weights()

        base_kwargs = {
            **self.dataset_kwargs,
            'n_folds': self.n_folds,
            'val_fold': self.val_fold,
        }

        self.train_dataset = self.dataset_cls(**{**base_kwargs, 'mode': 'train'})
        self.val_dataset   = self.dataset_cls(**{**base_kwargs, 'mode': 'val'})
        self.test_dataset  = self.dataset_cls(**{**base_kwargs, 'mode': 'test'})

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return self.test_dataloader()