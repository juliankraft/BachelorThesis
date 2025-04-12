import torch
from pytorch_lightning import LightningDataModule
from ba_dev.dataset import MammaliaData, MammaliaDataImage


class MammliaDataModule(LightningDataModule):
    def __init__(
            self,
            dataset: MammaliaData | MammaliaDataImage,
            n_folds: int = 5,
            val_fold: int = 0,
            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = True,
            ) -> None:
        super().__init__()
        self.dataset = dataset
        self.n_folds = n_folds
        self.val_fold = val_fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str | None = None) -> None:
        pass

    def get_dataset(self, stage: str) -> MammaliaData | MammaliaDataImage:
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return self.test_dataloader()