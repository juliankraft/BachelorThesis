import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from typing import Literal, Any
from pathlib import Path

from ba_dev.utils import PredictionWriter


class MammaliaTrainer(L.Trainer):
    def __init__(
            self,
            log_dir: Path,
            accelerator: Literal['cpu', 'gpu'] = 'cpu',
            devices: int = 1,
            patience: int = 5,
            log_every_n_steps: int = 1,
            max_epochs: int | None = None,
            trainer_kwargs: dict[str, Any] | None = None
            ):

        if trainer_kwargs is None:
            trainer_kwargs = {}

        tb_logger = TensorBoardLogger(
            save_dir=log_dir,
            name='',
            version=''
            )

        csv_logger = CSVLogger(
            save_dir=log_dir,
            name='',
            version=''
            )

        loggers = [tb_logger, csv_logger]

        callbacks = [
            ModelCheckpoint(
                filename='best',
                save_last=True,
                save_top_k=1,
                every_n_epochs=1
                ),
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                ),
            LearningRateMonitor(
                logging_interval='epoch'
                ),
            PredictionWriter(
                output_path=log_dir
                )
            ]

        super().__init__(
            default_root_dir=log_dir,
            logger=loggers,
            callbacks=callbacks,
            accelerator=accelerator,
            devices=devices,
            log_every_n_steps=log_every_n_steps,
            max_epochs=max_epochs,
            **trainer_kwargs
            )
