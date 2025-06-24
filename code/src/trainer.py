import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from typing import Literal, Sequence, Any
from pathlib import Path

from src.utils import PredictionWriter


class MammaliaTrainer(L.Trainer):
    def __init__(
            self,
            log_dir: Path,
            pred_writer_log_keys: Sequence[str] | None = None,
            pred_writer_prob_precision: int | None = None,
            accelerator: Literal['cpu', 'gpu'] = 'cpu',
            patience: int = 5,
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
            RichProgressBar(
                refresh_rate=1,
                leave=True,
                ),
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
                output_path=log_dir,
                log_keys=pred_writer_log_keys,
                prob_precision=pred_writer_prob_precision
                )
            ]

        super().__init__(
            default_root_dir=log_dir,
            logger=loggers,
            callbacks=callbacks,
            accelerator=accelerator,
            **trainer_kwargs
            )
