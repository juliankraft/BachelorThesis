import csv
import yaml
import torch
import numpy as np
import pytorch_lightning as L
from pathlib import Path
from typing import Sequence, Any
from os import PathLike

BBox = Sequence[float]


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    trainable = 0
    non_trainable = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            non_trainable += p.numel()
    return {
        'trainable': trainable,
        'non_trainable': non_trainable,
        'total': trainable + non_trainable
    }


def load_path_yaml(path_to_config):
    with open(path_to_config, 'r') as f:
        path_config = yaml.safe_load(f)
    return {k: Path(v) for k, v in path_config.items()}


def best_weighted_split(
        weights: np.ndarray,
        target_left_sum: float
        ) -> int:

    total = weights.sum()
    target_right = total - target_left_sum

    best_cut = 0
    best_error = abs(0 - target_left_sum) + abs(total - target_right)

    for idx in range(1, len(weights) + 1):
        left = weights[:idx].sum()
        right = total - left
        err = abs(left - target_left_sum) + abs(right - target_right)
        if err < best_error:
            best_error = err
            best_cut = idx

    return best_cut


class PredictionWriter(L.Callback):
    def __init__(
            self,
            output_path: PathLike,
            log_keys: Sequence[str] | None = None,
            prob_precision: int | None = None
            ):
        super().__init__()

        self.output_file = Path(output_path) / "predictions.csv"
        self._csv = None
        self._writer = None
        self.prob_precision = prob_precision

        _available_keys = ['class_id', 'bbox', 'conf', 'seq_id', 'set', 'file', 'pred_id', 'probs']
        if log_keys is not None:
            for key in log_keys:
                if key not in _available_keys:
                    raise ValueError(
                        f"Invalid key '{key}' in log_keys. "
                        f"Available keys are: {_available_keys}"
                    )
            self.log_keys = log_keys
        else:
            self.log_keys = _available_keys

    def on_predict_start(
            self, trainer: L.Trainer,
            pl_module: L.LightningModule
            ):
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._csv = open(self.output_file, "w", newline="", encoding="utf-8")

        self._writer = csv.DictWriter(
                                self._csv,
                                fieldnames=self.log_keys
                                )
        self._writer.writeheader()

    def on_predict_batch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            outputs: list[dict[str, Any]],
            batch: dict[str, Any],
            batch_idx: int,
            dataloader_idx: int | None = None
            ):

        assert self._writer is not None

        if isinstance(outputs, dict):
            outputs_list = [outputs]
        else:
            outputs_list = outputs

        for out_dict in outputs_list:

            output = self.reconstruct_batch(out_dict)

            for row in output:

                self._writer.writerow(row)

    def on_predict_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule
            ):
        if self._csv:
            self._csv.close()

    def reconstruct_batch(
            self,
            batch
            ) -> list[dict[str, Any]]:

        batch_size = len(batch['class_id'])
        reconstructed = []

        for i in range(batch_size):
            bbox = [tensor[i].item() for tensor in batch['bbox']]
            probs = batch['probs'][i].tolist()
            if self.prob_precision is not None:
                probs = [round(p, self.prob_precision) for p in probs]
            item = {
                'class_id': batch['class_id'][i].item(),
                'bbox': bbox,
                'conf': batch['conf'][i].item(),
                'seq_id': batch['seq_id'][i].item(),
                'set': batch['set'][i],
                'file': batch['file'][i],
                'pred_id': batch['preds'][i].item(),
                'probs': probs
                }
            filtered = {k: item[k] for k in self.log_keys}
            reconstructed.append(filtered)

        return reconstructed
