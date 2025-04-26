import torch
import torch.nn as nn
import pytorch_lightning as L
from torchvision.models import get_model
from torchmetrics import Accuracy, MetricCollection
from torch import Tensor
from typing import Dict, Literal, Any


class LightningModelImage(L.LightningModule):
    def __init__(
            self,
            num_classes: int,
            class_weights: Tensor | None = None,
            backbone_name: str = 'efficientnet_b0',
            backbone_pretrained: bool = True,
            backbone_weights: str = 'DEFAULT',
            optimizer_name: str = 'AdamW',
            optimizer_kwargs: Dict[str, Any] | None = None,
            scheduler_name: str | None = None,
            scheduler_kwargs: Dict[str, Any] | None = None,
            ):
        super().__init__()

        self.save_hyperparameters()

        self.num_classes = num_classes
        self.class_weights = class_weights

        self.backbone = get_model(
            backbone_name,
            weights=(backbone_weights if backbone_pretrained else None)
            )

        self._swap_head()

        if self.class_weights is not None:
            if not isinstance(self.class_weights, Tensor):
                raise TypeError(
                    "class_weights must be a Tensor or None."
                )
            if len(self.class_weights) != self.num_classes:
                raise ValueError(
                    f'Length of class_weights ({len(self.class_weights)})'
                    f'does not match num_classes ({self.num_classes}).'
                    )
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer_cls = getattr(torch.optim, optimizer_name)
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.train_metrics = self.make_metrics('train')
        self.val_metrics = self.make_metrics('val')
        self.test_metrics = self.make_metrics('test')

    def _swap_head(self):
        if hasattr(self.backbone, 'fc'):
            if isinstance(self.backbone.fc, torch.nn.Linear):
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Linear(in_features, self.num_classes)
            else:
                raise AttributeError("The backbone's 'fc' layer is not a torch.nn.Linear module.")

        elif hasattr(self.backbone, 'classifier'):
            cls = self.backbone.classifier

            if isinstance(cls, torch.nn.Sequential):
                *body, last = list(cls.children())
                in_features = last.in_features
                if isinstance(in_features, int):
                    body.append(nn.Linear(in_features, self.num_classes))
                else:
                    raise AttributeError(
                        "The backbone's 'classifier' layer does not have an integer input feature size."
                        )
                self.backbone.classifier = nn.Sequential(*body)

            elif isinstance(cls, nn.Linear):
                in_features = cls.in_features
                self.backbone.classifier = nn.Linear(in_features, self.num_classes)

            else:
                raise ValueError(
                    f"Unexpected classifier structure: {type(cls)}"
                    )
        else:
            raise ValueError(
                f"Cannot find a head to replace on model {type(self.backbone)}"
                )

    def make_metrics(
            self,
            split: str
            ) -> MetricCollection:

        mc = MetricCollection({
            'acc':     Accuracy(task="multiclass",
                                num_classes=self.num_classes,
                                average="micro"),
            'bal_acc': Accuracy(task="multiclass",
                                num_classes=self.num_classes,
                                average="macro"),
        }, prefix=f"{split}_")
        return mc

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def common_step(
            self,
            batch: Dict[str, Any],
            mode: Literal['train', 'val', 'test', 'pred']
            ) -> tuple[Tensor, Tensor]:
        x = batch['sample']
        y = batch['class_id']
        logits = self(x)
        loss = self.criterion(logits, y)

        if mode != 'pred':
            mc = {
                'train': self.train_metrics,
                'val':   self.val_metrics,
                'test':  self.test_metrics
                }[mode]

            metric_dict = mc(logits, y)

            self.log_dict({
                    f"{mode}_loss": loss,
                    **metric_dict
                    },
                on_step=(mode == 'train'),
                on_epoch=True,
                logger=True
            )
        return logits, loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        _, loss = self.common_step(batch, 'train')
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        _, loss = self.common_step(batch, 'val')
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        _, loss = self.common_step(batch, 'test')
        return loss

    def predict_step(self, batch: Dict[str, Any], batch_idx: int):
        logits, _ = self.common_step(batch, 'pred')
        probs = nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        batch['preds'] = preds
        batch['probs'] = probs
        return batch

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer_cls(
            self.parameters(),
            **self.optimizer_kwargs
            )
        if self.scheduler_name is not None:
            sched_cls = getattr(torch.optim.lr_scheduler, self.scheduler_name, None)
            if sched_cls is None:
                raise ValueError(f"Scheduler '{self.scheduler_name}' not found in torch.optim.lr_scheduler")
            scheduler = sched_cls(optimizer, **self.scheduler_kwargs)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch'
                }
            }
        return optimizer

    def on_train_epoch_end(self) -> None:
        if not self.scheduler_name:
            return

        optim = self.optimizers()
        if isinstance(optim, list):
            optim0 = optim[0]
        else:
            optim0 = optim

        lr = optim0.param_groups[0]['lr']
        self.log('lr', lr, on_epoch=True)
