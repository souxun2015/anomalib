"""EfficientAd: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.
https://arxiv.org/pdf/2303.14535.pdf
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, optim

from anomalib.models.components import AnomalyModule

from .torch_model import SimplenetModel

logger = logging.getLogger(__name__)


class Simplenet(AnomalyModule):
    """PL Lightning Module for the EfficientAd algorithm.

    Args:
        image_size (tuple): size of input images
        backbone (str): model name of pretrained backbone
        layers (list[str, str]): extract features from layers of backbone
        lr (float): learning rate
        weight_decay (float): optimizer weight decay
        patchsize: the size of patch
        patchstride: the stride while do patch feature extraction
    """

    def __init__(
        self,
        layers: list[str],
        image_size: tuple[int, int],
        backbone: str,
        patchsize: int,
        patchstride: int,
        lr: float = 0.0001,
        weight_decay: float = 0.00001,
    ) -> None:
        super().__init__()

        self.model: SimplenetModel = SimplenetModel(
            input_size=image_size,
            patchsize=patchsize,
            patchstride=patchstride,
            layers=layers,
            backbone=backbone,
        )
        self.image_size = image_size
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> dict[str, Tensor]:
        """Training step for EfficientAd returns the student, autoencoder and combined loss.

        Args:
            batch (batch: dict[str, str | Tensor]): Batch containing image filename, image, label and mask

        Returns:
          Loss.
        """
        del args, kwargs  # These variables are not used.

        loss = self.model(batch=batch["image"])

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Validation Step of EfficientAd returns anomaly maps for the input image batch

        Args:
          batch (dict[str, str | Tensor]): Input batch

        Returns:
          Dictionary containing anomaly maps.
        """
        del args, kwargs  # These variables are not used.
        feature, mask, score = self.model(batch["image"])
        batch["anomaly_maps"] = mask
        batch["pred_scores"] = score
        return batch

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(
            list(self.model.pre_projection.parameters()) + list(self.model.discriminator.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        num_steps = min(
            self.trainer.max_steps // len(self.trainer.datamodule.train_dataloader()), self.trainer.max_epochs
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * num_steps), gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class SimplenetLightning(Simplenet):
    """PL Lightning Module for the EfficientAd Algorithm.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            layers=hparams.model.layers,
            backbone=hparams.model.backbone,
            lr=hparams.model.lr,
            weight_decay=hparams.model.weight_decay,
            patchsize=hparams.model.patchsize,
            patchstride=hparams.model.patchstride,
            image_size=hparams.dataset.image_size,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
