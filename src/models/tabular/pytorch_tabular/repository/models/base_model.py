# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Base Model"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

try:
    import plotly.graph_objects as go

    import wandb

    WANDB_INSTALLED = True
except ImportError:
    WANDB_INSTALLED = False


logger = logging.getLogger(__name__)


class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(
        self,
        config: DictConfig,
        custom_loss: Optional[torch.nn.Module] = None,
        custom_metrics: Optional[List[Callable]] = None,
        custom_optimizer: Optional[torch.optim.Optimizer] = None,
        custom_optimizer_params: Dict = {},
        **kwargs,
    ):
        super().__init__()
        self.custom_loss = custom_loss
        self.custom_metrics = custom_metrics
        self.custom_optimizer = custom_optimizer
        self.custom_optimizer_params = custom_optimizer_params
        # Updating config with custom parameters for experiment tracking
        if self.custom_loss is not None:
            config.loss = str(self.custom_loss)
        if self.custom_metrics is not None:
            config.metrics = [str(m) for m in self.custom_metrics]
            config.metrics_params = [vars(m) for m in self.custom_metrics]
        if self.custom_optimizer is not None:
            config.optimizer = str(self.custom_optimizer.__class__.__name__)
        if len(self.custom_optimizer_params) > 0:
            config.optimizer_params = self.custom_optimizer_params
        self.save_hyperparameters(config)
        # The concatenated output dim of the embedding layer
        self._build_network()

    @abstractmethod
    def _build_network(self):
        pass

    def data_aware_initialization(self, datamodule):
        pass

    def compute_backbone(self, x: Dict):
        # Returns output
        x = self.backbone(x)
        return x

    def apply_output_sigmoid_scaling(self, y_hat: torch.Tensor):
        if (self.hparams.task == "regression") and (
            self.hparams.target_range is not None
        ):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                y_hat[:, i] = y_min + nn.Sigmoid()(y_hat[:, i]) * (y_max - y_min)
        return y_hat

    def pack_output(
        self, y_hat: torch.Tensor, backbone_features: torch.tensor
    ) -> Dict[str, Any]:
        # if self.head is the Identity function it means that we cannot extract backbone features,
        # because the model cannot be divide in backbone and head (i.e. TabNet)
        if type(self.head) == nn.Identity:
            return {"logits": y_hat}
        else:
            return {"logits": y_hat, "backbone_features": backbone_features}

    def compute_head(self, backbone_features: Tensor):
        y_hat = self.head(backbone_features)
        y_hat = self.apply_output_sigmoid_scaling(y_hat)
        return self.pack_output(y_hat, backbone_features)

    def forward(self, x: Dict):
        x = self.compute_backbone(x)
        return self.compute_head(x)
