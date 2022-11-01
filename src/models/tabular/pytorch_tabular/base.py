import torch
import numpy as np
from src.models.tabular.base import BaseModel


class PTBaseModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_network()
        self.feats_con_ids = []
        if self.hparams.continuous_cols:
            self.feats_con_ids = np.arange(len(self.hparams.continuous_cols))
        self.feats_cat_ids = []
        if self.hparams.categorical_cols:
            self.feats_cat_ids = np.arange(len(self.hparams.categorical_cols)) + len(self.feats_con_ids)

    def build_network(self):
        pass

    def forward(self, batch):
        if not isinstance(batch, dict):
            x = {
                "continuous": batch[:, self.feats_con_ids],
                "categorical": batch[:, self.feats_cat_ids],
            }
        else:
            x = batch
        x = self.model(x)['logits']
        if self.produce_probabilities:
            return torch.softmax(x, dim=1)
        else:
            return x
