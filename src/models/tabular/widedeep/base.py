import torch
from src.models.tabular.base import BaseModel


class WDBaseModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_network()
        self.feats_all_ids = list(self.hparams.column_idx.values())
        self.feats_cat_ids = []
        if self.hparams.cat_embed_input:
            for x in self.hparams.cat_embed_input:
                self.feats_cat_ids.append(self.hparams.column_idx[x[0]])
        self.feats_con_ids = []
        if self.hparams.continuous_cols:
            for x in self.hparams.continuous_cols:
                self.feats_con_ids.append(self.hparams.column_idx[x])

    def build_network(self):
        pass

    def forward(self, batch):
        if isinstance(batch, dict):
            x = batch["all"]
        else:
            x = batch[:, self.feats_all_ids]
        x = self.model(x)
        if isinstance(x, tuple):
            x = x[0]
        if self.produce_probabilities:
            return torch.softmax(x, dim=1)
        else:
            return x
