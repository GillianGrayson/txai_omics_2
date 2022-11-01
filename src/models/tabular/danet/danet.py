import torch
from src.models.tabular.base import BaseModel
from .repository.DANet import DANet


class DANetModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_network()

    def build_network(self):
        self.model = DANet(
            input_dim=self.hparams.input_dim,
            num_classes=self.hparams.output_dim,
            layer_num=self.hparams.layer_num,
            base_outdim=self.hparams.base_outdim,
            k=self.hparams.k,
            virtual_batch_size=self.hparams.virtual_batch_size,
            drop_rate=self.hparams.drop_rate
        )

    def forward(self, batch):
        if isinstance(batch, dict):
            x = batch["all"]
        else:
            x = batch
        x = self.model(x)
        if self.produce_probabilities:
            return torch.softmax(x, dim=1)
        else:
            return x
