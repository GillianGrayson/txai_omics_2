import torch
from src.models.tabular.base import BaseModel
from src.models.tabular.nbm_spam.repository.concept_nam import ConceptNAMNary


class NAMModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_network()

    def build_network(self):
        self.model = ConceptNAMNary(
            num_concepts=self.hparams.input_dim,
            num_classes=self.hparams.output_dim,
            nary=self.hparams.nary,
            first_layer=self.hparams.first_layer,
            first_hidden_dim=self.hparams.first_hidden_dim,
            hidden_dims=self.hparams.hidden_dims,
            num_subnets=self.hparams.num_subnets,
            dropout=self.hparams.dropout,
            concept_dropout=self.hparams.concept_dropout,
            batchnorm=self.hparams.batchnorm,
            output_penalty=self.hparams.output_penalty,
            polynomial=self.hparams.polynomial
        )

    def forward(self, batch):
        if isinstance(batch, dict):
            x = batch["all"]
        else:
            x = batch
        x = self.model(x)
        if isinstance(x, tuple):
            return x
        else:
            if self.produce_probabilities:
                return torch.softmax(x, dim=1)
            else:
                return x

    def calc_out_and_loss(self, out, y, stage):
        if stage == "trn":
            out_base, out_nn = out
            loss = self.loss_fn(out_base, y)
            loss_output_penalty = (torch.pow(out_nn, 2).mean(dim=-1)).mean() * self.hparams.output_penalty
            loss += loss_output_penalty
            return out_base, loss
        elif stage == "val":
            loss = self.loss_fn(out, y)
            return out, loss
        elif stage == "tst":
            loss = self.loss_fn(out, y)
            return out, loss
        else:
            raise ValueError(f"Unsupported stage: {stage}")
