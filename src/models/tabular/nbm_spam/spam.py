import torch
from src.models.tabular.base import BaseModel
from src.models.tabular.nbm_spam.repository.concept_spam import ConceptSPAM


class SPAMModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_network()

    def build_network(self):
        self.model = ConceptSPAM(
            num_concepts=self.hparams.input_dim,
            num_classes=self.hparams.output_dim,
            ranks=self.hparams.ranks,
            dropout=self.hparams.dropout,
            ignore_unary=self.hparams.ignore_unary,
            reg_order=self.hparams.reg_order,
            lower_order_correction=self.hparams.lower_order_correction,
            use_geometric_mean=self.hparams.use_geometric_mean,
            orthogonal=self.hparams.orthogonal,
            proximal=self.hparams.proximal,
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
            loss = self.loss_fn(out, y)
            reg_loss = self.model.tensor_regularization()
            basis_l1_loss = self.model.basis_l1_regularization()
            loss += reg_loss * self.hparams.regularization_scale + basis_l1_loss * self.hparams.basis_l1_regularization
            return out, loss
        elif stage == "val":
            loss = self.loss_fn(out, y)
            return out, loss
        elif stage == "tst":
            loss = self.loss_fn(out, y)
            return out, loss
        else:
            raise ValueError(f"Unsupported stage: {stage}")
