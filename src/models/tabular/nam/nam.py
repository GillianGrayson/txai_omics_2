import torch
from src.models.tabular.base import BaseModel
from .repository.models.nam import NAM
from .repository.config.base import Config


class NeuralAdditiveModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_network()

    def build_network(self):
        cfg = Config(
            hidden_sizes=list(self.hparams.hidden_sizes),
            activation=self.hparams.activation,
            dropout=self.hparams.dropout,
            feature_dropout=self.hparams.feature_dropout,
            decay_rate=self.hparams.decay_rate,
            l2_regularization=self.hparams.l2_regularization,
            output_regularization=self.hparams.output_regularization,
            num_basis_functions=self.hparams.num_basis_functions,
            units_multiplier=self.hparams.units_multiplier,
        )

        self.model = NAM(
            config=cfg,
            name="NAM",
            num_inputs=self.hparams.input_dim,
            num_units=list(self.hparams.num_units),
            num_outputs=self.hparams.output_dim
        )

    def forward(self, batch):
        if isinstance(batch, dict):
            x = batch["all"]
        else:
            x = batch
        predictions, fnn_out = self.model(x)
        if self.produce_probabilities:
            return torch.softmax(predictions, dim=1)
        else:
            return predictions

    def forward_train(self, batch):
        if isinstance(batch, dict):
            x = batch["all"]
        else:
            x = batch
        predictions, fnn_out = self.model(x)
        if self.produce_probabilities:
            return torch.softmax(predictions, dim=1), fnn_out
        else:
            return predictions, fnn_out

    def forward_eval(self, batch):
        return self.forward_train(batch)

    def calc_out_and_loss(self, out, y, stage):
        predictions, fnn_out = out

        if stage in ["trn", "val", "tst"]:

            def features_loss(per_feature_outputs: torch.Tensor) -> torch.Tensor:
                """Penalizes the L2 norm of the prediction of each feature net."""
                per_feature_norm = [  # L2 Regularization
                    torch.mean(torch.square(outputs)) for outputs in per_feature_outputs
                ]
                return sum(per_feature_norm) / len(per_feature_norm)

            def weight_decay(model: torch.nn.Module) -> torch.Tensor:
                """Penalizes the L2 norm of weights in each feature net."""
                num_networks = 1 if self.hparams.use_dnn else len(model.feature_nns)
                l2_losses = [(x ** 2).sum() for x in model.parameters()]
                return sum(l2_losses) / num_networks

            loss = self.loss_fn(predictions, y)

            reg_loss = 0.0
            if self.hparams.output_regularization > 0:
                reg_loss += self.hparams.output_regularization * features_loss(fnn_out)

            if self.hparams.l2_regularization > 0:
                reg_loss += self.hparams.l2_regularization * weight_decay(self.model)

            loss = loss + reg_loss
            return predictions, loss
        else:
            raise ValueError(f"Unsupported stage: {stage}")

