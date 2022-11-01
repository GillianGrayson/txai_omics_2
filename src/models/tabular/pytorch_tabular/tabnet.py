from src.models.tabular.pytorch_tabular.base import PTBaseModel
from omegaconf import DictConfig
from src.models.tabular.pytorch_tabular.repository.models.tabnet.tabnet_model import TabNetModel


class PTTabNetModel(PTBaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_network(self):
        config = DictConfig(
            {
                'task': self.hparams.task,
                'loss': self.hparams.loss_type,
                'metrics': [],
                'metrics_params': [],
                'target_range': None,
                'output_dim': self.hparams.output_dim,
                'embedding_dims': self.hparams.embedding_dims,
                'continuous_cols': self.hparams.continuous_cols,
                'categorical_cols': self.hparams.categorical_cols,
                'continuous_dim': len(self.hparams.continuous_cols),
                'categorical_dim': len(self.hparams.categorical_cols),
                'n_d': self.hparams.n_d,
                'n_a': self.hparams.n_a,
                'n_steps': self.hparams.n_steps,
                'gamma': self.hparams.gamma,
                'n_independent': self.hparams.n_independent,
                'n_shared': self.hparams.n_shared,
                'virtual_batch_size': self.hparams.virtual_batch_size,
                'mask_type': self.hparams.mask_type,
            }
        )
        self.model = TabNetModel(
            config=config
        )
