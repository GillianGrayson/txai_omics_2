from src.models.tabular.pytorch_tabular.base import PTBaseModel
from src.models.tabular.pytorch_tabular.repository.models.category_embedding.category_embedding_model import CategoryEmbeddingModel
from omegaconf import DictConfig


class PTCategoryEmbeddingModel(PTBaseModel):

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
                'layers': self.hparams.layers,
                'batch_norm_continuous_input': self.hparams.batch_norm_continuous_input,
                'activation': self.hparams.activation,
                'embedding_dropout': self.hparams.embedding_dropout,
                'dropout': self.hparams.dropout,
                'use_batch_norm': self.hparams.use_batch_norm,
                'initialization': self.hparams.initialization
            }
        )
        self.model = CategoryEmbeddingModel(
            config=config
        )
