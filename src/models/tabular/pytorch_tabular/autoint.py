from src.models.tabular.pytorch_tabular.base import PTBaseModel
from src.models.tabular.pytorch_tabular.repository.models.autoint.autoint import AutoIntModel
from omegaconf import DictConfig


class PTAutoIntModel(PTBaseModel):

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
                'attn_embed_dim': self.hparams.attn_embed_dim,
                'num_heads': self.hparams.num_heads,
                'num_attn_blocks': self.hparams.num_attn_blocks,
                'attn_dropouts': self.hparams.attn_dropouts,
                'has_residuals': self.hparams.has_residuals,
                'embedding_dim': self.hparams.embedding_dim,
                'embedding_dropout': self.hparams.embedding_dropout,
                'deep_layers': self.hparams.deep_layers,
                'layers': self.hparams.layers,
                'activation': self.hparams.activation,
                'dropout': self.hparams.dropout,
                'use_batch_norm': self.hparams.use_batch_norm,
                'batch_norm_continuous_input': self.hparams.batch_norm_continuous_input,
                'attention_pooling': self.hparams.attention_pooling,
                'initialization': self.hparams.initialization,
            }
        )
        self.model = AutoIntModel(
            config=config
        )
