from src.models.tabular.pytorch_tabular.base import PTBaseModel
from src.models.tabular.pytorch_tabular.repository.models.ft_transformer.ft_transformer import FTTransformerModel
from omegaconf import DictConfig


class PTFTTransformerModel(PTBaseModel):

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
                'input_embed_dim': self.hparams.input_embed_dim,
                'embedding_initialization': self.hparams.embedding_initialization,
                'embedding_bias': self.hparams.embedding_bias,
                'embedding_dropout': self.hparams.embedding_dropout,
                'share_embedding': self.hparams.share_embedding,
                'share_embedding_strategy': self.hparams.share_embedding_strategy,
                'shared_embedding_fraction': self.hparams.shared_embedding_fraction,
                'attn_feature_importance': self.hparams.attn_feature_importance,
                'num_heads': self.hparams.num_heads,
                'num_attn_blocks': self.hparams.num_attn_blocks,
                'transformer_head_dim': self.hparams.transformer_head_dim,
                'attn_dropout': self.hparams.attn_dropout,
                'add_norm_dropout': self.hparams.add_norm_dropout,
                'ff_dropout': self.hparams.ff_dropout,
                'ff_hidden_multiplier': self.hparams.ff_hidden_multiplier,
                'transformer_activation': self.hparams.transformer_activation,
                'out_ff_layers': self.hparams.out_ff_layers,
                'out_ff_activation': self.hparams.out_ff_activation,
                'out_ff_dropout': self.hparams.out_ff_dropout,
                'use_batch_norm': self.hparams.use_batch_norm,
                'batch_norm_continuous_input': self.hparams.batch_norm_continuous_input,
                'out_ff_initialization': self.hparams.out_ff_initialization,
            }
        )
        self.model = FTTransformerModel(
            config=config
        )
