from src.models.tabular.pytorch_tabular.base import PTBaseModel
from src.models.tabular.pytorch_tabular.repository.models.node.node_model import NODEModel
from omegaconf import DictConfig


class PTNODEModel(PTBaseModel):

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
                'num_layers': self.hparams.num_layers,
                'num_trees': self.hparams.num_trees,
                'additional_tree_output_dim': self.hparams.additional_tree_output_dim,
                'depth': self.hparams.depth,
                'choice_function': self.hparams.choice_function,
                'bin_function': self.hparams.bin_function,
                'max_features': self.hparams.max_features,
                'input_dropout': self.hparams.input_dropout,
                'initialize_response': self.hparams.initialize_response,
                'initialize_selection_logits': self.hparams.initialize_selection_logits,
                'threshold_init_beta': self.hparams.threshold_init_beta,
                'threshold_init_cutoff': self.hparams.threshold_init_cutoff,
                'embed_categorical': self.hparams.embed_categorical,
                'embedding_dropout': self.hparams.embedding_dropout,
            }
        )
        self.model = NODEModel(
            config=config
        )
