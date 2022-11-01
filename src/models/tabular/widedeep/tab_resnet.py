from src.models.tabular.widedeep.base import WDBaseModel
from pytorch_widedeep.models import TabResnet


class WDTabResnetModel(WDBaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_network(self):
        self.model = TabResnet(
            column_idx=self.hparams.column_idx,
            cat_embed_input=self.hparams.cat_embed_input,
            cat_embed_dropout=self.hparams.cat_embed_dropout,
            use_cat_bias=self.hparams.use_cat_bias,
            cat_embed_activation=self.hparams.cat_embed_activation,
            continuous_cols=self.hparams.continuous_cols,
            cont_norm_layer=self.hparams.cont_norm_layer,
            embed_continuous=self.hparams.embed_continuous,
            cont_embed_dim=self.hparams.cont_embed_dim,
            cont_embed_dropout=self.hparams.cont_embed_dropout,
            use_cont_bias=self.hparams.use_cont_bias,
            cont_embed_activation=self.hparams.cont_embed_activation,
            blocks_dims=self.hparams.blocks_dims,
            blocks_dropout=self.hparams.blocks_dropout,
            simplify_blocks=self.hparams.simplify_blocks,
            mlp_hidden_dims=self.hparams.mlp_hidden_dims,
            mlp_activation=self.hparams.mlp_activation,
            mlp_dropout=self.hparams.mlp_dropout,
            mlp_batchnorm=self.hparams.mlp_batchnorm,
            mlp_batchnorm_last=self.hparams.mlp_batchnorm_last,
            mlp_linear_first=self.hparams.mlp_linear_first
        )
