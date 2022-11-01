from src.models.tabular.widedeep.base import WDBaseModel
from pytorch_widedeep.models import TabNet


class WDTabNetModel(WDBaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_network(self):
        self.model = TabNet(
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
            n_steps=self.hparams.n_steps,
            step_dim=self.hparams.output_dim,
            attn_dim=self.hparams.attn_dim,
            dropout=self.hparams.dropout,
            n_glu_step_dependent=self.hparams.n_glu_step_dependent,
            n_glu_shared=self.hparams.n_glu_shared,
            ghost_bn=self.hparams.ghost_bn,
            virtual_batch_size=self.hparams.virtual_batch_size,
            momentum=self.hparams.momentum,
            gamma=self.hparams.gamma,
            epsilon=self.hparams.epsilon,
            mask_type=self.hparams.mask_type,
        )
