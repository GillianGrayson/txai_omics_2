from src.models.tabular.widedeep.base import WDBaseModel
from pytorch_widedeep.models import TabPerceiver


class WDTabPerceiverModel(WDBaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_network(self):
        self.model = TabPerceiver(
            column_idx=self.hparams.column_idx,
            cat_embed_input=self.hparams.cat_embed_input,
            cat_embed_dropout=self.hparams.cat_embed_dropout,
            use_cat_bias=self.hparams.use_cat_bias,
            cat_embed_activation=self.hparams.cat_embed_activation,
            full_embed_dropout=self.hparams.full_embed_dropout,
            shared_embed=self.hparams.shared_embed,
            add_shared_embed=self.hparams.add_shared_embed,
            frac_shared_embed=self.hparams.frac_shared_embed,
            continuous_cols=self.hparams.continuous_cols,
            cont_norm_layer=self.hparams.cont_norm_layer,
            cont_embed_dropout=self.hparams.cont_embed_dropout,
            use_cont_bias=self.hparams.use_cont_bias,
            cont_embed_activation=self.hparams.cont_embed_activation,
            input_dim=self.hparams.embed_dim,
            n_cross_attns = self.hparams.n_cross_attns,
            n_cross_attn_heads=self.hparams.n_cross_attn_heads,
            n_latents=self.hparams.n_latents,
            latent_dim=self.hparams.latent_dim,
            n_latent_heads=self.hparams.n_latent_heads,
            n_latent_blocks=self.hparams.n_latent_blocks,
            n_perceiver_blocks=self.hparams.n_perceiver_blocks,
            share_weights=self.hparams.share_weights,
            attn_dropout=self.hparams.attn_dropout,
            ff_dropout=self.hparams.ff_dropout,
            transformer_activation=self.hparams.transformer_activation,
            mlp_hidden_dims=self.hparams.mlp_hidden_dims,
            mlp_activation=self.hparams.mlp_activation,
            mlp_dropout=self.hparams.mlp_dropout,
            mlp_batchnorm=self.hparams.mlp_batchnorm,
            mlp_batchnorm_last=self.hparams.mlp_batchnorm_last,
            mlp_linear_first=self.hparams.mlp_linear_first,
        )
