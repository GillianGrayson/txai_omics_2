import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from src.datamodules.tabular import TabularDataModule
from src.utils import utils
import statsmodels.formula.api as smf
import xgboost as xgb
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout
from src.tasks.routines import eval_regression
from catboost import CatBoost
from scripts.python.routines.plot.scatter import add_scatter_trace
import plotly.express as px
from src.tasks.regression.shap import explain_shap
from src.tasks.regression.lime import explain_lime
from scipy.stats import mannwhitneyu


log = utils.get_logger(__name__)

def inference(config: DictConfig):

    if "seed" in config:
        seed_everything(config.seed)

    if 'wandb' in config.logger:
        config.logger.wandb["project"] = config.project_name

    # Init Lightning datamodule for test
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: TabularDataModule = hydra.utils.instantiate(config.datamodule)
    feature_names = datamodule.get_feature_names()
    num_features = len(feature_names['all'])
    config.in_dim = num_features
    target_name = datamodule.get_target()
    df = datamodule.get_data()

    df = df[df[config.data_part_column].notna()]
    data_parts = df[config.data_part_column].dropna().unique()
    data_part_main = config.data_part_main
    data_parts = [data_part_main] + list(set(data_parts) - set([data_part_main]))
    indexes = {}
    X = {}
    y = {}
    y_pred = {}
    colors = {}
    for data_part_id, data_part in enumerate(data_parts):
        indexes[data_part] = df.loc[df[config.data_part_column] == data_part, :].index.values
        X[data_part] = df.loc[indexes[data_part], feature_names['all']].values
        y[data_part] = df.loc[indexes[data_part], target_name].values
        colors[data_part] = px.colors.qualitative.Light24[data_part_id]

    if config.model_framework == "pytorch":
        config.model = config[config.model_type]

        widedeep = datamodule.get_widedeep()
        embedding_dims = [(x[1], x[2]) for x in widedeep['cat_embed_input']] if widedeep['cat_embed_input'] else []
        if config.model_type.startswith('widedeep'):
            config.model.column_idx = widedeep['column_idx']
            config.model.cat_embed_input = widedeep['cat_embed_input']
            config.model.continuous_cols = widedeep['continuous_cols']
        elif config.model_type.startswith('pytorch_tabular'):
            config.model.continuous_cols = feature_names['con']
            config.model.categorical_cols = feature_names['cat']
            config.model.embedding_dims = embedding_dims
        elif config.model_type == 'nam':
            num_unique_vals = [len(np.unique(X[data_part_main][:, i])) for i in range(X[data_part_main].shape[1])]
            num_units = [min(config.model.num_basis_functions, i * config.model.units_multiplier) for i in num_unique_vals]
            config.model.num_units = num_units
        log.info(f"Instantiating model <{config.model._target_}>")
        model = hydra.utils.instantiate(config.model)

        model = type(model).load_from_checkpoint(checkpoint_path=f"{config.path_ckpt}")
        model.eval()
        model.freeze()

        for data_part in data_parts:
            y_pred[data_part] = model(torch.from_numpy(X[data_part])).cpu().detach().numpy().ravel()

        def predict_func(X):
            batch = {
                'all': torch.from_numpy(np.float32(X[:, feature_names['all_ids']])),
                'continuous': torch.from_numpy(np.float32(X[:, feature_names['con_ids']])),
                'categorical': torch.from_numpy(np.float32(X[:, feature_names['cat_ids']])),
            }
            tmp = model(batch)
            return tmp.cpu().detach().numpy()

    elif config.model_framework == "stand_alone":
        if config.model_type == "xgboost":
            model = xgb.Booster()
            model.load_model(config.path_ckpt)

            for data_part in data_parts:
                dmat = xgb.DMatrix(X[data_part], y[data_part], feature_names=feature_names['all'])
                y_pred[data_part] = model.predict(dmat)

            def predict_func(X):
                X = xgb.DMatrix(X, feature_names=feature_names['all'])
                y = model.predict(X)
                return y

        elif config.model_type == "catboost":
            model = CatBoost()
            model.load_model(config.path_ckpt)

            for data_part in data_parts:
                y_pred[data_part] = model.predict(X[data_part]).astype('float32')

            def predict_func(X):
                y = model.predict(X)
                return y

        elif config.model_type == "lightgbm":
            model = lgb.Booster(model_file=config.path_ckpt)

            for data_part in data_parts:
                y_pred[data_part] = model.predict(X[data_part], num_iteration=model.best_iteration).astype('float32')

            def predict_func(X):
                y = model.predict(X, num_iteration=model.best_iteration)
                return y

        else:
            raise ValueError(f"Model {config.model_type} is not supported")

    else:
        raise ValueError(f"Unsupported model_framework: {config.model_framework}")

    for data_part in data_parts:
        df.loc[indexes[data_part], "Estimation"] = y_pred[data_part]
        eval_regression(config, y[data_part], y_pred[data_part], None, data_part, is_log=False, is_save=True, file_suffix=f"")

    formula = f"Estimation ~ {target_name}"
    model_linear = smf.ols(formula=formula, data=df.loc[indexes[data_part_main], :]).fit()
    fig = go.Figure()
    for data_part in data_parts:
        df.loc[indexes[data_part], "Estimation acceleration"] = df.loc[indexes[data_part], "Estimation"].values - model_linear.predict(df.loc[indexes[data_part], :])
        if data_part == data_part_main:
            add_scatter_trace(fig, df.loc[indexes[data_part], target_name].values, df.loc[indexes[data_part], "Estimation"].values, data_part)
            add_scatter_trace(fig, df.loc[indexes[data_part], target_name].values, model_linear.fittedvalues.values, "", "lines")
        else:
            add_scatter_trace(fig, df.loc[indexes[data_part], target_name].values, df.loc[indexes[data_part], "Estimation"].values, data_part)
    add_layout(fig, target_name, f"Estimation", f"")
    fig.update_layout({'colorway': [colors[data_part_main]] + [colors[data_part] for data_part in data_parts]})
    fig.update_layout(legend_font_size=20)
    fig.update_layout(margin=go.layout.Margin(l=90, r=20, b=80, t=65, pad=0))
    save_figure(fig, f"scatter")

    fig = go.Figure()
    dist_num_bins = 15
    df_mw = pd.DataFrame(data=np.zeros(shape=(len(data_parts) - 1, 2)),index=list(set(data_parts) - set([data_part_main])), columns=["stat", "pval"])
    for data_part in data_parts:
        fig.add_trace(
            go.Violin(
                y=df.loc[indexes[data_part], "Estimation acceleration"].values,
                name=data_part,
                box_visible=True,
                meanline_visible=True,
                showlegend=True,
                line_color='black',
                fillcolor=colors[data_part],
                marker=dict(color=colors[data_part], line=dict(color='black', width=0.3), opacity=0.8),
                points='all',
                bandwidth=np.ptp(df.loc[indexes[data_part], "Estimation acceleration"].values) / dist_num_bins,
                opacity=0.8
            )
        )
        add_layout(fig, "", "Estimation acceleration", f"")


        if data_part != data_part_main:
            df_mw.at[data_part, "stat"], df_mw.at[data_part, "pval"] = mannwhitneyu(
                df.loc[indexes[data_part_main], "Estimation acceleration"].values,
                df.loc[indexes[data_part], "Estimation acceleration"].values,
                alternative='two-sided'
            )

    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_layout(
        margin=go.layout.Margin(
            l=110,
            r=20,
            b=50,
            t=90,
            pad=0
        )
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.25,
            xanchor="center",
            x=0.5
        )
    )
    save_figure(fig, f"violin")
    df_mw.to_excel("df_mw.xlsx", index=True)

    df['ids'] = np.arange(df.shape[0])
    ids = {'all': df['ids']}
    for data_part in data_parts:
        ids[data_part] = df.loc[indexes[data_part], 'ids'].values

    expl_data = {
        'model': model,
        'predict_func': predict_func,
        'df': df,
        'feature_names': feature_names['all'],
        'target_name': target_name,
        'ids': ids
    }
    if config.is_lime == True:
        explain_lime(config, expl_data)
    if config.is_shap == True:
        explain_shap(config, expl_data)

    df.to_excel("df.xlsx", index=True)


