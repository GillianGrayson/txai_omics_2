import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from src.utils import utils
import plotly.graph_objects as go
from src.utils.plot.save import save_figure
from src.utils.plot.layout import add_layout
import plotly.express as px


log = utils.get_logger(__name__)


def explain_samples(config, y_real, y_pred, indexes, shap_values, base_value, features, feature_names, path):
    Path(f"{path}").mkdir(parents=True, exist_ok=True)
    y_diff = np.array(y_pred) - np.array(y_real)
    order = np.argsort(y_diff)
    order_abs = np.argsort(np.abs(y_diff))
    num_examples = config.num_examples

    # Select samples with the biggest positive difference, the biggest positive difference, the smallest difference
    ids = list(set(np.concatenate((order[0:num_examples], order[-num_examples:], order_abs[0:num_examples]))))
    log.info(f"Number of samples: {len(ids)}")
    for m_id in ids:
        diff = y_diff[m_id]
        log.info(f"Plotting sample {m_id}: {indexes[m_id]} (real = {y_real[m_id]:0.4f}, estimated = {y_pred[m_id]:0.4f}) with diff = {diff:0.4f}")

        if isinstance(indexes[m_id], str):
            ind_save = indexes[m_id].replace('/', '_')
        else:
            ind_save = indexes[m_id]
        Path(f"{path}/{ind_save}_{diff:0.4f}").mkdir(parents=True, exist_ok=True)

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[m_id],
                base_values=base_value,
                data=features[m_id],
                feature_names=feature_names
            ),
            # max_display=config.num_top_features,
            show=False,
        )
        fig = plt.gcf()
        plt.title(f"{indexes[m_id]}: Real = {y_real[m_id]:0.4f}, Estimated = {y_pred[m_id]:0.4f}", {'fontsize': 20})
        fig.savefig(f"{path}/{ind_save}_{diff:0.4f}/waterfall.pdf", bbox_inches='tight')
        fig.savefig(f"{path}/{ind_save}_{diff:0.4f}/waterfall.png", bbox_inches='tight')
        plt.close()

        shap.plots.decision(
            base_value=base_value,
            shap_values=shap_values[m_id],
            features=features[m_id],
            feature_names=feature_names,
            show=False,
        )
        fig = plt.gcf()
        plt.title(f"{indexes[m_id]}: Real = {y_real[m_id]:0.4f}, Estimated = {y_pred[m_id]:0.4f}", {'fontsize': 20})
        fig.savefig(f"{path}/{ind_save}_{diff:0.4f}/decision.pdf", bbox_inches='tight')
        fig.savefig(f"{path}/{ind_save}_{diff:0.4f}/decision.png", bbox_inches='tight')
        plt.close()

        shap.plots.force(
            base_value=base_value,
            shap_values=shap_values[m_id],
            features=features[m_id],
            feature_names=feature_names,
            show=False,
            matplotlib=True
        )
        fig = plt.gcf()
        fig.savefig(f"{path}/{ind_save}_{diff:0.4f}/force.pdf", bbox_inches='tight')
        fig.savefig(f"{path}/{ind_save}_{diff:0.4f}/force.png", bbox_inches='tight')
        plt.close()


def explain_shap(config, expl_data):
    model = expl_data['model']
    predict_func = expl_data['predict_func']
    df = expl_data['df']
    feature_names = expl_data['feature_names']
    target_name = expl_data['target_name']

    if config.shap_explainer == 'Tree' and config.shap_bkgrd == 'tree_path_dependent':
        explainer = shap.TreeExplainer(model)
    else:
        ids_bkgrd = expl_data["ids"][config.shap_bkgrd]
        indexes_bkgrd = df.index[ids_bkgrd]
        X_bkgrd = df.loc[indexes_bkgrd, feature_names].values
        if config.shap_explainer == 'Tree':
            explainer = shap.TreeExplainer(model, data=X_bkgrd, feature_perturbation='interventional')
        elif config.shap_explainer == "Kernel":
            explainer = shap.KernelExplainer(predict_func, X_bkgrd)
        elif config.shap_explainer == "Deep":
            explainer = shap.DeepExplainer(model, torch.from_numpy(X_bkgrd))
        else:
            raise ValueError(f"Unsupported explainer type: {config.shap_explainer}")

    for part in expl_data["ids"]:
        if expl_data["ids"][part] is not None and len(expl_data["ids"][part]) > 0:
            log.info(f"Calculating SHAP for {part}")
            Path(f"shap/{part}/global").mkdir(parents=True, exist_ok=True)
            ids = expl_data["ids"][part]
            indexes = df.index[ids]
            X = df.loc[indexes, feature_names].values
            y_pred = df.loc[indexes, "Estimation"].values

            if config.shap_explainer == "Tree":
                shap_values = explainer.shap_values(X)
                expected_value = explainer.expected_value
            elif config.shap_explainer == "Kernel":
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[0]
            elif config.shap_explainer == "Deep":
                shap_values = explainer.shap_values(torch.from_numpy(X))
                expected_value = explainer.expected_value[0]
            else:
                raise ValueError(f"Unsupported explainer type: {config.shap_explainer}")

            if config.is_shap_save:
                df_shap = pd.DataFrame(index=indexes, columns=feature_names, data=shap_values)
                df_shap.index.name = 'index'
                df_shap.to_excel(f"shap/{part}/shap.xlsx", index=True)

            shap.summary_plot(
                shap_values=shap_values,
                features=X,
                feature_names=feature_names,
                # max_display=config.num_top_features,
                plot_type="bar",
                show=False,
            )
            plt.savefig(f'shap/{part}/global/bar.png', bbox_inches='tight')
            plt.savefig(f'shap/{part}/global/bar.pdf', bbox_inches='tight')
            plt.close()

            shap.summary_plot(
                shap_values=shap_values,
                features=X,
                feature_names=feature_names,
                # max_display=config.num_top_features,
                plot_type="violin",
                show=False,
            )
            plt.savefig(f"shap/{part}/global/beeswarm.png", bbox_inches='tight')
            plt.savefig(f"shap/{part}/global/beeswarm.pdf", bbox_inches='tight')
            plt.close()

            explanation = shap.Explanation(
                values=shap_values,
                base_values=np.array([expected_value] * len(ids)),
                data=X,
                feature_names=feature_names
            )
            shap.plots.heatmap(
                explanation,
                show=False,
                # max_display=config.num_top_features,
                instance_order=explanation.sum(1)
            )
            plt.savefig(f"shap/{part}/global/heatmap.png", bbox_inches='tight')
            plt.savefig(f"shap/{part}/global/heatmap.pdf", bbox_inches='tight')
            plt.close()

            Path(f"shap/{part}/features").mkdir(parents=True, exist_ok=True)
            mean_abs_impact = np.mean(np.abs(shap_values), axis=0)
            features_order = np.argsort(mean_abs_impact)[::-1]
            feat_ids_to_plot = features_order[0:config.num_top_features]
            for rank, feat_id in enumerate(feat_ids_to_plot):
                feat = feature_names[feat_id]
                shap.dependence_plot(
                    ind=feat_id,
                    shap_values=shap_values,
                    features=X,
                    feature_names=feature_names,
                    show=False,
                )
                plt.savefig(f"shap/{part}/features/{rank}_{feat}.png", bbox_inches='tight')
                plt.savefig(f"shap/{part}/features/{rank}_{feat}.pdf", bbox_inches='tight')
                plt.close()

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=X[:, feat_id],
                        y=shap_values[:, feat_id],
                        showlegend=False,
                        name=feat,
                        mode='markers',
                        marker=dict(
                            size=10,
                            opacity=0.7,
                            line=dict(
                                width=1
                            ),
                            color=y_pred,
                            colorscale=px.colors.sequential.Bluered,
                            showscale=True,
                            colorbar=dict(title=dict(text="Estimation", font=dict(size=20)), tickfont=dict(size=20))
                        )
                    )
                )
                add_layout(fig, feat, f"SHAP value for<br>{feat}", f"", font_size=20)
                fig.update_layout(legend_font_size=20)
                fig.update_layout(
                    margin=go.layout.Margin(
                        l=120,
                        r=20,
                        b=80,
                        t=25,
                        pad=0
                    )
                )
                save_figure(fig, f"shap/{part}/features/{rank}_{feat}_scatter")

            explain_samples(
                config,
                df.loc[indexes, target_name].values,
                df.loc[indexes, "Estimation"].values,
                indexes,
                shap_values,
                expected_value,
                X,
                feature_names,
                f"shap/{part}/samples"
            )
