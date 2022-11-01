from typing import Any, List, Dict
from torchmetrics import MetricCollection
import wandb
import pytorch_lightning as pl
import torch
from src.tasks.metrics import get_cls_pred_metrics, get_cls_prob_metrics, get_reg_metrics
import numpy as np
import shap
import pandas as pd


def get_model_framework_dict():
    model_framework = {
        "elastic_net": "stand_alone",
        "xgboost": "stand_alone",
        "catboost": "stand_alone",
        "lightgbm": "stand_alone",
        "widedeep_tab_mlp": "pytorch",
        "widedeep_tab_resnet": "pytorch",
        "widedeep_tab_net": "pytorch",
        "widedeep_tab_transformer": "pytorch",
        "widedeep_ft_transformer": "pytorch",
        "widedeep_saint": "pytorch",
        "widedeep_tab_fastformer": "pytorch",
        "widedeep_tab_perceiver": "pytorch",
        "pytorch_tabular_autoint": "pytorch",
        "pytorch_tabular_tabnet": "pytorch",
        "pytorch_tabular_node": "pytorch",
        "pytorch_tabular_category_embedding": "pytorch",
        "pytorch_tabular_ft_transformer": "pytorch",
        "pytorch_tabular_tab_transformer": "pytorch",
        "nbm_spam_spam": "pytorch",
        "nbm_spam_nam": "pytorch",
        "nbm_spam_nbm": "pytorch",
        "arm_net_models": "pytorch",
        "danet": "pytorch",
        "nam": "pytorch",
        "stg": "pytorch",
    }
    return model_framework


class BaseModel(pl.LightningModule):

    def __init__(self,  **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.produce_probabilities = False
        self.produce_importance = False

        if self.hparams.task == "classification":
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            if self.hparams.output_dim < 2:
                raise ValueError(f"Classification with {self.hparams.output_dim} classes")
            self.metrics = get_cls_pred_metrics(self.hparams.output_dim)
            self.metrics = {f'{k}_pl': v for k, v in self.metrics.items()}
            self.metrics_dict = {k:v[0] for k,v in self.metrics.items()}
            self.metrics_prob = get_cls_prob_metrics(self.hparams.output_dim)
            self.metrics_prob = {f'{k}_pl': v for k, v in self.metrics_prob.items()}
            self.metrics_prob_dict =  {k:v[0] for k,v in self.metrics_prob.items()}
        elif self.hparams.task == "regression":
            if self.hparams.loss_type == "MSE":
                self.loss_fn = torch.nn.MSELoss(reduction='mean')
            elif self.hparams.loss_type == "L1Loss":
                self.loss_fn = torch.nn.L1Loss(reduction='mean')
            else:
                raise ValueError("Unsupported loss_type")
            self.metrics = get_reg_metrics()
            self.metrics = {f'{k}_pl': v for k, v in self.metrics.items()}
            self.metrics_dict = {k: v[0] for k, v in self.metrics.items()}
            self.metrics_prob_dict = {}

        self.metrics_trn = MetricCollection(self.metrics_dict)
        self.metrics_trn_prob = MetricCollection(self.metrics_prob_dict)
        self.metrics_val = self.metrics_trn.clone()
        self.metrics_val_prob = self.metrics_trn_prob.clone()
        self.metrics_tst = self.metrics_trn.clone()
        self.metrics_tst_prob = self.metrics_trn_prob.clone()

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure all MaxMetric doesn't store accuracy from these checks
        # self.max_metric.reset()
        pass

    def on_fit_start(self) -> None:
        if wandb.run is not None:
            for stage_type in ['trn', 'val', 'tst']:
                for m in self.metrics:
                    wandb.define_metric(f"{stage_type}/{m}", summary=self.metrics[m][1])
                if self.hparams.task == "classification":
                    for m in self.metrics_prob:
                        wandb.define_metric(f"{stage_type}/{m}", summary=self.metrics_prob[m][1])
                wandb.define_metric(f"{stage_type}/loss", summary='min')

    def calc_out_and_loss(self, out, y, stage):
        loss = self.loss_fn(out, y)
        return out, loss

    def forward(self, batch):
        pass

    def forward_train(self, batch):
        return self.forward(batch)

    def forward_eval(self, batch):
        return self.forward(batch)

    def step(self, batch: Dict, stage:str):
        y = batch["target"]
        batch_size = y.size(0)
        if self.hparams.task == "regression":
            y = y.view(batch_size, -1)

        if stage == "trn":
            out = self.forward_train(batch)
        else:
            out = self.forward_eval(batch)
        out, loss = self.calc_out_and_loss(out, y, stage)

        logs = {"loss": loss}
        non_logs = {}
        if self.hparams.task == "classification":
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(out, dim=1)
            non_logs["preds"] = preds
            non_logs["targets"] = y
            if stage == "trn":
                logs.update(self.metrics_trn(preds, y))
                try:
                    logs.update(self.metrics_trn_prob(probs, y))
                except ValueError:
                    pass
            elif stage == "val":
                logs.update(self.metrics_val(preds, y))
                try:
                    logs.update(self.metrics_val_prob(probs, y))
                except ValueError:
                    pass
            elif stage == "tst":
                logs.update(self.metrics_tst(preds, y))
                try:
                    logs.update(self.metrics_tst_prob(probs, y))
                except ValueError:
                    pass
        elif self.hparams.task == "regression":
            if stage == "trn":
                logs.update(self.metrics_trn(out, y))
            elif stage == "val":
                logs.update(self.metrics_val(out, y))
            elif stage == "tst":
                logs.update(self.metrics_tst(out, y))

        return loss, logs, non_logs

    def training_step(self, batch: Dict, batch_idx: int):
        loss, logs, non_logs = self.step(batch, "trn")
        d = {f"trn/{k}": v for k, v in logs.items()}
        self.log_dict(d, on_step=False, on_epoch=True, logger=True)
        logs.update(non_logs)
        return logs

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Dict, batch_idx: int):
        loss, logs, non_logs = self.step(batch, "val")
        d = {f"val/{k}": v for k, v in logs.items()}
        self.log_dict(d, on_step=False, on_epoch=True, logger=True)
        logs.update(non_logs)
        return logs

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Dict, batch_idx: int):
        loss, logs, non_logs = self.step(batch, "tst")
        d = {f"tst/{k}": v for k, v in logs.items()}
        self.log_dict(d, on_step=False, on_epoch=True, logger=True)
        logs.update(non_logs)
        return logs

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def predict_step(self, batch: Dict, batch_idx):
        out = self.forward(batch)
        return out

    def on_epoch_end(self):
        for m in self.metrics_dict:
            self.metrics_trn[m].reset()
            self.metrics_val[m].reset()
            self.metrics_tst[m].reset()
        for m in self.metrics_prob_dict:
            self.metrics_trn_prob[m].reset()
            self.metrics_val_prob[m].reset()
            self.metrics_tst_prob[m].reset()

    def get_feature_importance(self, data, feature_names, method="shap_kernel"):

        if method.startswith("shap"):

            if self.hparams.task == "regression":

                def predict_func(X):
                    batch = {
                        'all': torch.from_numpy(np.float32(X[:, feature_names['all_ids']])),
                        'continuous': torch.from_numpy(np.float32(X[:, feature_names['con_ids']])),
                        'categorical': torch.from_numpy(np.float32(X[:, feature_names['cat_ids']])),
                    }
                    tmp = self.forward(batch)
                    return tmp.cpu().detach().numpy()

                if method == "shap_kernel":
                    explainer = shap.KernelExplainer(predict_func, data)
                    shap_values = explainer.shap_values(data)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                elif method == "shap_deep":
                    explainer = shap.DeepExplainer(self, torch.from_numpy(data))
                    shap_values = explainer.shap_values(torch.from_numpy(data))
                else:
                    raise ValueError(f"Unsupported feature importance method: {method}")

                importance_values = np.mean(np.abs(shap_values), axis=0)

            elif self.hparams.task == "classification":

                def predict_func(X):
                    self.produce_probabilities = True
                    batch = {
                        'all': torch.from_numpy(np.float32(X[:, feature_names['all_ids']])),
                        'continuous': torch.from_numpy(np.float32(X[:, feature_names['con_ids']])),
                        'categorical': torch.from_numpy(np.float32(X[:, feature_names['cat_ids']])),
                    }
                    tmp = self.forward(batch)
                    return tmp.cpu().detach().numpy()

                if method == "shap_kernel":
                    explainer = shap.KernelExplainer(predict_func, data)
                    shap_values = explainer.shap_values(data)
                elif method == "shap_deep":
                    explainer = shap.DeepExplainer(self, torch.from_numpy(data))
                    shap_values = explainer.shap_values(torch.from_numpy(data))
                else:
                    raise ValueError(f"Unsupported feature importance method: {method}")

                importance_values = np.zeros(len(feature_names['all']))
                for cl_id in range(len(shap_values)):
                    importance_values += np.mean(np.abs(shap_values[cl_id]), axis=0)

            else:
                raise ValueError("Unsupported task")

        elif method == "none":
            importance_values = np.zeros(len(feature_names['all']))
        else:
            raise ValueError(f"Unsupported feature importance method: {method}")

        feature_importances = pd.DataFrame.from_dict(
            {
                'feature': feature_names['all'],
                'importance': importance_values
            }
        )

        return feature_importances

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.optimizer_lr,
            weight_decay=self.hparams.optimizer_weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.hparams.scheduler_step_size,
            gamma=self.hparams.scheduler_gamma
        )

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        )
