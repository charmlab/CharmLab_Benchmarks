import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from experiment_utils import deep_merge
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject

@register_method("GRAVITATIONAL")
class Gravitational(MethodObject):
    """
    Implementation of Gravitational Recourse Algorithm [1]_.


    .. [1] "Endogenous Macrodynamics in Algorithmic Recourse"
        Patrick Altmeyer, Giovan Angela, Karol Dobiczek, Arie van Deursen, Cynthia C. S. Liem
    """
    def __init__(self, data: DataObject, model: ModelObject, x_center: np.array = None, config_override = None):
        super().__init__(data, model, config_override)

        # get configs from config file
        self.config = yaml.safe_load(open("method/catalog/GRAVITATIONAL/library/config.yml", 'r'))
        
        # merge configs with user specified, if they exist
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._feature_order = self._data.get_feature_names(expanded=True)

        self._prediction_loss_lambda = self.config['prediction_loss_lambda']
        self._original_dist_lambda = self.config['original_dist_lambda']
        self._grav_penalty_lambda = self.config['grav_penalty_lambda']
        self._lr = self.config["lr"]
        self._num_steps = self.config["num_steps"]
        self._target_class = self.config["target_class"]
        self._scheduler_step_size = self.config["scheduler_step_size"]
        self._scheduler_gamma = self.config["scheduler_gamma"]

        self.x_center = x_center

        if self.x_center is None:
            x_train, y_train = self._model.get_train_data()
            mask = y_train == self._target_class
            if mask.any():
                x_center = x_train[mask].mean(axis=0).to_numpy(dtype=np.float32)
            else:
                y_pred = self._model.predict(x_train).squeeze() == self._target_class
                if np.asarray(y_pred).sum() > 0:
                    x_center = x_train[y_pred].mean(axis=0).to_numpy(dtype=np.float32)
                else:
                    x_center = x_train.mean(axis=0).to_numpy(dtype=np.float32)

            x_center = np.nan_to_num(x_center, nan=0.0, posinf=1e6, neginf=-1e6)
            self.x_center = x_center

        self._criterion = nn.BCELoss()
    
    def prediction_loss(self, x_cf):
        x_cf = x_cf.to(self.device)
        output = self._model.predict_proba(x_cf)[:, 1] # TODO: Since this outputs probabilities and not logits, should the loss be BCEloss?
        target_class = torch.tensor(
            [self._target_class] * output.size(0), dtype=torch.float32
        ).to(self.device)
        loss = self._criterion(output, target_class)
        return loss

    def cost(self, x_original, x_cf):
        return torch.norm(x_original - x_cf)

    def gravitational_penalty(self, x_cf, x_center):
        return torch.norm(x_cf - torch.tensor(x_center, dtype=torch.float32))

    def get_counterfactuals(self, factuals: pd.DataFrame):
        factuals = factuals.reset_index()
        factuals = factuals[self._feature_order]
        x_cf = torch.tensor(factuals.values, dtype=torch.float32, requires_grad=True)

        optimizer = optim.Adam([x_cf], lr=self._lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self._scheduler_step_size, gamma=self._scheduler_gamma
        )

        for step in range(self._num_steps):
            optimizer.zero_grad()

            prediction_loss_value = self.prediction_loss(x_cf)
            original_dist = self.cost(
                torch.tensor(factuals.values, dtype=torch.float32), x_cf
            )
            grav_penalty = self.gravitational_penalty(x_cf, self.x_center)

            loss = (
                self._prediction_loss_lambda * prediction_loss_value
                + self._original_dist_lambda * original_dist
                + self._grav_penalty_lambda * grav_penalty
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

        x_cf = x_cf.detach().numpy()
        x_cf_df = pd.DataFrame(x_cf, columns=factuals.columns)
        df_cfs = check_counterfactuals(self._model, self._data, x_cf_df, factuals.index) 
        
        return df_cfs

    def set_x_center(self, x_center):
        self.x_center = x_center

    def reset_x_center(self):
        x_train, y_train = self._model.get_train_data()
        self.x_center = np.mean(x_train[y_train == self._target_class], axis=0)
        return self.x_center
    

        

