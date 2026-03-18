from typing import Any, Dict, Optional

import os

import numpy as np
import pandas as pd
import torch
import yaml

from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from experiment_utils import deep_merge
from method.catalog.CLUE.library.clue_ml.AE.fc_gauss_cat import VAE_gauss_cat_net
from method.catalog.CLUE.library.clue_ml.AE.vae_training import training
from method.catalog.CLUE.library.clue_ml.Clue_model.CLUE_counterfactuals import vae_gradient_search
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject


@register_method("CLUE")
class CLUE(MethodObject):
    """
    Implementation of CLUE from Antorán et.al. [1]_.

    .. [1] Javier Antorán, Umang Bhatt, Tameem Adel, Adrian Weller, and
           José Miguel Hernández-Lobato. Getting a CLUE: A Method for
           Explaining Uncertainty Estimates. ICLR.
    """

    def __init__(
        self,
        data: DataObject,
        model: ModelObject,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(data, model, config_override=config_override)

        backend = (
            self._model._config.get("backend", "").lower()
            if hasattr(self._model, "_config")
            else ""
        )
        if backend and backend != "pytorch":
            raise ValueError(
                f"CLUE currently supports only PyTorch backend, got: {backend}"
            )

        self.config = yaml.safe_load(open("method/catalog/CLUE/library/config.yml", "r"))
        
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

        self._feature_order = self._data.get_feature_names(expanded=True)

        self._train_vae_flag = bool(self.config["train_vae"])
        self._width = int(self.config["width"])
        self._depth = int(self.config["depth"])
        self._latent_dim = int(self.config["latent_dim"])
        self._data_name = self.config["data_name"] or "clue"
        self._batch_size = int(self.config["batch_size"])
        self._epochs = int(self.config["epochs"])
        self._lr = float(self.config["lr"])
        self._early_stop = int(self.config["early_stop"])

        self._continuous = self._data.get_continuous_features()
        categorical_groups = self._data.get_categorical_features(expanded=True)
        self._categorical = [feat for group in categorical_groups for feat in group]

        input_dims_continuous = list(np.repeat(1, len(self._continuous)))
        input_dims_binary = list(np.repeat(1, len(self._categorical)))
        self._input_dimension = input_dims_continuous + input_dims_binary

        if len(self._input_dimension) != len(self._feature_order):
            self._input_dimension = list(np.repeat(1, len(self._feature_order)))

        self._vae = self._load_vae()

    def _load_vae(self):
        path = os.environ.get(
            "CF_MODELS",
            os.path.join(
                "~",
                "carla",
                "models",
                "autoencoders",
                "clue",
                f"fc_VAE_{self._data_name}_models",
            ),
        )
        path = os.path.expanduser(path)
        os.makedirs(path, exist_ok=True)

        theta_path = os.path.join(path, "theta_best.dat")

        if self._train_vae_flag:
            self._train_vae(path)
        elif not os.path.isfile(theta_path):
            raise ValueError(
                'No pre-trained VAE available. Please set "train_vae" to true in config to train a VAE.'
            )

        if not os.path.isfile(theta_path):
            raise FileNotFoundError(
                f'Expected trained VAE weights at "{theta_path}", but none were found.'
            )

        flat_vae_bools = False
        cuda = torch.cuda.is_available()
        vae = VAE_gauss_cat_net(
            self._input_dimension,
            self._width,
            self._depth,
            self._latent_dim,
            pred_sig=False,
            lr=self._lr,
            cuda=cuda,
            flatten=flat_vae_bools,
        )
        vae.load(theta_path)
        return vae

    def _train_vae(self, path: str):
        x_train, _ = self._model.get_train_data()
        x_test, _ = self._model.get_test_data()

        if isinstance(x_train, pd.DataFrame):
            x_train = x_train[self._feature_order].to_numpy(dtype=np.float32)
        else:
            x_train = np.asarray(x_train, dtype=np.float32)

        if isinstance(x_test, pd.DataFrame):
            x_test = x_test[self._feature_order].to_numpy(dtype=np.float32)
        else:
            x_test = np.asarray(x_test, dtype=np.float32)

        training(
            x_train,
            x_test,
            self._input_dimension,
            path,
            self._width,
            self._depth,
            self._latent_dim,
            self._batch_size,
            self._epochs,
            self._lr,
            self._early_stop,
        )

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        factuals = factuals[self._feature_order]

        list_cfs = []
        for _, row in factuals.iterrows():
            counterfactual = vae_gradient_search(row.values, self._model, self._vae)

            if isinstance(counterfactual, torch.Tensor):
                counterfactual = counterfactual.detach().cpu().numpy()

            counterfactual = np.asarray(counterfactual).reshape(-1)
            list_cfs.append(counterfactual)

        df_cfs = pd.DataFrame(
            np.asarray(list_cfs),
            columns=self._feature_order,
            index=factuals.index,
        )
        df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index)
        return df_cfs