from typing import List, Tuple, Union
from numpy import linalg as LA

import numpy as np
import pandas as pd
import torch
import logging
import yaml
from data.data_object import DataObject
from method.method_factory import register_method
from evaluation.utils import check_counterfactuals
from experiment_utils import deep_merge, reconstruct_encoding_constraints
from method.method_object import MethodObject
from model.catalog.autoencoder.vae import VariationalAutoencoder
from model.model_object import ModelObject

@register_method("CCHVAE")
class CCHVAE(MethodObject):
    """
    Implementation of CCHVAE [1]_

    .. [1] Pawelczyk, Martin, Klaus Broelemann and Gjergji Kasneci. “Learning Model-Agnostic Counterfactual Explanations
          for Tabular Data.” Proceedings of The Web Conference 2020 (2020): n. pag..
    """
    def __init__(self, data: DataObject, model: ModelObject, config_override = None):
        super().__init__(data, model, config_override)

        # get configs from config file
        self.config = yaml.safe_load(open("method/catalog/CCHVAE/library/config.yml", 'r'))
        
        # merge configs with user specified, if they exist
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

        self._feature_order = self._data.get_feature_names(expanded=True)

        self._n_search_samples = self.config["n_search_samples"]
        self._p_norm = self.config["p_norm"]
        self._step = self.config["step"]
        self._max_iter = self.config["max_iter"]
        self._clamp = self.config["clamp"]

        vae_params = self.config["vae_params"]
        self._vae = VariationalAutoencoder(
            data_name = self.config['data_name'] if self.config['data_name'] else "Temp",
            layers=[int(sum(self._model.get_mutable_mask()))] + vae_params['layers'],
            mutable_mask=self._model.get_mutable_mask(),
        )

        if vae_params["train"]:
            self._vae.fit(
                xtrain=self._model.get_train_data()[0][self._feature_order],
                kl_weight=vae_params["kl_weight"],
                lambda_reg=vae_params["lambda_reg"],
                epochs=vae_params["epochs"],
                lr=vae_params["lr"],
                batch_size=vae_params["batch_size"],
            )
        else:
            try:
                self._vae.load(vae_params["layers"][0])
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Loading of Autoencoder failed. {}".format(str(exc))
                )
            
    def _hyper_sphere_coordindates(
        self, instance, high: int, low: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param n_search_samples: int > 0
        :param instance: numpy input point array
        :param high: float>= 0, h>l; upper bound
        :param low: float>= 0, l<h; lower bound
        :param p: float>= 1; norm
        :return: candidate counterfactuals & distances
        """
        delta_instance = np.random.randn(self._n_search_samples, instance.shape[1])
        dist = (
            np.random.rand(self._n_search_samples) * (high - low) + low
        )  # length range [l, h)
        norm_p = LA.norm(delta_instance, ord=self._p_norm, axis=1)
        d_norm = np.divide(dist, norm_p).reshape(-1, 1)  # rescale/normalize factor
        delta_instance = np.multiply(delta_instance, d_norm)
        candidate_counterfactuals = instance + delta_instance
        return candidate_counterfactuals, dist

    def _counterfactual_search(
        self, step: int, factual: torch.Tensor, cat_features_indices: List[list[int]]
    ) -> pd.DataFrame:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # init step size for growing the sphere
        low = 0
        high = step
        # counter
        count = 0
        counter_step = 1

        torch_fact = torch.from_numpy(factual).to(device)

        # get predicted label of instance
        instance_label = np.argmax(
            self._model.predict_proba(torch_fact.float()).cpu().detach().numpy(),
            axis=1,
        )

        # vectorize z
        z = self._vae.encode(
            torch_fact[:, self._vae.mutable_mask].float()
        )[0]
        # add the immutable features to the latents
        z = torch.cat([z, torch_fact[:, ~self._vae.mutable_mask]], dim=-1)
        z = z.cpu().detach().numpy()
        z_rep = np.repeat(z.reshape(1, -1), self._n_search_samples, axis=0)

        # make copy such that we later easily combine the immutables and the reconstructed mutables
        fact_rep = torch_fact.reshape(1, -1).repeat_interleave(
            self._n_search_samples, dim=0
        )

        candidate_dist: List = []
        x_ce: Union[np.ndarray, torch.Tensor] = np.array([])
        while count <= self._max_iter or len(candidate_dist) <= 0:
            count = count + counter_step
            if count > self._max_iter:
                logging.debug("No counterfactual example found")
                return x_ce[0]

            # STEP 1 -- SAMPLE POINTS on hyper sphere around instance
            latent_neighbourhood, _ = self._hyper_sphere_coordindates(z_rep, high, low)
            torch_latent_neighbourhood = (
                torch.from_numpy(latent_neighbourhood).to(device).float()
            )
            x_ce = self._vae.decode(torch_latent_neighbourhood)

            # add the immutable features to the reconstruction
            temp = fact_rep.clone()
            temp[:, self._vae.mutable_mask] = x_ce.float()
            x_ce = temp

            x_ce = reconstruct_encoding_constraints(
                x_ce, cat_features_indices
            )
            x_ce = x_ce.detach().cpu().numpy()
            x_ce = x_ce.clip(0, 1) if self._clamp else x_ce

            # STEP 2 -- COMPUTE l1 & l2 norms
            if self._p_norm == 1:
                distances = np.abs((x_ce - torch_fact.cpu().detach().numpy())).sum(
                    axis=1
                )
            elif self._p_norm == 2:
                distances = LA.norm(x_ce - torch_fact.cpu().detach().numpy(), axis=1)
            else:
                raise ValueError("Possible values for p_norm are 1 or 2")

            # counterfactual labels
            y_candidate = np.argmax(
                self._model.predict_proba(torch.from_numpy(x_ce).float())
                .cpu()
                .detach()
                .numpy(),
                axis=1,
            )
            indices = np.where(y_candidate != instance_label)
            candidate_counterfactuals = x_ce[indices]
            candidate_dist = distances[indices]
            # no candidate found & push search range outside
            if len(candidate_dist) == 0:
                low = high
                high = low + step
            elif len(candidate_dist) > 0:
                # certain candidates generated
                min_index = np.argmin(candidate_dist)
                logging.debug("Counterfactual example found")
                return candidate_counterfactuals[min_index]

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        factuals = factuals[self._feature_order] # ensure the feature ordering is correct for the model input

        # pay attention to categorical features
        encoded_feature_names = self._data.get_categorical_features(expanded=True)

        cat_features_indices = []
        for features in encoded_feature_names:
            # Find the indices of these encoded features in the processed dataframe
            indices = [factuals.columns.get_loc(feat) for feat in features]
            cat_features_indices.append(indices)

        df_cfs = factuals.apply(
            lambda x: self._counterfactual_search(
                self._step, x.reshape((1, -1)), cat_features_indices
            ),
            raw=True,
            axis=1,
        )

        df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index) 
        return df_cfs

