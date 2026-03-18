from typing import Any, Dict, Optional

import pandas as pd
import yaml
from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from experiment_utils import deep_merge
from method.catalog.CRUDS.library.utils import cruds_search
from method.method_factory import register_method
from method.method_object import MethodObject
from model.catalog.autoencoder.csvae import CSVAE
from model.model_object import ModelObject


@register_method("CRUDS")
class CRUDS(MethodObject):
	"""
	Implementation of CRUDS [1]_.

	.. [1] M. Downs, J. Chu, Yacoby Y, Doshi-Velez F, WeiWei P. CRUDS: Counterfactual Recourse Using Disentangled
            Subspaces. ICML Workshop on Human Interpretability in Machine Learning. 2020 :1-23.
	"""

	def __init__(
		self,
		data: DataObject,
		model: ModelObject,
		config_override: Optional[Dict[str, Any]] = None,
	):
		super().__init__(data, model, config_override=config_override)

		self.config = yaml.safe_load(open("method/catalog/CRUDS/library/config.yml", "r"))
        
		if self._config_override is not None:
			self.config = deep_merge(self.config, self._config_override)

		self._feature_order = self._data.get_feature_names(expanded=True)

		self._target_class = self.config["target_class"]
		self._lambda_param = self.config["lambda_param"]
		self._optimizer = self.config["optimizer"]
		self._lr = self.config["lr"]
		self._max_iter = self.config["max_iter"]
		self._binary_cat_features = self.config["binary_cat_features"]

		vae_params = self.config["vae_params"]
		if vae_params['layers'] is None:
			vae_params['layers'] = [int(sum(self._model.get_mutable_mask())), 16, 8]
		else:
			vae_params['layers'] = [int(sum(self._model.get_mutable_mask()))] + vae_params['layers']

		self._csvae = CSVAE(
			self.config["data_name"],
			vae_params["layers"],
			self._model.get_mutable_mask(),
		)

		if vae_params["train"]:
			self._csvae.fit(
				# add label to the end of the data for training
				data=pd.concat([self._model.get_train_data()[0][self._feature_order], self._model.get_train_data()[1]], axis=1),
				epochs=vae_params["epochs"],
				lr=vae_params["lr"],
				batch_size=vae_params["batch_size"],
			)
		else:
			try:
				self._csvae.load(self._data.get_processed_data().shape[1] - 1)
			except FileNotFoundError as exc:
				raise FileNotFoundError(
					"Loading of Autoencoder failed. {}".format(str(exc))
				)


	def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
		"""
		Generate counterfactuals for input factual instances.
		
		Requires the target column to be included in the kwargs with the key "target_column". 
		This is because the CS-VAE model needs to be trained with the target label as part of the input data, 
		and thus the target column is needed to properly format the input for counterfactual generation.
		"""

		print(f"These are the factuals passed to the method: {factuals}")

		factuals = pd.concat(
			[
				factuals[self._feature_order],
				factuals[self._data.get_target_column()],
			],
			axis=1,
		)

		# pay attention to categorical features
		encoded_feature_names = self._data.get_categorical_features(expanded=True)

		cat_features_indices = []
		for features in encoded_feature_names:
			# Find the indices of these encoded features in the processed dataframe
			indices = [factuals.columns.get_loc(feat) for feat in features]
			cat_features_indices.append(indices)

		df_cfs = factuals.apply(
			lambda x: cruds_search(
				self._model,
				self._csvae,
				x.reshape((1, -1)),
				cat_features_indices,
				self._binary_cat_features,
				self._target_class,
				self._lambda_param,
				self._optimizer,
				self._lr,
				self._max_iter,
			),
			raw=True,
			axis=1,
		)

		cf_df = check_counterfactuals(self._model, self._data, df_cfs, factuals.index)
		cf_df = cf_df[self._feature_order] # ensure the feature ordering is correct for the model input
		return cf_df
		
