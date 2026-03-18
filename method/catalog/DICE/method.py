from typing import Any, Dict, Optional

import pandas as pd

from data.data_object import DataObject
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject


@register_method("DICE")
class Dice(MethodObject):
	"""
	Boilerplate scaffold for the DICE recourse method.

	This class defines the standard plugin contract used by the method layer.
	"""

	def __init__(
		self,
		data: DataObject,
		model: ModelObject,
		config_override: Optional[Dict[str, Any]] = None,
	):
		super().__init__(data, model, config_override=config_override)
		self.config = self._config_override if self._config_override is not None else {}
		self._feature_order = self._data.get_feature_names(expanded=True)

	def get_counterfactuals(self, factuals: pd.DataFrame):
		"""Generate counterfactuals for input factual instances."""
		factuals = factuals[self._feature_order]
		raise NotImplementedError("DICE method is not implemented yet.")
