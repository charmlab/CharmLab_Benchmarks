from typing import Any, Dict, Optional
import yaml
from experiment_utils import deep_merge
from data.data_object import DataObject
from method.method_object import MethodObject
from model.model_object import ModelObject


class TreX(MethodObject):
    """
    Implementation of TreX [1]_.

    .. [1] Hamman, Faisal and Noorani, Erfaun and Mishra, Saumitra and Magazzeni, Daniele and Dutta, Sanghamitra
    Robust Counterfactual Explanations for Neural Networks With Probabilistic Guarantees
    """

    def __init__(self, data: DataObject, 
                model: ModelObject,
                config_override: Optional[Dict[str, Any]] = None):
        super().__init__(data, model, config_override)

        self.config = yaml.safe_load(open("method/catalog/TREX/library/config.yml", 'r'))
        
        # merge configs with user specified, if they exist
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

        # store the feature ordering
        self._feature_order = self._data.get_feature_names(expanded=True) # ensure the feature ordering is correct for the model input
        