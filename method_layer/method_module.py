from abc import ABC, abstractmethod

import pandas as pd

from data_layer.data_module import DataModule
from model_layer.model_module import ModelModule


class MethodModule(ABC):
    """
    Abstract class to implement custom recourse methods for a given black-box-model.

    Parameters
    ----------
    data: data_layer.DataModule
        The data module containing the processed data and metadata.
    model: model_layer.ModelModule
        The model module containing the trained model and its configuration.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
   
    """

    def __init__(self, data: DataModule, model: ModelModule):
        self._data = data
        self._model = model

    @abstractmethod
    def get_counterfactuals(self, factuals: pd.DataFrame):
        """
        Generate counterfactual examples for given factuals.
        """
        pass