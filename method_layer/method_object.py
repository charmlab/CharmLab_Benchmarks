from abc import ABC, abstractmethod

import pandas as pd

from data_layer.data_object import DataObject
from model_layer.model_object import ModelObject


class MethodObject(ABC):
    """
    Abstract class to implement custom recourse methods for a given black-box-model.

    Parameters
    ----------
    data: data_layer.DataObject
        The data object containing the processed data and metadata.
    model: model_layer.ModelObject
        The model module containing the trained model and its configuration.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
   
    """

    def __init__(self, data: DataObject, model: ModelObject):
        self._data = data
        self._model = model

    @abstractmethod
    def get_counterfactuals(self, factuals: pd.DataFrame):
        """
        Generate counterfactual examples for given factuals.
        """
        pass