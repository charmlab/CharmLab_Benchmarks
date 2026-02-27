from abc import ABC, abstractmethod
import pandas as pd
from data_layer.data_object import DataObject


class EvaluationObject(ABC):
    def __init__(self, data: DataObject, hyperparameters: dict = None):
        """

        Parameters
        ----------
        data: DataObject
            The data object containing the processed data and metadata.
        hyperparameters:
            Dictionary with hyperparameters, could be used to pass other things. (optional)
        """
        self.data = data 
        self.hyperparameters = hyperparameters
    
    @abstractmethod
    def get_evaluation(
        self, factuals: pd.DataFrame, counterfactuals: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute evaluation measure"""
        pass