from abc import ABC, abstractmethod
import pandas as pd
from data_layer.data_module import DataModule


class EvaluationModule(ABC):
    def __init__(self, data: DataModule, hyperparameters: dict = None):
        """

        Parameters
        ----------
        model:
            Classification model. (optional)
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