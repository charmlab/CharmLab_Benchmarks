import numpy as np
import pandas as pd
import yaml
from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from experiment_utils import deep_merge
from method.catalog.FACE.library.utils import graph_search
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject

@register_method("FACE")
class Face(MethodObject):
    """
    Implementation of FACE from Poyiadzi et.al. [1]_.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "mode": {"knn", "epsilon"},
            Decides over type of FACE
        * "fraction": float [0 < x < 1]
            determines fraction of data set to be used to construct neighbourhood graph

    .. [1] Rafael Poyiadzi, Kacper Sokol, Raul Santos-Rodriguez, Tijl De Bie, and Peter Flach. 2020. In Proceedings
            of the AAAI/ACM Conference on AI, Ethics, and Society (AIES)
    """

    def __init__(self, data: DataObject, model: ModelObject, config_override = None):
        super().__init__(data, model, config_override)

        # get configs from config file
        self.config = yaml.safe_load(open("method/catalog/FACE/library/config.yml", 'r'))
        
        # merge configs with user specified, if they exist
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

        self._mode = self.config['mode']
        self._fraction = self.config['fraction']

        self._immutables = self._data.get_mutable_features(mutable=False)

    @property
    def fraction(self):
        """
        Controls the fraction of the used dataset to build the graph on.

        Returns
        ----------
        float
             The fraction of the used dataset to build the graph on.
        """
        return self._fraction
    
    @fraction.setter
    def fraction(self, value: float) -> None:
        """
        Sets the fraction of the used dataset to build the graph on.

        Parameters
        ----------
        value: float [0 < x < 1]
            The fraction of the used dataset to build the graph on.
        """
        if value <= 0 or value >= 1:
            raise ValueError("Fraction must be between 0 and 1.")
        self._fraction = value

    @property
    def mode(self):
        """
        Controls the type of FACE to be used.

        Returns
        ----------
        str
             The type of FACE to be used.
        """
        return self._mode
    
    @mode.setter
    def mode(self, mode: str) -> None:
        if mode in ["knn", "epsilon"]:
            self._mode = mode
        else:
            raise ValueError("Mode must be either 'knn' or 'epsilon'.")
        
    
    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:

        df = self._data.get_processed_data().copy().drop(columns=self._data.get_target_column()) # get a copy of the data without the target column

        # cond = df.isin(factuals).to_numpy()

        # df = df.drop(index=df[cond].index) # drop rows that are the same as the factuals
        merged = df.merge(factuals, how='outer', indicator=True)

        df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

        df = pd.concat([factuals, df], ignore_index=True) # add the factuals back to the data on top

        df = df[self._data.get_feature_names(expanded=True)] # reorder columns to match the original data

        factuals = factuals[self._data.get_feature_names(expanded=True)] # reorder columns to match the original data

        cfs = []
        for idx, factual in factuals.iterrows():
            cf = graph_search(
                df,
                idx,
                self._immutables,
                self._model,
                mode=self._mode,
                frac=self._fraction
            )
            cfs.append(cf)

        cfs = np.array(cfs)
        df_cfs = pd.DataFrame(cfs, columns=self._data.get_feature_names(expanded=True)) # ensure the feature ordering is correct for the model input
        df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index)

        return df_cfs
