from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data" / "catalog" / "adult_cfvae"
BIN_DIR = REPO_ROOT / "method" / "catalog" / "CFVAE" / "bin"


class DataLoader:
    """A small data interface copied from the original CFVAE reproduce workflow."""

    def __init__(self, params: Dict):
        if not isinstance(params["dataframe"], pd.DataFrame):
            raise ValueError("Expected `dataframe` to be a pandas DataFrame")
        if not isinstance(params["continuous_features"], list):
            raise ValueError("Expected `continuous_features` to be a list")
        if not isinstance(params["outcome_name"], str):
            raise ValueError("Expected `outcome_name` to be a string")

        self.data_df = params["dataframe"].copy()
        self.continuous_feature_names = params["continuous_features"]
        self.outcome_name = params["outcome_name"]

        self.categorical_feature_names = [
            name
            for name in self.data_df.columns.tolist()
            if name not in self.continuous_feature_names + [self.outcome_name]
        ]
        self.feature_names = [
            name for name in self.data_df.columns.tolist() if name != self.outcome_name
        ]

        if self.categorical_feature_names:
            self.data_df[self.categorical_feature_names] = self.data_df[
                self.categorical_feature_names
            ].astype("category")
            self.one_hot_encoded_data = self.one_hot_encode_data(self.data_df)
            self.encoded_feature_names = [
                col
                for col in self.one_hot_encoded_data.columns.tolist()
                if col != self.outcome_name
            ]
        else:
            self.one_hot_encoded_data = self.data_df
            self.encoded_feature_names = self.feature_names

        self.test_size = params.get("test_size", 0.2)
        self.test_split_random_state = params.get("test_split_random_state", 17)
        self.train_df, self.test_df = self.split_data(self.data_df)
        self.permitted_range = params.get("permitted_range", self.get_features_range())

    def get_features_range(self) -> Dict[str, List[float]]:
        ranges = {}
        for feature_name in self.continuous_feature_names:
            ranges[feature_name] = [
                float(self.data_df[feature_name].min()),
                float(self.data_df[feature_name].max()),
            ]
        return ranges

    def one_hot_encode_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(
            data,
            drop_first=False,
            columns=self.categorical_feature_names,
        )

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for feature_name in self.continuous_feature_names:
            max_value = self.data_df[feature_name].max()
            min_value = self.data_df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (
                max_value - min_value
            )
        return result

    def de_normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for feature_name in self.continuous_feature_names:
            max_value = self.data_df[feature_name].max()
            min_value = self.data_df[feature_name].min()
            result[feature_name] = (
                df[feature_name] * (max_value - min_value)
            ) + min_value
        return result

    def split_data(self, data: pd.DataFrame):
        return train_test_split(
            data,
            test_size=self.test_size,
            random_state=self.test_split_random_state,
        )

    def from_dummies(self, data: pd.DataFrame, prefix_sep: str = "_") -> pd.DataFrame:
        out = data.copy()
        for column in self.categorical_feature_names:
            cols = [c for c in data.columns if column + prefix_sep in c]
            labs = [c.replace(column + prefix_sep, "") for c in cols]
            out[column] = pd.Categorical(
                np.array(labs)[np.argmax(data[cols].values, axis=1)]
            )
            out.drop(cols, axis=1, inplace=True)
        return out

    def get_decoded_data(self, data):
        if isinstance(data, np.ndarray):
            index = [i for i in range(0, len(data))]
            data = pd.DataFrame(
                data=data,
                index=index,
                columns=self.encoded_feature_names,
            )
        return self.from_dummies(data)


def load_adult_income_dataset() -> pd.DataFrame:
    dataset_path = DATA_DIR / "adult_cfvae.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Adult CFVAE dataset not found at {dataset_path}")
    return pd.read_csv(dataset_path)


def load_pretrained_binaries(filename: str) -> str:
    if filename.endswith(".npy"):
        path = DATA_DIR / filename
    else:
        path = BIN_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Required CFVAE artifact not found: {path}")

    return str(path)
