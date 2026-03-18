import pandas as pd
import numpy as np

from typing import List, Optional, Union

from model.catalog.autoencoder.autoencoder import Autoencoder


def train_autoencoder(
    ae: Autoencoder,
    xtrain: Union[np.ndarray, pd.DataFrame],
    feature_order: Optional[List[str]] = None,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    save: bool = False,
    validation_split: float = 0.2,
    random_state: int = 42,
) -> Autoencoder:
    """
    Helper used to fit and optionally save an autoencoder.
    """
    if isinstance(xtrain, pd.DataFrame):
        if feature_order is not None:
            data = xtrain[feature_order].to_numpy(dtype=np.float32)
        else:
            data = xtrain.to_numpy(dtype=np.float32)
    else:
        data = np.asarray(xtrain, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D xtrain input, got shape {data.shape}")

    n = data.shape[0]
    if n < 2:
        train_data = data
        test_data = data
    else:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n)

        split_idx = int((1.0 - validation_split) * n)
        split_idx = min(max(split_idx, 1), n - 1)

        train_data = data[indices[:split_idx]]
        test_data = data[indices[split_idx:]]

    ae.fit(
        xtrain=train_data,
        xtest=test_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=1,
    )

    if save:
        ae.save()

    return ae