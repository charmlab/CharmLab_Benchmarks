import json
import os
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model.catalog.autoencoder.library.save_load import get_home


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2)


class Autoencoder(nn.Module):
    def __init__(
        self,
        data_name: str,
        layers: Optional[List[int]] = None,
        optimizer: str = "rmsprop",
        loss: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """
        PyTorch adaptation of the TensorFlow/Keras autoencoder scaffold.

        Parameters
        ----------
        data_name:
            Name of dataset. Used for save/load file names.
        layers:
            [input_dim, hidden_1, ..., hidden_n, latent_dim]
        optimizer:
            Optimizer name: "rmsprop", "adam", or "sgd".
        loss:
            Loss function. Defaults to MSE.
        """
        super().__init__()

        if layers is None or self.layers_valid(layers):
            self._layers = layers
        else:
            raise ValueError(
                "Number of layers have to be at least 2 (input and latent space), and number of neurons bigger than 0"
            )

        self.data_name = data_name
        self._optimizer_name = optimizer.lower()
        self._loss_fn = loss if loss is not None else mse_loss
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Underlying reconstruction network (encoder + decoder stack)
        self._network: Optional[nn.Sequential] = None
        if self._layers is not None:
            self._network = self._build_network(self._layers).to(self._device)

    @staticmethod
    def layers_valid(layers: List[int]) -> bool:
        if len(layers) < 2:
            return False
        return all(layer > 0 for layer in layers)

    def _build_network(self, layers: List[int]) -> nn.Sequential:
        # Encoder (mirrors provided TF code)
        encoder_layers: List[nn.Module] = [nn.Linear(layers[0], layers[1]), nn.ReLU()]
        for i in range(2, len(layers) - 1):
            encoder_layers.append(nn.Linear(layers[i - 1], layers[i]))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(layers[-2], layers[-1]))
        encoder_layers.append(nn.ReLU())

        # Decoder (keeps same dimension ordering as provided TF implementation)
        decoder_layers: List[nn.Module] = [nn.Linear(layers[-1], layers[1]), nn.ReLU()]
        for i in range(2, len(layers) - 1):
            decoder_layers.append(nn.Linear(layers[i - 1], layers[i]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(layers[-2], layers[0]))
        decoder_layers.append(nn.Sigmoid())

        return nn.Sequential(*(encoder_layers + decoder_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._network is None:
            raise RuntimeError("Autoencoder network is not initialized.")
        return self._network(x.to(self._device))

    def _build_optimizer(self, lr: float) -> torch.optim.Optimizer:
        if self._optimizer_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr)
        if self._optimizer_name == "sgd":
            return torch.optim.SGD(self.parameters(), lr=lr)
        # default to rmsprop to match original TF default
        return torch.optim.RMSprop(self.parameters(), lr=lr)

    @staticmethod
    def _to_numpy_2d(x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            arr = x.to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input data, got shape {arr.shape}")
        return arr

    def fit(
        self,
        xtrain: Union[np.ndarray, pd.DataFrame],
        xtest: Union[np.ndarray, pd.DataFrame],
        epochs: int,
        batch_size: int,
        lr: float = 1e-3,
        verbose: int = 1,
    ) -> "Autoencoder":
        """
        Train autoencoder to reconstruct input features.
        """
        if self._network is None:
            if self._layers is None:
                raise RuntimeError("Cannot train without layer definition.")
            self._network = self._build_network(self._layers).to(self._device)

        xtrain_np = self._to_numpy_2d(xtrain)
        xtest_np = self._to_numpy_2d(xtest)

        train_tensor = torch.tensor(xtrain_np, dtype=torch.float32, device=self._device)
        test_tensor = torch.tensor(xtest_np, dtype=torch.float32, device=self._device)

        train_loader = torch.utils.data.DataLoader(
            train_tensor, batch_size=batch_size, shuffle=True
        )

        optimizer = self._build_optimizer(lr)

        for epoch in range(epochs):
            self.train(mode=True)
            running_loss = 0.0

            for batch_x in train_loader:
                optimizer.zero_grad()
                reconstruction = self.forward(batch_x)
                loss = self._loss_fn(reconstruction, batch_x)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * batch_x.shape[0]

            train_loss = running_loss / max(1, xtrain_np.shape[0])

            self.train(mode=False)
            with torch.no_grad():
                val_recon = self.forward(test_tensor)
                val_loss = float(self._loss_fn(val_recon, test_tensor).item())

            if verbose:
                print(
                    f"AE Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}"
                )

        self.eval()
        return self

    def save(self, fitted_ae: Optional["Autoencoder"] = None) -> None:
        """
        Save weights and architecture metadata.
        """
        ae = fitted_ae if fitted_ae is not None else self
        if ae._layers is None:
            raise RuntimeError("Cannot save autoencoder without layer definition.")

        cache_path = get_home()
        prefix = os.path.join(cache_path, f"{ae.data_name}_{ae._layers[0]}")

        # Weights
        torch.save(ae.state_dict(), f"{prefix}.pt")

        # Architecture metadata (PyTorch analog of model JSON)
        payload = {
            "layers": ae._layers,
            "optimizer": ae._optimizer_name,
            "loss": getattr(ae._loss_fn, "__name__", "custom"),
        }
        with open(f"{prefix}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load(self, input_shape: int) -> "Autoencoder":
        """
        Load pretrained AE from cache.
        """
        cache_path = get_home()
        prefix = os.path.join(cache_path, f"{self.data_name}_{input_shape}")
        json_path = f"{prefix}.json"
        weights_path = f"{prefix}.pt"

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Autoencoder architecture not found: {json_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Autoencoder weights not found: {weights_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        layers = meta.get("layers")
        if layers is None or not self.layers_valid(layers):
            raise ValueError("Invalid or missing layer metadata in saved autoencoder.")

        self._layers = layers
        self._network = self._build_network(self._layers).to(self._device)

        state_dict = torch.load(weights_path, map_location=self._device)
        self.load_state_dict(state_dict)
        self.eval()

        return self
