from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.data_object import DataObject
from model.model_object import ModelObject


class PyTorchCFVAENeuralNetwork(ModelObject, torch.nn.Module):
    def __init__(
        self,
        config_path: str = None,
        data_object: DataObject = None,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        ModelObject.__init__(self, config_path, data_object, config_override)
        torch.nn.Module.__init__(self)
        self._build_network()

    def _build_network(self) -> None:
        self.batch_size = self._config.get("batch_size", 32)
        self.epochs = self._config.get("epochs", 100)
        self.learning_rate = self._config.get("learning_rate", 0.001)
        self.optimizer = self._config.get("optimizer", "adam")
        self.loss_function = self._config.get("loss_function", "BCE")
        self.activation = self._config.get("output_activation", "sigmoid")
        self.hidden_dim = self._config.get("hidden_dim", 10)
        self.n_output = self._config.get("n_output", 2)
        self.device = self._device

        input_size = len(self._data_object.get_feature_names(expanded=True))
        self.predict_net = nn.Sequential(
            nn.Linear(input_size, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.n_output),
            nn.Sigmoid(),
        )

        self.to(self.device)

        if self._config.get("load_pretrained") and self._config.get("pretrained_path"):
            self.load_weights(self._config["pretrained_path"])
        else:
            self.fit(self._x_train, self._y_train)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_net(x)

    def _to_tensor(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]
    ) -> tuple[torch.Tensor, bool]:
        is_tensor = isinstance(x, torch.Tensor)
        if isinstance(x, pd.DataFrame):
            x_numeric = x[self.feature_order].to_numpy(dtype=np.float32)
            x_tensor = torch.tensor(x_numeric, dtype=torch.float32, device=self._device)
        elif is_tensor:
            x_tensor = x.to(self._device).float()
        else:
            x_numeric = np.array(x, dtype=np.float32)
            x_tensor = torch.tensor(x_numeric, dtype=torch.float32, device=self._device)
        return x_tensor, is_tensor

    def fit(self, x_train, y_train):
        self.train()
        x_train_tensor = torch.from_numpy(np.array(x_train).astype(np.float32)).to(
            self.device
        )
        y_train_tensor = torch.from_numpy(np.array(y_train)).long().to(self.device)

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        if self.optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "rms":
            optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")

        if self.loss_function != "BCE":
            raise ValueError("mlp_cfvae only supports BCE to match CFVAE reproduction")
        criterion = nn.BCELoss()

        for _ in range(self.epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_x)
                targets = F.one_hot(batch_y, num_classes=self.n_output).float()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        return self

    def load_weights(self, path: str) -> None:
        load_path = Path(path).expanduser()
        self.load_state_dict(torch.load(load_path, map_location=self.device))
        self.eval()

    def save_weights(self, path: str) -> None:
        save_path = Path(path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)

    def get_train_accuracy(self) -> float:
        x_train = self._x_train
        if isinstance(x_train, pd.DataFrame):
            x_train = x_train[self.feature_order].values
        predictions = self.predict(x_train)
        return float(np.mean(predictions == np.asarray(self._y_train)))

    def get_test_accuracy(self) -> float:
        x_test = self._x_test
        if isinstance(x_test, pd.DataFrame):
            x_test = x_test[self.feature_order].values
        predictions = self.predict(x_test)
        return float(np.mean(predictions == np.asarray(self._y_test)))

    def get_auc(self) -> float:
        x_test = self._x_test
        if isinstance(x_test, pd.DataFrame):
            x_test = x_test[self.feature_order].values
        y_proba = self.predict_proba(x_test)[:, 1]
        return float(roc_auc_score(self._y_test, y_proba))

    def predict(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        self.eval()
        x_tensor, is_tensor = self._to_tensor(x)
        with torch.no_grad():
            predictions = torch.argmax(self(x_tensor), dim=1)
        return predictions if is_tensor else predictions.cpu().numpy()

    def predict_both_classes(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        self.eval()
        x_tensor, is_tensor = self._to_tensor(x)
        with torch.no_grad():
            predicted_labels = torch.argmax(self(x_tensor), dim=1)
            one_hot = F.one_hot(predicted_labels, num_classes=self.n_output).float()
        return one_hot if is_tensor else one_hot.cpu().numpy()

    def predict_proba(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        self.eval()
        x_tensor, is_tensor = self._to_tensor(x)
        with torch.no_grad():
            predictions = self(x_tensor)
        return predictions if is_tensor else predictions.cpu().numpy()
