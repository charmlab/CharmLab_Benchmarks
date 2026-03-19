import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm
import yaml

from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from experiment_utils import deep_merge, reconstruct_encoding_constraints
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject

LOGGER = logging.getLogger(__name__)


def _set_seed(seed: int = 10_000_000) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_value) -> torch.device:
    if isinstance(device_value, torch.device):
        return device_value
    if device_value in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_value))


class _CFVAE(nn.Module):
    def __init__(self, feature_num: int, encoded_size: int):
        super().__init__()
        self.feature_num = feature_num
        self.encoded_size = encoded_size

        self.encoder_mean = nn.Sequential(
            nn.Linear(self.feature_num + 1, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size),
        )

        self.encoder_var = nn.Sequential(
            nn.Linear(self.feature_num + 1, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size),
            nn.Sigmoid(),
        )

        self.decoder_mean = nn.Sequential(
            nn.Linear(self.encoded_size + 1, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, self.feature_num),
            nn.Sigmoid(),
        )

    def encoder(self, x: torch.Tensor):
        x = x.float()
        mean = self.encoder_mean(x)
        var = 0.5 + self.encoder_var(x)
        return mean, var

    def sample_latent_code(self, mean: torch.Tensor, var: torch.Tensor):
        mean, var = mean.float(), var.float()
        eps = torch.randn_like(var)
        return mean + torch.sqrt(var) * eps

    def decoder(self, z: torch.Tensor):
        return self.decoder_mean(z.float())

    def forward(self, x: torch.Tensor, conditions: torch.Tensor, sample: bool = True):
        x, conditions = x.float(), conditions.view(-1, 1).float()
        mean, var = self.encoder(torch.cat((x, conditions), dim=1))
        z = self.sample_latent_code(mean, var) if sample else mean
        return self.decoder(torch.cat((z, conditions), dim=1))

    def forward_with_kl(
        self, x: torch.Tensor, conditions: torch.Tensor, sample: bool = True
    ):
        x, conditions = x.float(), conditions.view(-1, 1).float()
        mean, var = self.encoder(torch.cat((x, conditions), dim=1))
        kl_divergence = 0.5 * torch.mean(mean**2 + var - torch.log(var) - 1, dim=1)
        z = self.sample_latent_code(mean, var) if sample else mean
        x_pred = self.decoder(torch.cat((z, conditions), dim=1))
        return x_pred, kl_divergence


@register_method("CFVAE")
class CFVAE(MethodObject):
    def __init__(
        self,
        data: DataObject,
        model: ModelObject,
        config_override: Optional[Dict] = None,
    ):
        super().__init__(data, model, config_override)

        if not isinstance(self._model, nn.Module):
            raise ValueError("CFVAE requires a differentiable torch model")

        config_path = Path(__file__).resolve().parent / "library" / "config.yml"
        self.config = yaml.safe_load(config_path.read_text())
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

        self._feature_order = self._data.get_feature_names(expanded=True)
        self._seed = int(self.config.get("seed", 10_000_000))
        self._device = _resolve_device(self.config.get("device"))
        self._sample_during_generation = bool(self.config.get("sample", False))

        self._cf_model = _CFVAE(
            feature_num=len(self._feature_order),
            encoded_size=int(self.config["encoded_size"]),
        )
        self._trained = False

        load_path = self.config.get("load_path")
        if load_path:
            self.load(load_path)
        elif self.config.get("train", True):
            self.train()
        else:
            raise ValueError("CFVAE requires either `train: true` or a `load_path`.")

    def _categorical_index_groups(self, columns: List[str]) -> List[List[int]]:
        groups: List[List[int]] = []
        for feature_group in self._data.get_categorical_features(expanded=True):
            if not feature_group:
                continue
            indices = [
                columns.index(feature)
                for feature in feature_group
                if feature in columns
            ]
            if indices:
                groups.append(indices)
        return groups

    def _continuous_ranges(self) -> Dict[int, Tuple[float, float]]:
        normalise_weights: Dict[int, Tuple[float, float]] = {}
        metadata = self._data.get_metadata()
        raw_df = getattr(self._data, "_raw_df", None)

        for feature in self._data.get_continuous_features():
            if feature not in self._feature_order:
                continue

            min_val = max_val = None
            feature_metadata = metadata.get(feature, {})
            domain = feature_metadata.get("domain")
            if domain is not None and len(domain) == 2:
                min_val, max_val = float(domain[0]), float(domain[1])
            elif raw_df is not None and feature in raw_df.columns:
                min_val = float(raw_df[feature].min())
                max_val = float(raw_df[feature].max())
            else:
                column = self._data.get_processed_data()[feature]
                min_val = float(column.min())
                max_val = float(column.max())

            normalise_weights[self._feature_order.index(feature)] = (min_val, max_val)

        return normalise_weights

    def _model_output(self, x: torch.Tensor) -> torch.Tensor:
        output = self._model.forward(x)
        if not torch.is_tensor(output):
            output = torch.tensor(output, dtype=torch.float32, device=x.device)
        output = output.float()
        if output.ndim == 1:
            output = output.unsqueeze(1)
        if output.shape[1] == 1:
            output = torch.cat((1.0 - output, output), dim=1)
        if output.shape[1] != 2:
            raise ValueError(
                f"CFVAE expects 2-class model outputs, got shape {output.shape}"
            )
        return output

    def _save_model(
        self, save_path: str, margin: float, validity_reg: float, epoch: int
    ):
        save_target = Path(save_path).expanduser()
        if save_target.suffix != ".pth":
            save_target.mkdir(parents=True, exist_ok=True)
            save_target = save_target / (
                f"CFVAE-margin-{margin}-validity_reg-{validity_reg}-epoch-{epoch}-ModelBasedCF.pth"
            )
        else:
            save_target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._cf_model.state_dict(), save_target)
        LOGGER.info("Saved CFVAE weights to %s", save_target)

    def train(self):
        _set_seed(self._seed)

        train_df = self._model.get_train_data()[0][self._feature_order]
        train_dataset = torch.tensor(train_df.values, dtype=torch.float32)
        dataset_size = train_dataset.size(0)

        batch_size = int(self.config["batch_size"])
        epoch = int(self.config["epoch"])
        learning_rate = float(self.config["learning_rate"])
        n_samples = int(self.config["n_samples"])
        margin = float(self.config["margin"])
        validity_reg = float(self.config["validity_reg"])
        constraint_reg = float(self.config["constraint_reg"])
        preference_reg = float(self.config["preference_reg"])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        cat_groups = self._categorical_index_groups(self._feature_order)
        categorical_indices = [index for group in cat_groups for index in group]
        normalise_weights = self._continuous_ranges()

        self._cf_model.train().to(self._device)
        optimizer = optim.SGD(
            self._cf_model.parameters(),
            lr=learning_rate,
            weight_decay=1e-2,
        )

        for _ in tqdm(range(epoch), desc="cfvae_epochs"):
            with tqdm(total=dataset_size, desc="loss: N/A", leave=False) as pbar:
                for train_x in train_loader:
                    train_x = train_x.float().to(self._device)
                    optimizer.zero_grad()

                    with torch.no_grad():
                        model_output = self._model_output(train_x)
                        train_y = 1.0 - torch.argmax(model_output, dim=1).float()

                    reconstruction_loss = torch.zeros(
                        train_x.size(0), device=self._device
                    )
                    kl_loss = torch.zeros(train_x.size(0), device=self._device)
                    validity_loss = torch.zeros(1, device=self._device)
                    constraint_loss = torch.zeros(1, device=self._device)
                    preference_loss = torch.zeros(1, device=self._device)

                    for _sample in range(n_samples):
                        x_pred, kl_loss = self._cf_model.forward_with_kl(
                            train_x, train_y
                        )

                        reconstruction_increment = torch.zeros(
                            train_x.size(0),
                            device=self._device,
                        )

                        if categorical_indices:
                            reconstruction_increment += -torch.sum(
                                torch.abs(
                                    train_x[:, categorical_indices]
                                    - x_pred[:, categorical_indices]
                                ),
                                dim=1,
                            )

                        for key, (min_val, max_val) in normalise_weights.items():
                            range_val = max(max_val - min_val, 1.0)
                            reconstruction_increment += -range_val * torch.abs(
                                train_x[:, key] - x_pred[:, key]
                            )

                        for index_group in cat_groups:
                            reconstruction_increment += -torch.abs(
                                1.0 - torch.sum(x_pred[:, index_group], dim=1)
                            )

                        reconstruction_loss += reconstruction_increment

                        y_pred = self._model_output(x_pred)
                        y_pred_pos = y_pred[train_y == 1, :]
                        y_pred_neg = y_pred[train_y == 0, :]

                        if torch.sum(train_y == 1) > 0:
                            validity_loss += F.hinge_embedding_loss(
                                y_pred_pos[:, 1] - y_pred_pos[:, 0],
                                -torch.ones(y_pred_pos.shape[0], device=self._device),
                                margin=margin,
                                reduction="mean",
                            )
                        if torch.sum(train_y == 0) > 0:
                            validity_loss += F.hinge_embedding_loss(
                                y_pred_neg[:, 0] - y_pred_neg[:, 1],
                                -torch.ones(y_pred_neg.shape[0], device=self._device),
                                margin=margin,
                                reduction="mean",
                            )

                    reconstruction_loss = reconstruction_loss / n_samples
                    kl_loss = kl_loss / n_samples
                    validity_loss = -1 * validity_reg * validity_loss / n_samples
                    constraint_loss = constraint_reg * constraint_loss / n_samples
                    preference_loss = preference_reg * preference_loss / n_samples

                    loss = (
                        -torch.mean(reconstruction_loss - kl_loss)
                        - validity_loss
                        + constraint_loss
                        + preference_loss
                    )
                    loss.backward()
                    optimizer.step()

                    pbar.set_description(
                        "Reconstruction: "
                        + str(-torch.mean(reconstruction_loss).item())
                        + " KL: "
                        + str(torch.mean(kl_loss).item())
                        + " Validity: "
                        + str(torch.mean(-validity_loss).item())
                    )
                    pbar.update(len(train_x))

        save_path = self.config.get("save_path")
        if save_path:
            self._save_model(save_path, margin, validity_reg, epoch)

        self._trained = True

    def load(self, load_path: str):
        target = Path(load_path).expanduser()
        self._cf_model.load_state_dict(
            torch.load(target, map_location=torch.device("cpu"))
        )
        self._trained = True
        LOGGER.info("Loaded CFVAE weights from %s", target)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        _set_seed(self._seed)
        if not self._trained:
            raise AssertionError("Error: Run train() or load() first!")

        factuals = factuals[self._feature_order].copy()
        self._cf_model.eval().to(self._device)
        cat_groups = self._categorical_index_groups(self._feature_order)

        generated = []
        for _, row in factuals.iterrows():
            with torch.no_grad():
                test_x = torch.tensor(
                    row.to_numpy(dtype=np.float32),
                    dtype=torch.float32,
                    device=self._device,
                ).view(1, -1)
                test_y = 1.0 - torch.argmax(self._model_output(test_x), dim=1).float()
                x_pred = self._cf_model(
                    test_x,
                    test_y,
                    sample=self._sample_during_generation,
                )
                x_pred = reconstruct_encoding_constraints(x_pred, cat_groups)
                generated.append(x_pred.view(-1).cpu().numpy())

        df_cfs = pd.DataFrame(
            generated, columns=self._feature_order, index=factuals.index
        )
        df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index)
        return df_cfs

    @staticmethod
    def constraint_loss_func_example(train_x: torch.Tensor, x_pred: torch.Tensor):
        return F.hinge_embedding_loss(
            x_pred[:, 0] - train_x[:, 0],
            -torch.ones(train_x.shape[0], device=train_x.device),
            margin=0,
        )
