import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm
import yaml

from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from experiment_utils import deep_merge
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject

from .cfrl_backend import set_seed
from .cfrl_tabular import CounterfactualRLTabular as CFRLExplainer
from .cfrl_tabular import get_conditional_dim, get_he_preprocessor

LOGGER = logging.getLogger(__name__)


def _coerce_mapping_key(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return int(round(float(value)))
        return float(value)
    return str(value)


class HeterogeneousEncoder(nn.Module):
    """PyTorch replica of the ADULT encoder from the Keras CFRL implementation."""

    def __init__(
        self, hidden_dim: int, latent_dim: int, input_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        use_lazy = hasattr(nn, "LazyLinear") and input_dim is None
        if input_dim is None and not use_lazy:
            raise ValueError(
                "input_dim must be provided when torch.nn.LazyLinear is unavailable."
            )
        if use_lazy:
            self.fc1 = nn.LazyLinear(hidden_dim)  # type: ignore[attr-defined]
            self.fc2 = nn.LazyLinear(latent_dim)  # type: ignore[attr-defined]
        else:
            assert input_dim is not None
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class HeterogeneousDecoder(nn.Module):
    """PyTorch replica of the ADULT decoder from the Keras CFRL implementation."""

    def __init__(
        self,
        hidden_dim: int,
        output_dims: List[int],
        latent_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        use_lazy = hasattr(nn, "LazyLinear") and latent_dim is None
        if latent_dim is None and not use_lazy:
            raise ValueError(
                "latent_dim must be provided when torch.nn.LazyLinear is unavailable."
            )
        if use_lazy:
            self.fc1 = nn.LazyLinear(hidden_dim)  # type: ignore[attr-defined]
            self.heads = nn.ModuleList(
                [nn.LazyLinear(dim) for dim in output_dims]  # type: ignore[attr-defined]
            )
        else:
            assert latent_dim is not None
            self.fc1 = nn.Linear(latent_dim, hidden_dim)
            self.heads = nn.ModuleList(
                [nn.Linear(hidden_dim, dim) for dim in output_dims]
            )

    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:  # noqa: D401
        h = F.relu(self.fc1(z))
        return [head(h) for head in self.heads]


@dataclass
class _FeatureMetadata:
    feature_names: List[str]
    long_to_short: Dict[str, str]
    short_to_long: Dict[str, str]
    attr_types: Dict[str, str]
    attr_bounds: Dict[str, List[float]]
    categorical_indices: List[int]
    numerical_indices: List[int]
    raw_to_idx: Dict[str, Dict[int, int]]
    idx_to_raw: Dict[str, Dict[int, int]]
    category_map: Dict[int, List[str]]
    feature_types: Dict[str, type]


@register_method("CFRL")
class CFRL(MethodObject):
    _DEFAULT_CONFIG: Dict[str, object] = {
        "seed": 0,
        "autoencoder_batch_size": 128,
        "autoencoder_target_steps": 100_000,
        "autoencoder_lr": 1e-3,
        "autoencoder_latent_dim": 15,
        "autoencoder_hidden_dim": 128,
        "coeff_sparsity": 0.5,
        "coeff_consistency": 0.5,
        "train_steps": 100_000,
        "batch_size": 128,
        "immutable_features": "_optional_",
        "constrained_ranges": "_optional_",
        "save_path": "_optional_",
        "save_prefix": "_optional_",
        "train": True,
    }
    _OPTIONAL_KEYS = {
        "immutable_features",
        "constrained_ranges",
        "save_path",
        "save_prefix",
    }

    def __init__(
        self,
        data: DataObject,
        model: ModelObject,
        config_override: Optional[Dict] = None,
    ) -> None:
        super().__init__(data, model, config_override)

        config_path = Path(__file__).resolve().parent / "library" / "config.yml"
        self.config = yaml.safe_load(config_path.read_text())
        if self.config is None:
            self.config = {}
        self.config = deep_merge(self._DEFAULT_CONFIG, self.config)
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)
        for key in self._OPTIONAL_KEYS:
            if self.config.get(key) == "_optional_":
                self.config[key] = None

        if not isinstance(self._model, nn.Module):
            raise ValueError("CFRL requires a differentiable torch-backed model.")
        if self._data._config.get("preprocessing_strategy") != "normalize":
            raise ValueError(
                "CFRL currently expects `normalize` preprocessing to reconstruct raw values correctly."
            )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._feature_order = self._data.get_feature_names(expanded=True)
        self._raw_feature_names = self._data.get_feature_names(expanded=False)

        set_seed(int(self.config["seed"]))

        self._metadata = self._prepare_metadata()
        self._feature_count = len(self._metadata.feature_names)
        self._raw_feature_set = set(self._metadata.feature_names)
        self._encoded_cat_columns: Dict[int, List[str]] = {}
        self._categorical_sizes: Dict[int, int] = {}

        data_metadata = self._data.get_metadata()
        for idx, long_name in enumerate(self._metadata.feature_names):
            attr_type = self._metadata.attr_types[long_name]
            if attr_type in {"numeric-int", "numeric-real", "binary"}:
                continue
            categories = self._metadata.category_map.get(idx, [])
            self._categorical_sizes[idx] = len(categories)
            self._encoded_cat_columns[idx] = list(
                data_metadata[long_name]["encoded_feature_names"]
            )

        self._encoder: Optional[HeterogeneousEncoder] = None
        self._decoder: Optional[HeterogeneousDecoder] = None
        self._encoder_preprocessor = None
        self._decoder_inv_preprocessor = None
        self._cf_model: Optional[CFRLExplainer] = None
        self._trained = False

        if self.config["train"]:
            self.train()

    def _prepare_metadata(self) -> _FeatureMetadata:
        raw_df = getattr(self._data, "_raw_df", None)
        if raw_df is None:
            raise ValueError("CFRL requires access to the raw dataframe.")

        feature_names = list(self._raw_feature_names)
        long_to_short = {name: name for name in feature_names}
        short_to_long = {name: name for name in feature_names}

        attr_bounds: Dict[str, List[float]] = {}
        categorical_indices: List[int] = []
        numerical_indices: List[int] = []
        raw_to_idx: Dict[str, Dict[int, int]] = {}
        idx_to_raw: Dict[str, Dict[int, int]] = {}
        category_map: Dict[int, List[str]] = {}
        feature_types: Dict[str, type] = {}
        attr_types: Dict[str, str] = {}

        metadata = self._data.get_metadata()
        for idx, name in enumerate(feature_names):
            feature_meta = metadata[name]
            feature_type = feature_meta["type"]

            domain = feature_meta.get("domain")
            if domain is not None and len(domain) == 2:
                lower_bound, upper_bound = float(domain[0]), float(domain[1])
            else:
                lower_bound = float(raw_df[name].min())
                upper_bound = float(raw_df[name].max())
            attr_bounds[name] = [lower_bound, upper_bound]

            if feature_type == "categorical":
                attr_types[name] = "categorical"
                categorical_indices.append(idx)
                unique_vals = sorted(raw_df[name].dropna().unique().tolist())
                mapped_unique = [_coerce_mapping_key(v) for v in unique_vals]
                raw_to_idx[name] = {val: i for i, val in enumerate(mapped_unique)}
                idx_to_raw[name] = {i: val for i, val in enumerate(mapped_unique)}
                category_map[idx] = [str(val) for val in mapped_unique]
                feature_types[name] = str
            elif feature_type == "binary":
                attr_types[name] = "binary"
                numerical_indices.append(idx)
                feature_types[name] = int
            else:
                is_integer = pd.api.types.is_integer_dtype(raw_df[name])
                attr_types[name] = "numeric-int" if is_integer else "numeric-real"
                numerical_indices.append(idx)
                feature_types[name] = int if is_integer else float

        return _FeatureMetadata(
            feature_names=feature_names,
            long_to_short=long_to_short,
            short_to_long=short_to_long,
            attr_types=attr_types,
            attr_bounds=attr_bounds,
            categorical_indices=categorical_indices,
            numerical_indices=numerical_indices,
            raw_to_idx=raw_to_idx,
            idx_to_raw=idx_to_raw,
            category_map=category_map,
            feature_types=feature_types,
        )

    def _default_immutable_features(self) -> List[str]:
        immutable: List[str] = []
        for feature, feature_meta in self._data.get_metadata().items():
            if (
                feature_meta.get("node_type") == "input"
                and feature_meta.get("mutability") is False
            ):
                immutable.append(feature)
        return immutable

    def _default_constrained_ranges(self) -> Dict[str, List[float]]:
        constrained_ranges: Dict[str, List[float]] = {}
        for feature, feature_meta in self._data.get_metadata().items():
            if feature_meta.get("node_type") != "input":
                continue
            if feature_meta.get("type") != "numerical":
                continue

            actionability = feature_meta.get("actionability", "any")
            if actionability == "same-or-increase":
                constrained_ranges[feature] = [0.0, 1.0]
            elif actionability == "same-or-decrease":
                constrained_ranges[feature] = [-1.0, 0.0]
            elif actionability == "none":
                constrained_ranges[feature] = [0.0, 0.0]
            else:
                constrained_ranges[feature] = [-1.0, 1.0]

        return constrained_ranges

    def _ordered_to_cfrl(self, frame: pd.DataFrame) -> np.ndarray:
        if self._raw_feature_set.issubset(frame.columns):
            raw_df = frame[self._metadata.feature_names]
            arr = np.zeros((raw_df.shape[0], self._feature_count), dtype=np.float32)

            for idx, name in enumerate(self._metadata.feature_names):
                col = raw_df[name].to_numpy()
                if idx in self._metadata.categorical_indices:
                    mapping = self._metadata.raw_to_idx[name]
                    arr[:, idx] = np.array(
                        [mapping.get(_coerce_mapping_key(v), 0) for v in col],
                        dtype=np.float32,
                    )
                else:
                    arr[:, idx] = col.astype(np.float32, copy=False)
            return arr

        ordered = frame.reindex(columns=self._feature_order, fill_value=0.0)
        arr = np.zeros((ordered.shape[0], self._feature_count), dtype=np.float32)

        for idx, long_name in enumerate(self._metadata.feature_names):
            attr_type = self._metadata.attr_types[long_name]

            if attr_type in {"numeric-int", "numeric-real", "binary"}:
                values = ordered[long_name].to_numpy(dtype=np.float32, copy=False)
                lower, upper = self._metadata.attr_bounds[long_name]
                raw_vals = values * (upper - lower) + lower
                if attr_type in {"numeric-int", "binary"}:
                    raw_vals = np.rint(raw_vals)
                arr[:, idx] = raw_vals.astype(np.float32, copy=False)
                continue

            columns = self._encoded_cat_columns[idx]
            block = ordered[columns].to_numpy(dtype=np.float32, copy=False)
            idxs = np.argmax(block, axis=1)
            arr[:, idx] = np.clip(idxs, 0, self._categorical_sizes[idx] - 1)

        return arr

    def _cfrl_to_ordered(self, arr_zero: np.ndarray) -> pd.DataFrame:
        arr = np.atleast_2d(arr_zero).astype(np.float32, copy=False)
        num_rows = arr.shape[0]
        blocks: List[np.ndarray] = []
        columns: List[str] = []

        for idx, long_name in enumerate(self._metadata.feature_names):
            attr_type = self._metadata.attr_types[long_name]

            if attr_type in {"numeric-int", "numeric-real", "binary"}:
                lower, upper = self._metadata.attr_bounds[long_name]
                diff = upper - lower
                if np.isclose(diff, 0.0):
                    norm_vals = np.zeros(num_rows, dtype=np.float32)
                else:
                    norm_vals = (arr[:, idx] - lower) / diff
                norm_vals = norm_vals.clip(0.0, 1.0)
                blocks.append(norm_vals.reshape(-1, 1))
                columns.append(long_name)
                continue

            n_categories = self._categorical_sizes.get(idx, 0)
            if n_categories == 0:
                continue

            idxs = np.clip(
                np.rint(arr[:, idx]).astype(int, copy=False), 0, n_categories - 1
            )
            block = np.zeros((num_rows, n_categories), dtype=np.float32)
            block[np.arange(num_rows), idxs] = 1.0
            blocks.append(block)
            columns.extend(self._encoded_cat_columns[idx])

        if blocks:
            data = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
            model_df = pd.DataFrame(data, columns=columns)
        else:
            model_df = pd.DataFrame(
                np.zeros((num_rows, len(self._feature_order)), dtype=np.float32),
                columns=self._feature_order,
            )

        model_df = model_df.reindex(columns=self._feature_order, fill_value=0.0)
        return model_df.astype(np.float32)

    def _build_predictor(self):
        def predictor(x: np.ndarray) -> np.ndarray:
            array = np.atleast_2d(x)
            model_input = self._cfrl_to_ordered(array)
            preds = self._model.predict_proba(model_input)
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().numpy()
            elif isinstance(preds, pd.DataFrame):
                preds = preds.to_numpy()
            elif isinstance(preds, np.ndarray):
                pass
            elif hasattr(preds, "numpy"):
                preds = preds.numpy()
            else:
                preds = np.asarray(preds)
            return preds

        return predictor

    def _train_autoencoder(self, X_pre: np.ndarray, X_zero: np.ndarray) -> None:
        target_steps = int(self.config["autoencoder_target_steps"])
        batch_size = int(self.config["autoencoder_batch_size"])
        lr = float(self.config["autoencoder_lr"])

        inputs = torch.tensor(X_pre, dtype=torch.float32, device=self._device)
        num_dim = len(self._metadata.numerical_indices)
        num_targets = inputs[:, :num_dim] if num_dim > 0 else None
        cat_targets = [
            torch.tensor(
                X_zero[:, idx].astype(np.int64),
                dtype=torch.long,
                device=self._device,
            )
            for idx in self._metadata.categorical_indices
        ]

        params = list(self._encoder.parameters()) + list(self._decoder.parameters())
        optimiser = optim.Adam(params, lr=lr)

        num_samples = inputs.size(0)
        if target_steps <= 0:
            raise ValueError("autoencoder_target_steps must be a positive integer.")
        steps_per_epoch = max(1, math.ceil(num_samples / batch_size))
        max_epochs = math.ceil(target_steps / steps_per_epoch)
        steps_run = 0

        with tqdm(total=target_steps, desc="AE steps", leave=False) as pbar:
            for epoch in range(max_epochs):
                perm = torch.randperm(num_samples, device=self._device)
                epoch_loss = 0.0
                steps_this_epoch = 0

                for start in range(0, num_samples, batch_size):
                    idx = perm[start : start + batch_size]
                    batch_x = inputs[idx]
                    outputs = self._decoder(self._encoder(batch_x))

                    loss = torch.zeros((), device=self._device)
                    output_offset = 0
                    if num_dim > 0 and num_targets is not None:
                        recon_num = outputs[0]
                        target_num = num_targets[idx]
                        loss = loss + F.mse_loss(recon_num, target_num)
                        output_offset = 1

                    cat_outputs = outputs[output_offset:]
                    cat_batches = [target[idx] for target in cat_targets]
                    if cat_outputs and cat_batches:
                        weight = 1.0 / len(cat_outputs)
                        for logits, target in zip(cat_outputs, cat_batches):
                            loss = loss + weight * F.cross_entropy(logits, target)

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                    epoch_loss += loss.item()
                    steps_run += 1
                    steps_this_epoch += 1
                    pbar.update(1)

                    if steps_run >= target_steps:
                        break

                if (
                    max_epochs <= 10
                    or (epoch + 1) % max(1, max_epochs // 5) == 0
                    or steps_run >= target_steps
                ):
                    LOGGER.info(
                        "CFRL autoencoder epoch %s/%s (~%s/%s steps) | loss %.4f",
                        epoch + 1,
                        max_epochs,
                        steps_run,
                        target_steps,
                        epoch_loss / max(1, steps_this_epoch),
                    )

                if steps_run >= target_steps:
                    break

        self._encoder.eval()
        self._decoder.eval()

    def _save_artifacts(self) -> None:
        save_path = self.config.get("save_path")
        if not save_path:
            return

        save_dir = Path(str(save_path)).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)

        data_name = self._data._config.get("name", "dataset")
        model_name = self._model._config.get("architecture", "model")
        prefix = self.config.get("save_prefix") or f"{data_name}_{model_name}"

        torch.save(self._encoder.state_dict(), save_dir / f"{prefix}_encoder.pt")
        torch.save(self._decoder.state_dict(), save_dir / f"{prefix}_decoder.pt")
        torch.save(
            self._cf_model.params["actor"].state_dict(),
            save_dir / f"{prefix}_actor.pt",
        )
        torch.save(
            self._cf_model.params["critic"].state_dict(),
            save_dir / f"{prefix}_critic.pt",
        )
        LOGGER.info("Saved CFRL artifacts to %s", save_dir)

    def train(self) -> None:
        set_seed(int(self.config["seed"]))

        df_train = self._model.get_train_data()[0][self._feature_order].copy()
        X_zero = self._ordered_to_cfrl(df_train).astype(np.float32)
        (
            self._encoder_preprocessor,
            self._decoder_inv_preprocessor,
        ) = get_he_preprocessor(
            X=X_zero,
            feature_names=self._metadata.feature_names,
            category_map=self._metadata.category_map,
            feature_types=self._metadata.feature_types,
        )

        X_pre = self._encoder_preprocessor(X_zero).astype(np.float32)
        latent_dim = int(self.config["autoencoder_latent_dim"])
        hidden_dim = int(self.config["autoencoder_hidden_dim"])

        self._encoder = HeterogeneousEncoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            input_dim=X_pre.shape[1],
        ).to(self._device)

        num_dim = len(self._metadata.numerical_indices)
        output_dims: List[int] = []
        if num_dim > 0:
            output_dims.append(num_dim)
        output_dims += [
            len(self._metadata.category_map[idx])
            for idx in self._metadata.categorical_indices
        ]
        self._decoder = HeterogeneousDecoder(
            hidden_dim=hidden_dim,
            output_dims=output_dims,
            latent_dim=latent_dim,
        ).to(self._device)

        LOGGER.info("Training CFRL heterogeneous autoencoder.")
        self._train_autoencoder(X_pre, X_zero)

        predictor = self._build_predictor()
        immutable_features = self.config.get("immutable_features")
        if immutable_features is None:
            immutable_features = self._default_immutable_features()

        ranges_long = (
            self.config.get("constrained_ranges") or self._default_constrained_ranges()
        )

        preds_shape = predictor(X_zero[:1]).shape
        num_classes = preds_shape[1] if len(preds_shape) == 2 else 1
        cond_dim = get_conditional_dim(
            self._metadata.feature_names,
            self._metadata.category_map,
        )
        actor_input_dim = latent_dim + 2 * num_classes + cond_dim

        LOGGER.info(
            "Training CFRL explainer (train_steps=%s).",
            self.config["train_steps"],
        )
        self._cf_model = CFRLExplainer(
            predictor=predictor,
            encoder=self._encoder,
            decoder=self._decoder,
            latent_dim=latent_dim,
            encoder_preprocessor=self._encoder_preprocessor,
            decoder_inv_preprocessor=self._decoder_inv_preprocessor,
            coeff_sparsity=float(self.config["coeff_sparsity"]),
            coeff_consistency=float(self.config["coeff_consistency"]),
            feature_names=self._metadata.feature_names,
            category_map=self._metadata.category_map,
            immutable_features=immutable_features,
            ranges=ranges_long,
            train_steps=int(self.config["train_steps"]),
            batch_size=int(self.config["batch_size"]),
            seed=int(self.config["seed"]),
            actor_input_dim=actor_input_dim,
        )
        self._cf_model.fit(X_zero.astype(np.float32))
        self._trained = True
        self._save_artifacts()

    def _generate_counterfactual(self, factual_row: pd.DataFrame) -> pd.DataFrame:
        factual_ordered = factual_row.reindex(
            columns=self._feature_order, fill_value=0.0
        )
        zero_input = self._ordered_to_cfrl(factual_ordered)

        preds = self._model.predict_proba(factual_ordered)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        elif not isinstance(preds, np.ndarray):
            preds = np.asarray(preds)

        target_class = 1 - int(np.argmax(preds, axis=1)[0])
        explanation = self._cf_model.explain(
            X=zero_input.astype(np.float32),
            Y_t=np.array([target_class]),
            C=[],
        )

        cf_data = explanation.get("cf", {}).get("X")
        if cf_data is None:
            LOGGER.warning(
                "CFRL failed to produce a counterfactual; falling back to the factual instance."
            )
            return factual_ordered

        cf_array = np.asarray(cf_data)
        if cf_array.ndim == 3:
            cf_array = cf_array[:, 0, :]
        cf_array = np.atleast_2d(cf_array)

        cf_ordered = self._cfrl_to_ordered(cf_array)
        cf_ordered.index = factual_ordered.index
        return cf_ordered

    def get_counterfactuals(self, factuals: pd.DataFrame):
        assert self._trained, "Error: run train() first."
        set_seed(int(self.config["seed"]))

        factuals = factuals[self._feature_order].copy()
        results: List[pd.DataFrame] = []
        for index, row in factuals.iterrows():
            cf_row = self._generate_counterfactual(pd.DataFrame([row]))
            cf_row.index = [index]
            results.append(cf_row)

        counterfactuals = pd.concat(results, axis=0).astype(np.float32)
        counterfactuals = check_counterfactuals(
            self._model,
            self._data,
            counterfactuals,
            factuals.index,
        )
        return counterfactuals.reindex(
            columns=self._feature_order + [self._data.get_target_column()]
        )
