from typing import Any, Dict, Optional, Union
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from experiment_utils import deep_merge, reconstruct_encoding_constraints
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject


ScalarLike = Union[float, int, torch.Tensor]


def _filter_hinge_loss(
    n_class: int,
    mask_vector: Union[np.ndarray, torch.Tensor],
    features: torch.Tensor,
    sigma: ScalarLike,
    temperature: ScalarLike,
    model_fn,
) -> torch.Tensor:
    n_input = int(features.shape[0])
    mask = torch.as_tensor(mask_vector, dtype=torch.bool, device=features.device).flatten()

    if mask.shape[0] != n_input:
        raise ValueError(
            f"mask_vector length {mask.shape[0]} does not match batch size {n_input}"
        )

    if not bool(torch.any(mask)):
        return torch.zeros((n_input, n_class), dtype=features.dtype, device=features.device)

    filtered_input = features[mask]

    sigma_local = sigma
    if isinstance(sigma, torch.Tensor) and sigma.ndim != 0:
        sigma_local = sigma[mask]

    temperature_local = temperature
    if isinstance(temperature, torch.Tensor) and temperature.ndim != 0:
        temperature_local = temperature[mask]

    filtered_loss = model_fn(filtered_input, sigma_local, temperature_local)

    hinge_loss = torch.zeros(
        (n_input, n_class),
        dtype=filtered_loss.dtype,
        device=filtered_loss.device,
    )
    hinge_loss[mask] = filtered_loss
    return hinge_loss


def distance_func(name: str, x1: torch.Tensor, x2: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    name = name.lower()
    if name == "l1":
        return l1_dist(x1, x2, ax=1, eps=eps)
    if name == "l2":
        return l2_dist(x1, x2, ax=1, eps=eps)
    if name == "cosine":
        return cosine_dist(x1, x2, ax=-1, eps=eps)
    raise ValueError(f"Unsupported distance function '{name}'. Use one of: l1, l2, cosine.")


def l1_dist(x1: torch.Tensor, x2: torch.Tensor, ax: int, eps: float = 0.0) -> torch.Tensor:
    x = x1 - x2
    return torch.sum(torch.abs(x), dim=ax) + float(eps)


def l2_dist(x1: torch.Tensor, x2: torch.Tensor, ax: int, eps: float = 0.0) -> torch.Tensor:
    x = x1 - x2
    return torch.sqrt(torch.sum(torch.square(x), dim=ax) + float(eps))


def cosine_dist(x1: torch.Tensor, x2: torch.Tensor, ax: int, eps: float = 0.0) -> torch.Tensor:
    normalize_x1 = F.normalize(x1, p=2, dim=1, eps=1e-12)
    normalize_x2 = F.normalize(x2, p=2, dim=1, eps=1e-12)
    cosine_similarity = torch.sum(normalize_x1 * normalize_x2, dim=ax)
    return (1.0 - cosine_similarity) + float(eps)


@register_method("FOCUS")
class Focus(MethodObject):
    """
    Implementation of Focus [1]_.
    
    This is a re-implementation in PyTorch of the original method implemented using tensoflow in the CARLA library.
    
    .. [1] Lucic, A., Oosterhuis, H., Haned, H., & de Rijke, M. (2018). FOCUS: Flexible optimizable counterfactual
            explanations for tree ensembles. arXiv preprint arXiv:1910.12199.
    """

    def __init__(
        self,
        data: DataObject,
        model: ModelObject,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(data, model, config_override=config_override)

        self._feature_order = self._data.get_feature_names(expanded=True)
        self._device = (
            self._model._device
            if hasattr(self._model, "_device")
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        backend = (
            self._model._config.get("backend", "").lower()
            if hasattr(self._model, "_config")
            else ""
        )
        if backend and backend != "pytorch":
            raise ValueError(
                f"FOCUS currently supports differentiable PyTorch models only, got: {backend}"
            )
        if not isinstance(self._model, torch.nn.Module):
            raise TypeError("FOCUS requires model to be a torch.nn.Module for gradient optimization")

        # get configs from config file
        self.config = yaml.safe_load(open("method/catalog/FOCUS/library/config.yml", 'r'))
        
        # merge configs with user specified, if they exist
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

        self._optimizer_name = str(self.config["optimizer"]).lower()
        self._lr = float(self.config["lr"])
        self.n_class = int(self.config["n_class"])
        self.n_iter = int(self.config["n_iter"])
        self.sigma_val = float(self.config["sigma"])
        self.temp_val = float(self.config["temperature"])
        self.distance_weight_val = float(self.config["distance_weight"])
        self.distance_function = str(self.config["distance_func"]).lower()
        self._clamp = bool(self.config["clamp"])
        self._enforce_encoding = bool(self.config["enforce_encoding"])
        self._enforce_mutability = bool(self.config["enforce_mutability"])

        if self.n_class != 2:
            raise ValueError("Current FOCUS implementation supports binary classification only")
        if self._optimizer_name not in {"adam", "gd", "sgd"}:
            raise ValueError("optimizer must be one of: adam, gd, sgd")

        self._mutable_mask = self._safe_get_mutable_mask()
        self._cat_feature_indices = self._build_cat_feature_indices()
        self._clip_min, self._clip_max = self._build_clip_bounds()

        self._model.to(self._device)
        self._model.eval()

    def _safe_get_mutable_mask(self) -> np.ndarray:
        try:
            mutable_mask = np.asarray(self._model.get_mutable_mask(), dtype=bool)
            if mutable_mask.shape[0] != len(self._feature_order):
                raise ValueError("Mutable mask shape does not match feature count")
            return mutable_mask
        except Exception:
            return np.ones(len(self._feature_order), dtype=bool)

    def _build_cat_feature_indices(self):
        cat_groups = self._data.get_categorical_features(expanded=True)
        cat_feature_indices = []
        for group in cat_groups:
            if not group:
                continue
            indices = [self._feature_order.index(feat) for feat in group if feat in self._feature_order]
            if indices:
                cat_feature_indices.append(indices)
        return cat_feature_indices

    def _build_clip_bounds(self):
        x_train, _ = self._model.get_train_data()
        if isinstance(x_train, pd.DataFrame):
            x_train_np = x_train[self._feature_order].to_numpy(dtype=np.float32)
        else:
            x_train_np = np.asarray(x_train, dtype=np.float32)

        clip_min_cfg = self.config.get("clip_min")
        clip_max_cfg = self.config.get("clip_max")

        if clip_min_cfg is None:
            clip_min = np.min(x_train_np, axis=0).astype(np.float32)
        else:
            clip_min = np.asarray(clip_min_cfg, dtype=np.float32)
            if clip_min.ndim == 0:
                clip_min = np.full(len(self._feature_order), float(clip_min), dtype=np.float32)

        if clip_max_cfg is None:
            clip_max = np.max(x_train_np, axis=0).astype(np.float32)
        else:
            clip_max = np.asarray(clip_max_cfg, dtype=np.float32)
            if clip_max.ndim == 0:
                clip_max = np.full(len(self._feature_order), float(clip_max), dtype=np.float32)

        if clip_min.shape[0] != len(self._feature_order) or clip_max.shape[0] != len(self._feature_order):
            raise ValueError("clip_min/clip_max must be scalar or have one value per feature")

        clip_min_t = torch.tensor(clip_min, dtype=torch.float32, device=self._device).unsqueeze(0)
        clip_max_t = torch.tensor(clip_max, dtype=torch.float32, device=self._device).unsqueeze(0)
        return clip_min_t, clip_max_t

    def _build_optimizer(self, params):
        if self._optimizer_name == "adam":
            return torch.optim.Adam(params, lr=self._lr)
        return torch.optim.SGD(params, lr=self._lr)

    def _to_two_class_probabilities(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 1:
            model_output = model_output.unsqueeze(1)

        if model_output.shape[1] == 1:
            pos = torch.clamp(model_output[:, 0], min=1e-7, max=1.0 - 1e-7)
            probs = torch.stack((1.0 - pos, pos), dim=1)
        else:
            probs = torch.clamp(model_output[:, :2], min=1e-7)
            probs = probs / torch.sum(probs, dim=1, keepdim=True)

        return probs

    def _prob_from_input(
        self,
        perturbed: torch.Tensor,
        sigma: ScalarLike,
        temperature: ScalarLike,
    ) -> torch.Tensor:
        model_output = self._model(perturbed)
        probs = self._to_two_class_probabilities(model_output)

        if isinstance(sigma, torch.Tensor):
            sigma_t = sigma.reshape(-1, 1).to(device=probs.device, dtype=probs.dtype)
        else:
            sigma_t = torch.full((probs.shape[0], 1), float(sigma), device=probs.device, dtype=probs.dtype)

        if isinstance(temperature, torch.Tensor):
            temp_t = temperature.reshape(-1, 1).to(device=probs.device, dtype=probs.dtype)
        else:
            temp_t = torch.full(
                (probs.shape[0], 1),
                float(temperature),
                device=probs.device,
                dtype=probs.dtype,
            )

        sigma_t = torch.clamp(sigma_t, min=1e-6)
        temp_t = torch.clamp(temp_t, min=1e-6)

        logits = torch.log(torch.clamp(probs, min=1e-8))
        scaled_logits = (sigma_t * logits) / temp_t
        return torch.softmax(scaled_logits, dim=1)

    def get_counterfactuals(self, factuals: pd.DataFrame):
        factuals = factuals[self._feature_order]

        if factuals.shape[0] == 0:
            return factuals.copy()

        original_input = factuals.to_numpy(dtype=np.float32)
        n_input = original_input.shape[0]

        gt_proba = np.asarray(self._model.predict_proba(original_input), dtype=np.float32)
        if gt_proba.ndim == 1:
            gt_proba = np.vstack((1.0 - gt_proba, gt_proba)).T
        elif gt_proba.shape[1] == 1:
            pos = gt_proba[:, 0]
            gt_proba = np.vstack((1.0 - pos, pos)).T

        ground_truth = np.argmax(gt_proba, axis=1).astype(np.int64)

        original_tensor = torch.tensor(original_input, dtype=torch.float32, device=self._device)
        perturbed = torch.tensor(
            original_input, dtype=torch.float32, device=self._device, requires_grad=True
        )

        optimizer = self._build_optimizer([perturbed])

        class_index = torch.tensor(ground_truth, dtype=torch.long, device=self._device)
        row_index = torch.arange(n_input, device=self._device)

        indicator = np.ones(n_input, dtype=np.float32)
        sigma = np.full(n_input, self.sigma_val, dtype=np.float32)
        temperature = np.full(n_input, self.temp_val, dtype=np.float32)
        distance_weight = np.full(n_input, self.distance_weight_val, dtype=np.float32)

        best_distance = np.full(n_input, np.inf, dtype=np.float32)
        best_perturb = original_input.copy()

        immutable_mask = torch.tensor(~self._mutable_mask, dtype=torch.bool, device=self._device)

        for _ in range(self.n_iter):
            optimizer.zero_grad()

            p_model = _filter_hinge_loss(
                n_class=self.n_class,
                mask_vector=indicator.astype(bool),
                features=perturbed,
                sigma=torch.tensor(sigma, dtype=torch.float32, device=self._device),
                temperature=torch.tensor(temperature, dtype=torch.float32, device=self._device),
                model_fn=self._prob_from_input,
            )
            approx_prob = p_model[row_index, class_index]

            distance = distance_func(
                self.distance_function,
                perturbed,
                original_tensor,
                eps=1e-10,
            )

            prediction_loss = torch.tensor(indicator, dtype=torch.float32, device=self._device) * approx_prob
            distance_loss = torch.tensor(distance_weight, dtype=torch.float32, device=self._device) * distance
            total_loss = torch.mean(prediction_loss + distance_loss)

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                if self._clamp:
                    perturbed.copy_(torch.maximum(torch.minimum(perturbed, self._clip_max), self._clip_min))

                if self._enforce_mutability and bool(torch.any(immutable_mask)):
                    perturbed[:, immutable_mask] = original_tensor[:, immutable_mask]

                if self._enforce_encoding and self._cat_feature_indices:
                    perturbed.copy_(reconstruct_encoding_constraints(perturbed, self._cat_feature_indices))

                true_distance = (
                    distance_func(self.distance_function, perturbed, original_tensor, eps=0.0)
                    .detach()
                    .cpu()
                    .numpy()
                )

                current_prob = self._prob_from_input(perturbed, self.sigma_val, self.temp_val)
                current_predict = torch.argmax(current_prob, dim=1).detach().cpu().numpy()

                indicator = np.equal(ground_truth, current_predict).astype(np.float32)

                mask_flipped = np.not_equal(ground_truth, current_predict)
                mask_smaller_dist = np.less(true_distance, best_distance)

                temp_dist = best_distance.copy()
                temp_dist[mask_flipped] = true_distance[mask_flipped]
                best_distance[mask_smaller_dist] = temp_dist[mask_smaller_dist]

                perturbed_np = perturbed.detach().cpu().numpy()
                temp_perturb = best_perturb.copy()
                temp_perturb[mask_flipped] = perturbed_np[mask_flipped]
                best_perturb[mask_smaller_dist] = temp_perturb[mask_smaller_dist]

        df_cfs = pd.DataFrame(best_perturb, columns=self._feature_order, index=factuals.index)
        df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index)
        return df_cfs