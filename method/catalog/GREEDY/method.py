from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from experiment_utils import deep_merge
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject


@register_method("GREEDY")
class Greedy(MethodObject):
    """
    PyTorch implementation of the GREEDY recourse method. [1]_.

    This is adapted from the original method as implemented in the CARLA library, which was implemented using TensorFlow.
    
    .. [1] "Generating Interpretable Counterfactual Explanations By Implicit Minimisation of Epistemic and Aleatoric Uncertainties"
        Lisa Schut, Oscar Key, Rory McGrathz, Luca Costabelloz, Bogdan Sacaleanuz, Medb Corcoranz, Yarin Galy.
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
            str(self._model._config.get("backend", "")).lower()
            if hasattr(self._model, "_config")
            else ""
        )
        if backend and backend != "pytorch":
            raise ValueError(f"GREEDY supports PyTorch backend only, got: {backend}")
        if not isinstance(self._model, torch.nn.Module):
            raise TypeError("GREEDY requires a differentiable torch.nn.Module model")

        # get configs from config file
        self.config = yaml.safe_load(open("method/catalog/GREEDY/library/config.yml", 'r'))
        
        # merge configs with user specified, if they exist
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

        self.lambda_param = float(self.config["lambda_param"])
        self.step_size = float(self.config["step_size"])
        self.max_iter = int(self.config["max_iter"])
        self.target_class = int(self.config["target_class"])
        if self.target_class not in (0, 1):
            raise ValueError("target_class must be 0 or 1")

        self.locked_features = self._resolve_locked_feature_indices(
            self.config.get("locked_features", [])
        )

        self._model.to(self._device)
        self._model.eval()

    def _resolve_locked_feature_indices(
        self, locked_features: Sequence[Union[int, str]]
    ) -> List[int]:
        locked_indices: List[int] = []

        for feature in locked_features:
            if isinstance(feature, (int, np.integer)):
                idx = int(feature)
            elif isinstance(feature, str):
                if feature not in self._feature_order:
                    raise ValueError(
                        f"Locked feature '{feature}' not found in feature order"
                    )
                idx = self._feature_order.index(feature)
            else:
                raise TypeError(
                    "locked_features entries must be feature indices (int) or names (str)"
                )

            if idx < 0 or idx >= len(self._feature_order):
                raise ValueError(
                    f"Locked feature index {idx} is out of bounds for "
                    f"{len(self._feature_order)} features"
                )

            locked_indices.append(idx)

        return sorted(set(locked_indices))

    def _positive_class_probability(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 1:
            if model_output.shape[0] == 2:
                model_output = model_output.unsqueeze(0)  # single sample, 2 classes
            else:
                model_output = model_output.unsqueeze(1)  # (n,) -> (n, 1)

        output_activation = (
            str(self._model._config.get("output_activation", "")).lower()
            if hasattr(self._model, "_config")
            else ""
        )

        if model_output.shape[1] == 1:
            # Sigmoid probability output or single-logit output.
            if output_activation == "sigmoid":
                pos_prob = model_output[:, 0]
            else:
                pos_prob = torch.sigmoid(model_output[:, 0])
        else:
            # 2-logit/2-probability output.
            if output_activation == "softmax":
                probs = model_output
            else:
                probs = torch.softmax(model_output, dim=1)
            pos_prob = probs[:, 1]

        return torch.clamp(pos_prob, min=1e-7, max=1.0 - 1e-7)

    def _classification_loss(self, positive_probability: torch.Tensor) -> torch.Tensor:
        target = torch.full_like(positive_probability, float(self.target_class))
        return F.binary_cross_entropy(positive_probability, target)

    def _generate_single_counterfactual(self, original_instance: np.ndarray) -> np.ndarray:
        original = torch.tensor(
            original_instance.reshape(1, -1),
            dtype=torch.float32,
            device=self._device,
        )

        x_cf = original.clone().detach().requires_grad_(True)

        locked_feature_tensor = None
        if self.locked_features:
            locked_feature_tensor = torch.tensor(
                self.locked_features, dtype=torch.long, device=self._device
            )

        for _ in range(self.max_iter):
            model_output = self._model(x_cf)
            pos_prob = self._positive_class_probability(model_output)

            distance_loss = torch.sum(torch.square(x_cf - original))
            classification_loss = self._classification_loss(pos_prob)
            loss = self.lambda_param * distance_loss + classification_loss

            gradients = torch.autograd.grad(
                loss, x_cf, retain_graph=False, create_graph=False
            )[0]

            with torch.no_grad():
                x_cf -= self.step_size * gradients

                if locked_feature_tensor is not None:
                    x_cf[:, locked_feature_tensor] = original[:, locked_feature_tensor]

                current_pos_prob = self._positive_class_probability(self._model(x_cf)).item()
                reached_target = (
                    (self.target_class == 1 and current_pos_prob >= 0.5)
                    or (self.target_class == 0 and current_pos_prob < 0.5)
                )
                if reached_target:
                    break

        return x_cf.detach().cpu().numpy().flatten()

    def get_counterfactuals(self, factuals: pd.DataFrame):
        factuals = factuals[self._feature_order]

        counterfactuals = []
        for _, row in factuals.iterrows():
            original_instance = row.to_numpy(dtype=np.float32)
            counterfactual = self._generate_single_counterfactual(original_instance)
            counterfactuals.append(counterfactual)

        df_cfs = pd.DataFrame(
            counterfactuals, columns=self._feature_order, index=factuals.index
        )
        df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index)
        return df_cfs