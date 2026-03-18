from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml

from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from experiment_utils import deep_merge, reconstruct_encoding_constraints
from method.method_factory import register_method
from method.method_object import MethodObject
from model.catalog.autoencoder.autoencoder import Autoencoder
from model.catalog.autoencoder.library.training import train_autoencoder
from model.model_object import ModelObject


@register_method("CEM")
class CEM(MethodObject):
    """
    implementation of CEM [1]_.

    .. [1] Amit Dhurandhar, Pin-Yu Chen, Ronny Luss, Chun-Chen Tu, Paishun Ting, Karthikeyan Shanmugam,and Payel Das.
            2018. Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives.
            In Advances in Neural Information Processing Systems344(NeurIPS).
    """

    def __init__(
        self,
        data: DataObject,
        model: ModelObject,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(data, model, config_override=config_override)

        self.config = yaml.safe_load(open("method/catalog/CEM/library/config.yml", "r"))
        
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

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
                f"CEM currently supports only PyTorch backend, got: {backend}"
            )

        self._kappa = float(self.config["kappa"])
        self._mode = self.config["mode"]
        self._batch_size = int(self.config["batch_size"])
        self._num_classes = int(self.config["num_classes"])
        self._beta = float(self.config["beta"])
        self._gamma = float(self.config["gamma"])
        self._init_learning_rate = float(self.config["init_learning_rate"])
        self._binary_search_steps = int(self.config["binary_search_steps"])
        self._max_iterations = int(self.config["max_iterations"])
        self._initial_const = float(self.config["initial_const"])

        if self._mode not in ["PP", "PN"]:
            raise ValueError("Mode not known, please use either PP or PN")
        if self._num_classes != 2:
            raise ValueError(
                "Current CEM implementation supports binary classification only"
            )
        if self._batch_size != 1:
            raise ValueError(
                "Current CEM implementation supports batch_size=1 only"
            )

        self._clip_min, self._clip_max = self._build_clip_bounds()
        self._AE = self._load_ae()

    def _build_clip_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
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
                clip_min = np.full(
                    len(self._feature_order), float(clip_min), dtype=np.float32
                )

        if clip_max_cfg is None:
            clip_max = np.max(x_train_np, axis=0).astype(np.float32)
        else:
            clip_max = np.asarray(clip_max_cfg, dtype=np.float32)
            if clip_max.ndim == 0:
                clip_max = np.full(
                    len(self._feature_order), float(clip_max), dtype=np.float32
                )

        if (
            clip_min.shape[0] != len(self._feature_order)
            or clip_max.shape[0] != len(self._feature_order)
        ):
            raise ValueError(
                "clip_min/clip_max must be scalar or match number of input features"
            )

        clip_min_t = torch.tensor(
            clip_min, dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        clip_max_t = torch.tensor(
            clip_max, dtype=torch.float32, device=self._device
        ).unsqueeze(0)

        return clip_min_t, clip_max_t

    def _load_ae(self) -> Autoencoder:
        ae_params = self.config["ae_params"]
        data_name = self.config["data_name"] if self.config["data_name"] else "cem"

        ae = Autoencoder(
            data_name=data_name,
            layers=[len(self._feature_order)] + ae_params["hidden_layer"],
        )

        if ae_params.get("train_ae", True):
            x_train, _ = self._model.get_train_data()
            return train_autoencoder(
                ae=ae,
                xtrain=x_train,
                feature_order=self._feature_order,
                epochs=int(ae_params.get("epochs", 5)),
                batch_size=int(ae_params.get("batch_size", 128)),
                lr=float(ae_params.get("lr", 1e-3)),
                save=True,
            )

        try:
            return ae.load(input_shape=len(self._feature_order))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Loading of Autoencoder failed. {str(exc)}")

    def _predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(self._model, torch.nn.Module):
            raise TypeError("CEM requires a differentiable torch.nn.Module model")

        scores = self._model(x)

        if scores.ndim == 1:
            scores = scores.unsqueeze(1)

        if scores.shape[1] == 1:
            pos = scores[:, 0]
            scores = torch.stack([1.0 - pos, pos], dim=1)

        return scores

    def _compute_l_norm(self, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.sum(torch.abs(delta), dim=1), torch.sum(torch.square(delta), dim=1)

    def _compute_target_lab_score(
        self, target_label: torch.Tensor, label_score: torch.Tensor
    ) -> torch.Tensor:
        return torch.sum(target_label * label_score, dim=1)

    def _compute_non_target_lab_score(
        self, target_label: torch.Tensor, label_score: torch.Tensor
    ) -> torch.Tensor:
        masked = (1.0 - target_label) * label_score - (target_label * 10000.0)
        return torch.max(masked, dim=1).values

    def _compute_AE_loss(self, delta_input: torch.Tensor, gamma: float) -> torch.Tensor:
        reconstruction = self._AE(delta_input)
        return gamma * torch.square(torch.norm(reconstruction - delta_input, p=2))

    def _compute_attack_loss(
        self,
        nontarget_label_score: torch.Tensor,
        target_label_score: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        sign = 1.0 if mode == "PP" else -1.0
        return torch.clamp(
            (sign * nontarget_label_score) - (sign * target_label_score) + self._kappa,
            min=0.0,
        )

    def _compute_AE_dist(
        self, adv: torch.Tensor, delta: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        delta_input_ae = delta if self._mode == "PP" else adv
        return self._compute_AE_loss(delta_input_ae, gamma)

    def _get_label_score(self, delta: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
        enforce_input = delta if self._mode == "PP" else adv
        return self._predict_scores(enforce_input)

    def _get_conditions(
        self, adv: torch.Tensor, orig: torch.Tensor, beta: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond_greater = (adv - orig > beta).float()
        cond_less_equal = ((adv - orig).abs() <= beta).float()
        cond_less = (adv - orig < (-beta)).float()
        return cond_greater, cond_less_equal, cond_less

    def _compute_with_mode(
        self, assign_adv_s: torch.Tensor, orig: torch.Tensor
    ) -> torch.Tensor:
        cond_greater, cond_less_equal, _ = self._get_conditions(assign_adv_s, orig)

        if self._mode == "PP":
            assign_adv_s = (cond_less_equal * assign_adv_s) + (cond_greater * orig)
        elif self._mode == "PN":
            assign_adv_s = (cond_greater * assign_adv_s) + (cond_less_equal * orig)

        return assign_adv_s

    def _compute_adv_s(
        self, zt: float, orig: torch.Tensor, adv: torch.Tensor, assign_adv: torch.Tensor
    ) -> torch.Tensor:
        assign_adv_s = assign_adv + (zt * (assign_adv - adv))
        return self._compute_with_mode(assign_adv_s, orig)

    def _compute_adv(
        self, orig: torch.Tensor, adv_s: torch.Tensor, beta: float
    ) -> torch.Tensor:
        cond_greater, cond_less_equal, cond_less = self._get_conditions(
            adv_s, orig, beta
        )

        upper = torch.minimum(adv_s - beta, self._clip_max)
        lower = torch.maximum(adv_s + beta, self._clip_min)

        assign_adv = (
            (cond_greater * upper)
            + (cond_less_equal * orig)
            + (cond_less * lower)
        )
        return self._compute_with_mode(assign_adv, orig)

    def _compare(self, x: Union[np.ndarray, float, int], y: int) -> bool:
        if not isinstance(x, (float, int, np.integer)):
            x_local = np.copy(x)
            if self._mode == "PP":
                x_local[y] -= self._kappa
            elif self._mode == "PN":
                x_local[y] += self._kappa
            x_local = np.argmax(x_local)

            if self._mode == "PP":
                return int(x_local) == int(y)
            return int(x_local) != int(y)

        if self._mode == "PP":
            return int(x) == int(y)
        return int(x) != int(y)

    def _attack(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        batch_size = self._batch_size
        if X.shape[0] < batch_size:
            raise ValueError("Input batch is smaller than configured batch_size")

        Const_LB = np.zeros(batch_size, dtype=np.float32)
        CONST = np.ones(batch_size, dtype=np.float32) * self._initial_const
        Const_UB = np.ones(batch_size, dtype=np.float32) * 1e10

        overall_best_dist = [1e10] * batch_size
        overall_best_attack = np.array(
            [np.zeros(X[0].shape, dtype=np.float32)] * batch_size
        )

        for _ in range(self._binary_search_steps):
            input_batch = torch.tensor(
                X[:batch_size], dtype=torch.float32, device=self._device
            )
            label_batch = torch.tensor(
                Y[:batch_size], dtype=torch.float32, device=self._device
            )
            const_t = torch.tensor(CONST, dtype=torch.float32, device=self._device)

            adv = input_batch.clone().detach()
            adv_s = input_batch.clone().detach().requires_grad_(True)

            optimizer = torch.optim.SGD([adv_s], lr=self._init_learning_rate)

            current_step_best_dist = [1e10] * batch_size
            current_step_best_score = [-1] * batch_size

            for i in range(self._max_iterations):
                if self._max_iterations > 1:
                    ratio = 1.0 - (float(i) / float(self._max_iterations - 1))
                    optimizer.param_groups[0]["lr"] = (
                        self._init_learning_rate * np.sqrt(max(ratio, 0.0))
                    )

                optimizer.zero_grad()

                zt = float(i) / (float(i) + 3.0)
                assign_adv = self._compute_adv(input_batch, adv_s, self._beta)
                assign_adv_s = self._compute_adv_s(
                    zt, input_batch, adv.detach(), assign_adv
                )

                delta_s = input_batch - assign_adv_s
                _, l2_dist_s = self._compute_l_norm(delta_s)

                score_s = self._get_label_score(delta_s, assign_adv_s)
                target_s = self._compute_target_lab_score(label_batch, score_s)
                nontarget_s = self._compute_non_target_lab_score(label_batch, score_s)

                loss_attack_s = torch.sum(
                    const_t
                    * self._compute_attack_loss(nontarget_s, target_s, self._mode)
                )
                loss_l2_s = torch.sum(l2_dist_s)
                loss_ae_s = self._compute_AE_dist(assign_adv_s, delta_s, self._gamma)

                loss_to_optimize = loss_attack_s + loss_l2_s + loss_ae_s
                loss_to_optimize.backward()
                optimizer.step()

                with torch.no_grad():
                    adv_prev = adv.clone()
                    assign_adv_post = self._compute_adv(input_batch, adv_s, self._beta)
                    adv.copy_(assign_adv_post)
                    adv_s.copy_(
                        self._compute_adv_s(zt, input_batch, adv_prev, assign_adv_post)
                    )

                    delta = input_batch - adv
                    l1_dist, l2_dist = self._compute_l_norm(delta)
                    loss_l2_l1_dist = l2_dist + (self._beta * l1_dist)
                    output_score = self._get_label_score(delta, adv)

                loss_l2_l1_np = loss_l2_l1_dist.detach().cpu().numpy()
                output_score_np = output_score.detach().cpu().numpy()
                adv_np = adv.detach().cpu().numpy()

                for batch_idx, (the_dist, the_score, the_adv) in enumerate(
                    zip(loss_l2_l1_np, output_score_np, adv_np)
                ):
                    target_idx = int(torch.argmax(label_batch[batch_idx]).item())

                    if (
                        the_dist < current_step_best_dist[batch_idx]
                        and self._compare(the_score, target_idx)
                    ):
                        current_step_best_dist[batch_idx] = float(the_dist)
                        current_step_best_score[batch_idx] = int(np.argmax(the_score))

                    if (
                        the_dist < overall_best_dist[batch_idx]
                        and self._compare(the_score, target_idx)
                    ):
                        overall_best_dist[batch_idx] = float(the_dist)
                        overall_best_attack[batch_idx] = the_adv

            for batch_idx in range(batch_size):
                target_idx = int(np.argmax(Y[batch_idx]))

                if (
                    self._compare(current_step_best_score[batch_idx], target_idx)
                    and current_step_best_score[batch_idx] != -1
                ):
                    Const_UB[batch_idx] = min(Const_UB[batch_idx], CONST[batch_idx])
                    if Const_UB[batch_idx] < 1e9:
                        CONST[batch_idx] = (
                            Const_LB[batch_idx] + Const_UB[batch_idx]
                        ) / 2.0
                else:
                    Const_LB[batch_idx] = max(Const_LB[batch_idx], CONST[batch_idx])
                    if Const_UB[batch_idx] < 1e9:
                        CONST[batch_idx] = (
                            Const_LB[batch_idx] + Const_UB[batch_idx]
                        ) / 2.0
                    else:
                        CONST[batch_idx] *= 10.0

        best = np.array(overall_best_attack[0], dtype=np.float32)
        return best.reshape((1,) + best.shape)

    def _counterfactual_search(self, instance: np.ndarray) -> np.ndarray:
        input_instance = np.expand_dims(instance.astype(np.float32), axis=0)
        input_tensor = torch.tensor(
            input_instance, dtype=torch.float32, device=self._device
        )

        with torch.no_grad():
            prob = self._predict_scores(input_tensor).detach().cpu().numpy()[0]
        orig_class = int(np.argmax(prob))

        target_vec = np.eye(self._num_classes, dtype=np.float32)[orig_class].reshape(
            1, -1
        )
        counterfactual = self._attack(input_instance, target_vec)
        return counterfactual.reshape(-1)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        factuals = factuals.reset_index(drop=True)
        factuals = factuals[self._feature_order]

        encoded_feature_names = self._data.get_categorical_features(expanded=True)
        cat_features_indices = []
        for features in encoded_feature_names:
            indices = [factuals.columns.get_loc(feat) for feat in features]
            cat_features_indices.append(indices)

        cfs = []
        for _, row in factuals.iterrows():
            cfs.append(self._counterfactual_search(row.to_numpy(dtype=np.float32)))

        cfs_np = np.asarray(cfs, dtype=np.float32)
        df_cfs = pd.DataFrame(cfs_np, columns=self._feature_order, index=factuals.index)

        if len(cat_features_indices) > 0:
            cf_tensor = torch.tensor(
                df_cfs.to_numpy(dtype=np.float32),
                dtype=torch.float32,
                device=self._device,
            )
            cf_tensor = reconstruct_encoding_constraints(cf_tensor, cat_features_indices)
            df_cfs = pd.DataFrame(
                cf_tensor.detach().cpu().numpy(),
                columns=self._feature_order,
                index=factuals.index,
            )

        df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index)
        return df_cfs


