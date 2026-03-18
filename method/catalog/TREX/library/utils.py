import numpy as np
import torch
import torch.nn as nn
from art.attacks.evasion import ElasticNet
from art.estimators.classification import PyTorchClassifier


class TreXCounterfactualTorch:
    """
    Single-class PyTorch implementation of:
    1) Initial counterfactual search (L1/L2 ElasticNet attack)
    2) TreX robustness refinement with Gaussian-volume objective
    """

    def __init__(
        self,
        model: nn.Module,
        input_dim: int,
        num_classes: int = 2,
        clamp=(0.0, 1.0),
        norm: int = 2,
        cf_steps: int = 60,
        cf_step_size: float = 0.02,
        cf_confidence: float = 0.5,
        tau: float = 0.75,
        K: int = 1000,
        sigma: float = 0.05,
        trex_max_steps: int = 20,
        trex_epsilon: float = 1.0,
        trex_step_size: float = 0.01,
        trex_p: float = 2,
        batch_size: int = 1,
        robust_class: int = 1,
        model_outputs_logits: bool = False,
        device: str = None,
    ):
        if norm not in (1, 2):
            raise ValueError("norm must be 1 or 2")
        if trex_p not in (1, 2, np.inf):
            raise ValueError("trex_p must be 1, 2, or np.inf")

        self.model = model
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.norm = norm

        self.cf_steps = cf_steps
        self.cf_step_size = cf_step_size
        self.cf_confidence = cf_confidence

        self.tau = tau
        self.K = K
        self.sigma = sigma
        self.trex_max_steps = trex_max_steps
        self.trex_epsilon = trex_epsilon
        self.trex_step_size = trex_step_size
        self.trex_p = trex_p

        self.batch_size = batch_size
        self.robust_class = robust_class
        self.model_outputs_logits = model_outputs_logits

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

        self._set_clamp(clamp)

    def _set_clamp(self, clamp):
        low, high = clamp
        self.per_feature_clamp = not np.isscalar(low) and not np.isscalar(high)

        if self.per_feature_clamp:
            low_np = np.asarray(low, dtype=np.float32).reshape(1, -1)
            high_np = np.asarray(high, dtype=np.float32).reshape(1, -1)
            self.clamp_low_t = torch.tensor(low_np, device=self.device)
            self.clamp_high_t = torch.tensor(high_np, device=self.device)
            self.clip_values = (float(low_np.min()), float(high_np.max()))
        else:
            self.clamp_low = float(low)
            self.clamp_high = float(high)
            self.clip_values = (self.clamp_low, self.clamp_high)

    def _proba(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if self.model_outputs_logits:
            out = torch.softmax(out, dim=1)
        return out

    @torch.no_grad()
    def _predict_labels_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self._proba(x).argmax(dim=1)

    def predict_labels(self, x_np: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
        return self._predict_labels_torch(x).cpu().numpy()

    def _make_art_classifier(self) -> PyTorchClassifier:
        return PyTorchClassifier(
            model=self.model,
            loss=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-3),
            input_shape=(self.input_dim,),
            nb_classes=self.num_classes,
            clip_values=self.clip_values,
        )

    def _initial_counterfactual(self, x_np: np.ndarray, y_np: np.ndarray) -> np.ndarray:
        classifier = self._make_art_classifier()
        decision_rule = "L1" if self.norm == 1 else "L2"
        beta = 1.0 if self.norm == 1 else 0.0

        attack = ElasticNet(
            classifier=classifier,
            confidence=self.cf_confidence,
            learning_rate=self.cf_step_size,
            beta=beta,
            max_iter=self.cf_steps,
            batch_size=min(self.batch_size, x_np.shape[0]),
            decision_rule=decision_rule,
            verbose=False,
        )

        y_onehot = np.eye(self.num_classes, dtype=np.float32)[y_np.astype(int)]
        x_adv = attack.generate(x=x_np.astype(np.float32), y=y_onehot)
        return np.asarray(x_adv, dtype=np.float32)

    def _clip_with_clamp(self, x: torch.Tensor) -> torch.Tensor:
        if self.per_feature_clamp:
            return torch.max(torch.min(x, self.clamp_high_t), self.clamp_low_t)
        return torch.clamp(x, min=self.clamp_low, max=self.clamp_high)

    @staticmethod
    def _normalize_l2(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        n = v.view(v.shape[0], -1).norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        return v / n

    def _update_delta(
        self,
        x0: torch.Tensor,
        delta: torch.Tensor,
        grad: torch.Tensor,
        step_size: float,
        epsilon: float,
        p: float,
    ) -> torch.Tensor:
        if p == np.inf:
            proposal = delta + step_size * grad.sign()
        elif p == 2:
            proposal = delta + step_size * self._normalize_l2(grad)
        elif p == 1:
            proposal = delta + step_size * grad.sign()
        else:
            raise ValueError("Unsupported p-norm")

        perturbation = proposal - x0
        flat = perturbation.view(perturbation.shape[0], -1)

        if p == 2:
            norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        elif p == np.inf:
            norm = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        else:  # p == 1
            norm = flat.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)

        coeff = torch.clamp(epsilon / norm, min=0.0, max=1.0)
        projected = x0 + perturbation * coeff
        return self._clip_with_clamp(projected)

    def _gaussian_volume(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mirrors repository gaussian_volume:
        score = mean(p_class(x + noise)) - mean(|p_class(x + noise) - p_class(x)|)
        """
        bsz, dim = x.shape
        noise = torch.randn(self.K, dim, device=x.device, dtype=x.dtype) * self.sigma

        scores = []
        for i in range(bsz):
            x_i = x[i : i + 1].repeat(self.K, 1)
            x_noisy = x_i + noise

            p_noisy = self._proba(x_noisy)[:, self.robust_class]
            p_clean = self._proba(x_i)[:, self.robust_class]
            score = p_noisy.mean() - torch.abs(p_noisy - p_clean).mean()
            scores.append(score)

        return torch.stack(scores, dim=0)

    def gaussian_score(self, x_np: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            score = self._gaussian_volume(x)
        return score.cpu().numpy()

    def _trex_refine_batch(self, x_batch_np: np.ndarray) -> np.ndarray:
        x0 = torch.as_tensor(x_batch_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            y_keep = self._predict_labels_torch(x0)

        optimal_adv = x0.clone()
        delta = x0.clone()

        for _ in range(self.trex_max_steps):
            delta.requires_grad_(True)
            scores = self._gaussian_volume(delta)

            if torch.all(scores >= self.tau):
                delta = delta.detach()
                break

            grad = torch.autograd.grad(scores.sum(), delta)[0]

            with torch.no_grad():
                delta_next = self._update_delta(
                    x0=x0,
                    delta=delta,
                    grad=grad,
                    step_size=self.trex_step_size,
                    epsilon=self.trex_epsilon,
                    p=self.trex_p,
                )

                keep_mask = y_keep == self._predict_labels_torch(delta_next)
                optimal_adv[keep_mask] = delta_next[keep_mask]
                delta = delta_next.detach()

        return optimal_adv.detach().cpu().numpy()

    def _trex_refine(self, x_adv_np: np.ndarray) -> np.ndarray:
        refined = []
        for i in range(0, x_adv_np.shape[0], self.batch_size):
            refined.append(self._trex_refine_batch(x_adv_np[i : i + self.batch_size]))
        return np.concatenate(refined, axis=0)

    def generate(
        self,
        x_np: np.ndarray,
        original_pred: np.ndarray = None,
        apply_trex: bool = True,
    ):
        x_np = np.asarray(x_np, dtype=np.float32)

        if original_pred is None:
            original_pred = self.predict_labels(x_np)

        # Stage 1: closest CF (IGD-style L1/L2 ElasticNet)
        x_cf = self._initial_counterfactual(x_np, original_pred)

        # Stage 2: TreX robustness refinement
        if apply_trex:
            x_cf = self._trex_refine(x_cf)

        pred_cf = self.predict_labels(x_cf)
        is_valid = pred_cf != original_pred
        return x_cf, pred_cf, is_valid