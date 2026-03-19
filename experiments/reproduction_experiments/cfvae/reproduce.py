import argparse
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pytest
import torch
from torch import nn
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method.catalog.CFVAE.model import _CFVAE
from method.catalog.CFVAE.resources import (
    DataLoader,
    load_adult_income_dataset,
    load_pretrained_binaries,
)


class BlackBox(nn.Module):
    """Keep the original reproduce-time target model definition for exact weight loading."""

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_dim = 10
        self.predict_net = nn.Sequential(
            nn.Linear(self.input_shape, self.hidden_dim),
            nn.Linear(self.hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.predict_net(x)


def target_class_validity(
    cf_model: _CFVAE,
    target_model: nn.Module,
    test_dataset: torch.Tensor,
    sample_sizes: List[int],
):
    results = []
    for sample_size in sample_sizes:
        test_x = test_dataset.float()
        test_y = torch.argmax(target_model(test_x), dim=1)

        valid_cf_count = 0
        for _ in range(sample_size):
            x_pred = cf_model(test_x, 1.0 - test_y)
            cf_label = torch.argmax(target_model(x_pred), dim=1)
            valid_cf_count += np.sum(test_y.cpu().numpy() != cf_label.cpu().numpy())

        dataset_size = test_x.shape[0]
        valid_cf_count = valid_cf_count / sample_size
        results.append(100 * valid_cf_count / dataset_size)

    return results


def constraint_feasibility_score_age(
    cf_model: _CFVAE,
    target_model: nn.Module,
    test_dataset: torch.Tensor,
    dataloader: DataLoader,
    sample_sizes: List[int],
):
    results_valid = []
    results_invalid = []
    for sample_size in sample_sizes:
        test_x = test_dataset.float()
        test_y = torch.argmax(target_model(test_x), dim=1)

        valid_change = 0
        invalid_change = 0
        dataset_size = 0
        for _ in range(sample_size):
            x_pred = cf_model(test_x, 1.0 - test_y)
            cf_label = torch.argmax(target_model(x_pred), dim=1)

            x_pred = dataloader.de_normalize_data(
                dataloader.get_decoded_data(x_pred.detach().cpu().numpy())
            )
            x_ori = dataloader.de_normalize_data(
                dataloader.get_decoded_data(test_x.detach().cpu().numpy())
            )

            age_idx = x_ori.columns.get_loc("age")
            for i in range(x_ori.shape[0]):
                if cf_label[i] == 0:
                    continue
                dataset_size += 1
                if x_pred.iloc[i, age_idx] >= x_ori.iloc[i, age_idx]:
                    valid_change += 1
                else:
                    invalid_change += 1

        valid_change = valid_change / sample_size
        invalid_change = invalid_change / sample_size
        dataset_size = dataset_size / sample_size

        results_valid.append(100 * valid_change / dataset_size)
        results_invalid.append(100 * invalid_change / dataset_size)

    return results_valid, results_invalid


def constraint_feasibility_score_age_ed(
    cf_model: _CFVAE,
    target_model: nn.Module,
    test_dataset: torch.Tensor,
    dataloader: DataLoader,
    sample_sizes: List[int],
):
    # Keep the original reproduce-time scoring map to preserve reference numbers.
    education_score = {
        "HS-grad": 0,
        "School": 0,
        "Bachelors": 1,
        "Assoc": 1,
        "Some-college": 1,
        "Masters": 2,
        "Prof-school": 2,
        "Doctorate": 3,
    }

    results_valid = []
    results_invalid = []
    for sample_size in sample_sizes:
        test_x = test_dataset.float()
        test_y = torch.argmax(target_model(test_x), dim=1)

        valid_change = 0
        invalid_change = 0
        dataset_size = 0
        for _ in range(sample_size):
            x_pred = cf_model(test_x, 1.0 - test_y)
            cf_label = torch.argmax(target_model(x_pred), dim=1)

            x_pred = dataloader.de_normalize_data(
                dataloader.get_decoded_data(x_pred.detach().cpu().numpy())
            )
            x_ori = dataloader.de_normalize_data(
                dataloader.get_decoded_data(test_x.detach().cpu().numpy())
            )

            age_idx = x_ori.columns.get_loc("age")
            ed_idx = x_ori.columns.get_loc("education")
            for i in range(x_ori.shape[0]):
                if cf_label[i] == 0:
                    continue
                dataset_size += 1

                if (
                    education_score[x_pred.iloc[i, ed_idx]]
                    < education_score[x_ori.iloc[i, ed_idx]]
                ):
                    invalid_change += 1
                elif (
                    education_score[x_pred.iloc[i, ed_idx]]
                    == education_score[x_ori.iloc[i, ed_idx]]
                ):
                    if x_pred.iloc[i, age_idx] >= x_ori.iloc[i, age_idx]:
                        valid_change += 1
                    else:
                        invalid_change += 1
                elif (
                    education_score[x_pred.iloc[i, ed_idx]]
                    > education_score[x_ori.iloc[i, ed_idx]]
                ):
                    if x_pred.iloc[i, age_idx] > x_ori.iloc[i, age_idx]:
                        valid_change += 1
                    else:
                        invalid_change += 1

        valid_change = valid_change / sample_size
        invalid_change = invalid_change / sample_size
        dataset_size = dataset_size / sample_size
        results_valid.append(100 * valid_change / dataset_size)
        results_invalid.append(100 * invalid_change / dataset_size)

    return results_valid, results_invalid


def cat_proximity(
    cf_model: _CFVAE,
    target_model: nn.Module,
    test_dataset: torch.Tensor,
    dataloader: DataLoader,
    sample_sizes: List[int],
):
    results = []
    for sample_size in sample_sizes:
        test_x = test_dataset.float()
        test_y = torch.argmax(target_model(test_x), dim=1)

        diff_count = 0
        for _ in range(sample_size):
            x_pred = cf_model(test_x, 1.0 - test_y)

            x_pred = dataloader.de_normalize_data(
                dataloader.get_decoded_data(x_pred.detach().cpu().numpy())
            )
            x_ori = dataloader.de_normalize_data(
                dataloader.get_decoded_data(test_x.detach().cpu().numpy())
            )

            for column in dataloader.categorical_feature_names:
                diff_count += np.sum(
                    np.array(x_ori[column]) != np.array(x_pred[column])
                )

        dataset_size = test_x.shape[0]
        diff_count = diff_count / sample_size
        results.append(-1 * diff_count / dataset_size)

    return results


def cont_proximity(
    cf_model: _CFVAE,
    target_model: nn.Module,
    test_dataset: torch.Tensor,
    dataloader: DataLoader,
    mad_feature_weights: Dict[str, float],
    sample_sizes: List[int],
):
    results = []
    for sample_size in sample_sizes:
        test_x = test_dataset.float()
        test_y = torch.argmax(target_model(test_x), dim=1)

        diff_amount = 0
        for _ in range(sample_size):
            x_pred = cf_model(test_x, 1.0 - test_y)

            x_pred = dataloader.de_normalize_data(
                dataloader.get_decoded_data(x_pred.detach().cpu().numpy())
            )
            x_ori = dataloader.de_normalize_data(
                dataloader.get_decoded_data(test_x.detach().cpu().numpy())
            )

            for column in dataloader.continuous_feature_names:
                diff_amount += (
                    np.sum(np.abs(x_ori[column] - x_pred[column]))
                    / mad_feature_weights[column]
                )

        dataset_size = test_x.shape[0]
        diff_amount = diff_amount / sample_size
        results.append(-1 * diff_amount / dataset_size)

    return results


def eval_adult(
    methods: Dict[str, str],
    encoded_size: int,
    target_model: nn.Module,
    val_dataset_np: np.ndarray,
    dataloader: DataLoader,
    mad_feature_weights: Dict[str, float],
    sample_sizes: List[int],
    constraint: str,
    n_test: int = 10,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose: bool = False,
):
    with torch.no_grad():
        results = {}

        np.random.shuffle(val_dataset_np)
        val_dataset = torch.tensor(val_dataset_np).to(device)

        target_model.eval().to(device)

        method_items = list(methods.items())
        method_iter = (
            tqdm(method_items, desc=f"Adult {constraint}", unit="method")
            if verbose
            else method_items
        )

        for name, path in method_iter:
            cf_val = {}

            cf_vae = _CFVAE(len(dataloader.encoded_feature_names), encoded_size)
            cf_vae.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
            cf_vae.eval().to(device)

            test_x = val_dataset.float().to(device)
            test_y = torch.argmax(target_model(test_x), dim=1).to(device)
            val_dataset = val_dataset[test_y == 0]

            repeat_iter = (
                tqdm(range(n_test), desc=name, leave=False, unit="run")
                if verbose
                else range(n_test)
            )
            for _ in repeat_iter:
                cf_val.setdefault("target_class_validity", []).append(
                    target_class_validity(
                        cf_vae, target_model, val_dataset, sample_sizes
                    )
                )
                if constraint == "age":
                    val, inval = constraint_feasibility_score_age(
                        cf_vae,
                        target_model,
                        val_dataset,
                        dataloader,
                        sample_sizes,
                    )
                else:
                    val, inval = constraint_feasibility_score_age_ed(
                        cf_vae,
                        target_model,
                        val_dataset,
                        dataloader,
                        sample_sizes,
                    )
                cf_val.setdefault("constraint_feasibility_score", []).append(
                    100 * np.array(val) / (np.array(val) + np.array(inval))
                )
                cf_val.setdefault("cont_proximity", []).append(
                    cont_proximity(
                        cf_vae,
                        target_model,
                        val_dataset,
                        dataloader,
                        mad_feature_weights,
                        sample_sizes,
                    )
                )
                cf_val.setdefault("cat_proximity", []).append(
                    cat_proximity(
                        cf_vae,
                        target_model,
                        val_dataset,
                        dataloader,
                        sample_sizes,
                    )
                )

            for key, value in cf_val.items():
                cf_val[key] = np.mean(np.array(value), axis=0)

            results[name] = cf_val

        return results


def compare_results(
    results: Dict,
    ref: Dict,
    tolerance: float = 1.0,
    raise_on_fail: bool = False,
    verbose: bool = False,
) -> bool:
    success = True
    failure_messages: List[str] = []

    for dataset_name, ref_methods in ref.items():
        dataset_results = results.get(dataset_name)
        if dataset_results is None:
            failure_messages.append(
                f"[MISSING] Dataset `{dataset_name}` not found in results."
            )
            if verbose:
                print(failure_messages[-1])
            success = False
            continue
        for method_name, ref_metrics in ref_methods.items():
            method_results = dataset_results.get(method_name)
            if method_results is None:
                failure_messages.append(
                    f"[MISSING] Method `{dataset_name}/{method_name}` not found in results."
                )
                if verbose:
                    print(failure_messages[-1])
                success = False
                continue
            for metric_name, ref_value in ref_metrics.items():
                result_value = method_results.get(metric_name)
                if result_value is None:
                    failure_messages.append(
                        f"[MISSING] Metric `{dataset_name}/{method_name}/{metric_name}` not found in results."
                    )
                    if verbose:
                        print(failure_messages[-1])
                    success = False
                    continue

                result_array = np.array(result_value, dtype=float)
                ref_array = np.array(ref_value, dtype=float)
                if result_array.shape != ref_array.shape:
                    failure_messages.append(
                        f"[SHAPE DIFF] `{dataset_name}/{method_name}/{metric_name}` result shape {result_array.shape} != reference shape {ref_array.shape}."
                    )
                    if verbose:
                        print(failure_messages[-1])
                    success = False
                    continue

                diff = np.abs(result_array - ref_array)
                if not np.all(diff <= tolerance):
                    failure_messages.append(
                        f"[DIFF] `{dataset_name}/{method_name}/{metric_name}` max diff {float(np.max(diff)):.6f} exceeds tolerance"
                    )
                    if verbose:
                        print(failure_messages[-1])
                    success = False
                elif verbose:
                    print(
                        f"[OK] `{dataset_name}/{method_name}/{metric_name}` max diff {float(np.max(diff)):.6f}"
                    )

    if raise_on_fail and failure_messages:
        raise AssertionError("\n".join(failure_messages))

    return success


def run_cfvae_reproduce(
    verbose: bool = False,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    torch.manual_seed(10_000_000)

    results = {}

    dataset = load_adult_income_dataset()
    params = {
        "dataframe": dataset.copy(),
        "continuous_features": ["age", "hours_per_week"],
        "outcome_name": "income",
    }
    dataloader = DataLoader(params)

    vae_test_dataset = np.load(load_pretrained_binaries("adult-test-set.npy"))
    vae_test_dataset = vae_test_dataset[vae_test_dataset[:, -1] == 0, :]
    vae_test_dataset = vae_test_dataset[:, :-1]

    mad_feature_weights = {
        "age": 10.0,
        "hours_per_week": 3.0,
    }

    data_size = len(dataloader.encoded_feature_names)
    target_model = BlackBox(data_size)
    target_model.load_state_dict(
        torch.load(
            load_pretrained_binaries("adult-target-model.pth"),
            map_location=torch.device("cpu"),
        )
    )
    target_model.eval()

    results["adult-age"] = eval_adult(
        {
            "BaseCVAE": load_pretrained_binaries(
                "adult-margin-0.165-validity_reg-42.0-epoch-25-base-gen.pth"
            ),
            "BaseVAE": load_pretrained_binaries(
                "adult-margin-0.369-validity_reg-73.0-ae_reg-2.0-epoch-25-ae-gen.pth"
            ),
            "ModelApprox": load_pretrained_binaries(
                "adult-margin-0.764-constraint-reg-192.0-validity_reg-29.0-epoch-25-unary-gen.pth"
            ),
            "ExampleBased": load_pretrained_binaries(
                "adult-eval-case-0-supervision-limit-100-const-case-0-margin-0.084-oracle_reg-5999.0-validity_reg-159.0-epoch-50-oracle-gen.pth"
            ),
        },
        encoded_size=10,
        target_model=target_model,
        val_dataset_np=vae_test_dataset,
        dataloader=dataloader,
        mad_feature_weights=mad_feature_weights,
        sample_sizes=[1, 2, 3],
        constraint="age",
        verbose=verbose,
    )

    results["adult-age-ed"] = eval_adult(
        {
            "BaseCVAE": load_pretrained_binaries(
                "adult-margin-0.165-validity_reg-42.0-epoch-25-base-gen.pth"
            ),
            "BaseVAE": load_pretrained_binaries(
                "adult-margin-0.369-validity_reg-73.0-ae_reg-2.0-epoch-25-ae-gen.pth"
            ),
            "ModelApprox": load_pretrained_binaries(
                "adult-margin-0.344-constraint-reg-87.0-validity_reg-76.0-epoch-25-unary-ed-gen.pth"
            ),
            "ExampleBased": load_pretrained_binaries(
                "adult-eval-case-0-supervision-limit-100-const-case-1-margin-0.117-oracle_reg-3807.0-validity_reg-175.0-epoch-50-oracle-gen.pth"
            ),
        },
        encoded_size=10,
        target_model=target_model,
        val_dataset_np=vae_test_dataset,
        dataloader=dataloader,
        mad_feature_weights=mad_feature_weights,
        sample_sizes=[1, 2, 3],
        constraint="age-ed",
        verbose=verbose,
    )

    return results


@pytest.mark.parametrize("tolerance", [1.0])
def test_cfvae_reproduce(tolerance):
    reference_results = {
        "adult-age": {
            "BaseCVAE": {
                "target_class_validity": ([100.0, 100.0, 100.0]),
                "constraint_feasibility_score": (
                    [56.82554814, 56.93040991, 56.9399428]
                ),
                "cont_proximity": ([-2.24059021, -2.254801, -2.24498223]),
                "cat_proximity": ([-3.26024786, -3.26024786, -3.26024786]),
            },
            "BaseVAE": {
                "target_class_validity": ([100.0, 100.0, 100.0]),
                "constraint_feasibility_score": (
                    [42.85033365, 43.11248808, 42.99014935]
                ),
                "cont_proximity": ([-2.66647616, -2.66112855, -2.66558379]),
                "cat_proximity": ([-3.12011439, -3.12011439, -3.12011439]),
            },
            "ModelApprox": {
                "target_class_validity": ([99.5900858, 99.5900858, 99.55513187]),
                "constraint_feasibility_score": (
                    [84.26668079, 83.60122942, 83.58960991]
                ),
                "cont_proximity": ([-2.73294234, -2.73510212, -2.73455902]),
                "cat_proximity": ([-3.26167779, -3.26172545, -3.26151891]),
            },
            "ExampleBased": {
                "target_class_validity": ([99.52335558, 99.52812202, 99.46298062]),
                "constraint_feasibility_score": (
                    [74.03769743, 74.18632186, 74.21309514]
                ),
                "cont_proximity": ([-6.80110826, -6.7996033, -6.80052672]),
                "cat_proximity": ([-3.72411821, -3.7242612, -3.72443597]),
            },
        },
        "adult-age-ed": {
            "BaseCVAE": {
                "target_class_validity": ([100.0, 100.0, 100.0]),
                "constraint_feasibility_score": (
                    [57.01620591, 57.20209724, 56.80012711]
                ),
                "cont_proximity": ([-2.25337728, -2.24797755, -2.24245525]),
                "cat_proximity": ([-3.26024786, -3.26024786, -3.26024786]),
            },
            "BaseVAE": {
                "target_class_validity": ([100.0, 100.0, 100.0]),
                "constraint_feasibility_score": (
                    [42.59294566, 43.06482364, 43.02192564]
                ),
                "cont_proximity": ([-2.6661057, -2.66556556, -2.6650458]),
                "cat_proximity": ([-3.12011439, -3.12011439, -3.12011439]),
            },
            "ModelApprox": {
                "target_class_validity": ([100.0, 100.0, 100.0]),
                "constraint_feasibility_score": (
                    [79.5042898, 79.32793136, 79.30092151]
                ),
                "cont_proximity": ([-2.90320554, -2.90264891, -2.90204051]),
                "cat_proximity": ([-3.22097235, -3.22054337, -3.21916111]),
            },
            "ExampleBased": {
                "target_class_validity": ([99.93326978, 99.89513823, 99.92691452]),
                "constraint_feasibility_score": (
                    [66.35181132, 66.50929669, 66.46958292]
                ),
                "cont_proximity": ([-3.20324281, -3.2077721, -3.20357766]),
                "cat_proximity": ([-3.5914204, -3.59394662, -3.59634573]),
            },
        },
    }

    compare_results(
        run_cfvae_reproduce(),
        reference_results,
        tolerance=tolerance,
        raise_on_fail=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Allowed absolute difference when comparing against reference.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars and per-metric comparison output.",
    )
    args = parser.parse_args()
    print("Running CFVAE reproduction on Adult reference artifacts...")
    results = run_cfvae_reproduce(verbose=not args.quiet)
    compare_results(
        results,
        {
            "adult-age": {
                "BaseCVAE": {
                    "target_class_validity": ([100.0, 100.0, 100.0]),
                    "constraint_feasibility_score": (
                        [56.82554814, 56.93040991, 56.9399428]
                    ),
                    "cont_proximity": ([-2.24059021, -2.254801, -2.24498223]),
                    "cat_proximity": ([-3.26024786, -3.26024786, -3.26024786]),
                },
                "BaseVAE": {
                    "target_class_validity": ([100.0, 100.0, 100.0]),
                    "constraint_feasibility_score": (
                        [42.85033365, 43.11248808, 42.99014935]
                    ),
                    "cont_proximity": ([-2.66647616, -2.66112855, -2.66558379]),
                    "cat_proximity": ([-3.12011439, -3.12011439, -3.12011439]),
                },
                "ModelApprox": {
                    "target_class_validity": ([99.5900858, 99.5900858, 99.55513187]),
                    "constraint_feasibility_score": (
                        [84.26668079, 83.60122942, 83.58960991]
                    ),
                    "cont_proximity": ([-2.73294234, -2.73510212, -2.73455902]),
                    "cat_proximity": ([-3.26167779, -3.26172545, -3.26151891]),
                },
                "ExampleBased": {
                    "target_class_validity": ([99.52335558, 99.52812202, 99.46298062]),
                    "constraint_feasibility_score": (
                        [74.03769743, 74.18632186, 74.21309514]
                    ),
                    "cont_proximity": ([-6.80110826, -6.7996033, -6.80052672]),
                    "cat_proximity": ([-3.72411821, -3.7242612, -3.72443597]),
                },
            },
            "adult-age-ed": {
                "BaseCVAE": {
                    "target_class_validity": ([100.0, 100.0, 100.0]),
                    "constraint_feasibility_score": (
                        [57.01620591, 57.20209724, 56.80012711]
                    ),
                    "cont_proximity": ([-2.25337728, -2.24797755, -2.24245525]),
                    "cat_proximity": ([-3.26024786, -3.26024786, -3.26024786]),
                },
                "BaseVAE": {
                    "target_class_validity": ([100.0, 100.0, 100.0]),
                    "constraint_feasibility_score": (
                        [42.59294566, 43.06482364, 43.02192564]
                    ),
                    "cont_proximity": ([-2.6661057, -2.66556556, -2.6650458]),
                    "cat_proximity": ([-3.12011439, -3.12011439, -3.12011439]),
                },
                "ModelApprox": {
                    "target_class_validity": ([100.0, 100.0, 100.0]),
                    "constraint_feasibility_score": (
                        [79.5042898, 79.32793136, 79.30092151]
                    ),
                    "cont_proximity": ([-2.90320554, -2.90264891, -2.90204051]),
                    "cat_proximity": ([-3.22097235, -3.22054337, -3.21916111]),
                },
                "ExampleBased": {
                    "target_class_validity": ([99.93326978, 99.89513823, 99.92691452]),
                    "constraint_feasibility_score": (
                        [66.35181132, 66.50929669, 66.46958292]
                    ),
                    "cont_proximity": ([-3.20324281, -3.2077721, -3.20357766]),
                    "cat_proximity": ([-3.5914204, -3.59394662, -3.59634573]),
                },
            },
        },
        tolerance=args.tolerance,
        raise_on_fail=True,
        verbose=not args.quiet,
    )
    print("CFVAE reproduction passed.")


if __name__ == "__main__":
    main()
