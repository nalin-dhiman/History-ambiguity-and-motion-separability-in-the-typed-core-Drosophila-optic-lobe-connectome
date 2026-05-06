"""Rule-search evaluation protocols for local plasticity discovery."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import sparse

from thermodynamic_connectomics.learning.history_gates import gates_for_graph
from thermodynamic_connectomics.learning.local_rules import normalize_per_target
from thermodynamic_connectomics.learning.metrics import classification_metrics, softmax
from thermodynamic_connectomics.learning.plasticity_metrics import plasticity_efficiency, split_train_test, train_linear_probe
from thermodynamic_connectomics.learning.rules import BASELINE_RULES, LocalRule, make_rule

EPS = 1e-12


def representation(X: np.ndarray, W: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(X @ W, dtype=np.float32)


def _init_readout(n_features: int, n_classes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.01, size=(n_features, n_classes)).astype(np.float32), np.zeros(n_classes, dtype=np.float32)


def _readout_step(H: np.ndarray, y: np.ndarray, readout: np.ndarray, bias: np.ndarray, lr: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    n_classes = readout.shape[1]
    logits = H @ readout + bias
    prob = softmax(logits)
    onehot = np.eye(n_classes, dtype=np.float32)[y]
    grad = (prob - onehot) / max(len(y), 1)
    readout = readout - lr * (H.T @ grad)
    bias = bias - lr * grad.sum(axis=0)
    metrics = classification_metrics(y, H @ readout + bias)
    return readout, bias, metrics


def _apply_plasticity(
    graph,
    X_train: np.ndarray,
    y_train: np.ndarray,
    rule: LocalRule,
    gates: dict[str, np.ndarray],
    epochs: int,
    readout_lr: float,
    plasticity_lr: float,
    decay: float,
    seed: int,
) -> tuple[sparse.csr_matrix, float, float]:
    W = normalize_per_target(graph.W.copy())
    n_classes = int(y_train.max()) + 1
    H = representation(X_train, W)
    readout, bias = _init_readout(H.shape[1], n_classes, seed)
    total_l1 = 0.0
    total_l2_sq = 0.0
    rule.reset(graph, seed)
    chance = 1.0 / max(n_classes, 1)
    for _ in range(max(1, epochs)):
        H = representation(X_train, W)
        readout, bias, metrics = _readout_step(H, y_train, readout, bias, readout_lr)
        reward = 1.0 if metrics["accuracy"] > chance else -1.0
        W, l1, l2 = rule.update(
            W,
            pre=X_train.mean(axis=0),
            post=H.mean(axis=0),
            reward=reward,
            gates=gates,
            lr=plasticity_lr,
            decay=decay,
        )
        total_l1 += l1
        total_l2_sq += l2 * l2
    return W, float(total_l1), float(math.sqrt(total_l2_sq))


def evaluate_rule(
    graph,
    task_data: dict,
    rule: LocalRule,
    protocol: str,
    seed: int,
    epochs: int,
    readout_lr: float,
    plasticity_lr: float,
    decay: float,
    probe_epochs: int = 80,
) -> dict[str, float]:
    """Evaluate a local plasticity rule under online or representation-probe protocol."""

    X = task_data["X_source"]
    y = task_data["y"]
    X_train, X_test, y_train, y_test = split_train_test(X, y, seed=seed)
    gates = gates_for_graph(graph)
    W0 = normalize_per_target(graph.W.copy())
    if protocol == "online_readout":
        n_classes = int(y.max()) + 1
        readout, bias = _init_readout(W0.shape[1], n_classes, seed)
        init_metrics = classification_metrics(y_test, representation(X_test, W0) @ readout + bias)
        W = W0.copy()
        total_l1 = 0.0
        total_l2_sq = 0.0
        rule.reset(graph, seed)
        chance = 1.0 / max(n_classes, 1)
        final_metrics = init_metrics
        for _ in range(max(1, epochs)):
            H_train = representation(X_train, W)
            readout, bias, train_metrics = _readout_step(H_train, y_train, readout, bias, readout_lr)
            reward = 1.0 if train_metrics["accuracy"] > chance else -1.0
            W, l1, l2 = rule.update(
                W,
                pre=X_train.mean(axis=0),
                post=H_train.mean(axis=0),
                reward=reward,
                gates=gates,
                lr=plasticity_lr,
                decay=decay,
            )
            total_l1 += l1
            total_l2_sq += l2 * l2
            final_metrics = classification_metrics(y_test, representation(X_test, W) @ readout + bias)
        before_acc = init_metrics["accuracy"]
        before_mi = init_metrics["task_MI_bits"]
        after_acc = final_metrics["accuracy"]
        after_mi = final_metrics["task_MI_bits"]
    elif protocol == "representation_probe":
        before_metrics = train_linear_probe(
            representation(X_train, W0),
            y_train,
            representation(X_test, W0),
            y_test,
            seed=seed,
            epochs=probe_epochs,
            lr=readout_lr,
        )
        W, total_l1, total_l2 = _apply_plasticity(
            graph,
            X_train,
            y_train,
            rule,
            gates,
            epochs=epochs,
            readout_lr=readout_lr,
            plasticity_lr=plasticity_lr,
            decay=decay,
            seed=seed,
        )
        after_metrics = train_linear_probe(
            representation(X_train, W),
            y_train,
            representation(X_test, W),
            y_test,
            seed=seed,
            epochs=probe_epochs,
            lr=readout_lr,
        )
        before_acc = before_metrics["accuracy"]
        before_mi = before_metrics["task_MI_bits"]
        after_acc = after_metrics["accuracy"]
        after_mi = after_metrics["task_MI_bits"]
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    delta_acc = float(after_acc - before_acc)
    delta_mi = float(after_mi - before_mi)
    total_l2 = float(math.sqrt(total_l2_sq)) if protocol == "online_readout" else float(total_l2)
    return {
        "final_accuracy": float(after_acc),
        "final_task_MI_bits": float(after_mi),
        "before_accuracy": float(before_acc),
        "before_task_MI_bits": float(before_mi),
        "plasticity_delta_accuracy": delta_acc,
        "plasticity_delta_MI": delta_mi,
        "total_update_L1": float(total_l1),
        "total_update_L2": float(total_l2),
        "eta_plastic": plasticity_efficiency(delta_mi, float(total_l1)),
    }


def symbolic_rule_specs(max_rules: int = 200) -> list[dict]:
    """Generate sparse symbolic rules with at most three active basis terms."""

    basis = [f"b{i}" for i in range(1, 11)]
    coeffs = [-1.0, -0.3, -0.1, 0.1, 0.3, 1.0]
    specs = []
    for k in [1, 2, 3]:
        for combo in itertools.combinations(basis, k):
            for coef_tuple in itertools.product(coeffs, repeat=k):
                active = dict(zip(combo, coef_tuple))
                specs.append({"name": "symbolic:" + "+".join(f"{v:g}{b}" for b, v in active.items()), "active_basis": active})
                if len(specs) >= max_rules:
                    return specs
    return specs


def make_rules(rule_names: list[str], symbolic_specs: list[dict] | None = None) -> list[LocalRule]:
    rules = []
    for name in rule_names:
        rules.append(make_rule(name))
    for spec in symbolic_specs or []:
        rules.append(make_rule(spec["name"], active_basis=spec["active_basis"]))
    return rules


def score_rule_results(results: pd.DataFrame) -> pd.DataFrame:
    """Score rules on discovery results using the requested conservative objective."""

    if results.empty:
        return pd.DataFrame()
    rows = []
    for (rule_name, protocol), group in results.groupby(["rule_name", "protocol"]):
        real = group[group["condition"].eq("real")]
        degree = group[group["condition"].eq("degree_preserving_null")]
        key = ["seed", "sample_id"]
        paired = real.set_index(key).join(
            degree.set_index(key)[["plasticity_delta_MI", "plasticity_delta_accuracy"]],
            rsuffix="_degree",
            how="inner",
        )
        real_minus_degree_mi = (
            paired["plasticity_delta_MI"] - paired["plasticity_delta_MI_degree"]
            if not paired.empty
            else pd.Series([0.0])
        )
        real_minus_degree_acc = (
            paired["plasticity_delta_accuracy"] - paired["plasticity_delta_accuracy_degree"]
            if not paired.empty
            else pd.Series([0.0])
        )
        mean_gain = float(real["plasticity_delta_MI"].mean()) if not real.empty else 0.0
        mean_cost = float(real["total_update_L1"].mean()) if not real.empty else 0.0
        instability = float(real["plasticity_delta_MI"].std(ddof=0)) if len(real) > 1 else 0.0
        complexity = 0
        if rule_name.startswith("symbolic:"):
            complexity = rule_name.count("b")
        elif rule_name != "no_plasticity":
            complexity = 1
        positive_gain = max(mean_gain, 0.0)
        score = (
            positive_gain
            + 0.5 * float(real_minus_degree_mi.mean())
            + 0.2 * float(real_minus_degree_acc.mean())
            - 0.05 * np.log1p(max(mean_cost, 0.0))
            - 0.1 * instability
            - 0.02 * complexity
        )
        rows.append(
            {
                "rule_name": rule_name,
                "protocol": protocol,
                "score": float(score),
                "mean_plasticity_delta_MI": mean_gain,
                "mean_plasticity_delta_accuracy": float(real["plasticity_delta_accuracy"].mean()) if not real.empty else 0.0,
                "mean_real_minus_degree_delta_MI": float(real_minus_degree_mi.mean()),
                "mean_real_minus_degree_accuracy": float(real_minus_degree_acc.mean()),
                "mean_update_cost_L1": mean_cost,
                "instability_penalty": instability,
                "rule_complexity": complexity,
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def baseline_family(rule_name: str) -> str:
    if rule_name in {"hebbian", "reward_hebbian", "ambiguity_gated_hebbian"}:
        return "hebbian"
    if rule_name in {"oja", "reward_oja", "ambiguity_gated_oja", "energy_normalized_ambiguity_oja"}:
        return "oja"
    if rule_name in {"bcm", "reward_bcm"}:
        return "bcm"
    return rule_name
