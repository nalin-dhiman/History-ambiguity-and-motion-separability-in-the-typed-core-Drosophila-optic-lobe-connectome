"""Local plasticity rules for connectome-constrained rule discovery."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from thermodynamic_connectomics.learning.local_rules import normalize_per_target

EPS = 1e-12


@dataclass
class LocalRule:
    name: str
    active_basis: dict[str, float] | None = None
    state: dict[str, np.ndarray] = field(default_factory=dict)

    def reset(self, graph_state, seed: int = 0) -> None:
        self.state = {}
        if self.name in {"bcm", "reward_bcm"}:
            self.state["theta"] = np.zeros(graph_state.W.shape[1], dtype=np.float32)

    def update(
        self,
        W: sparse.csr_matrix,
        pre: np.ndarray,
        post: np.ndarray,
        reward: float,
        gates: dict[str, np.ndarray],
        lr: float,
        decay: float,
        **kwargs,
    ) -> tuple[sparse.csr_matrix, float, float]:
        if self.name == "no_plasticity":
            return W, 0.0, 0.0
        coo = W.tocoo(copy=True)
        if coo.data.size == 0:
            return W, 0.0, 0.0
        row = coo.row
        col = coo.col
        x = pre[row]
        y = post[col]
        w = coo.data
        S = gates.get("S_inverse_ambiguity", np.ones(W.shape[1], dtype=np.float32))[col]
        B = np.maximum(gates.get("target_update_budget", np.ones(W.shape[1], dtype=np.float32))[col], EPS)

        if self.active_basis is not None:
            plastic = np.zeros_like(w, dtype=np.float64)
            basis = _basis_values(x, y, w, reward, S, B)
            for key, coef in self.active_basis.items():
                plastic += float(coef) * basis[key]
            delta = lr * plastic - decay * w
        elif self.name == "hebbian":
            delta = lr * x * y - decay * w
        elif self.name == "reward_hebbian":
            delta = lr * reward * x * y - decay * w
        elif self.name == "oja":
            delta = lr * y * (x - y * w) - decay * w
        elif self.name == "reward_oja":
            delta = lr * reward * y * (x - y * w) - decay * w
        elif self.name == "bcm":
            theta = _bcm_theta(self, post)
            delta = lr * x * y * (y - theta[col]) - decay * w
        elif self.name == "reward_bcm":
            theta = _bcm_theta(self, post)
            delta = lr * reward * x * y * (y - theta[col]) - decay * w
        elif self.name == "ambiguity_gated_hebbian":
            delta = lr * reward * x * y * S - decay * w
        elif self.name == "ambiguity_gated_oja":
            delta = lr * reward * S * y * (x - y * w) - decay * w
        elif self.name == "energy_normalized_ambiguity_oja":
            delta = lr * reward * S * y * (x - y * w) / (EPS + B) - decay * w
        else:
            raise ValueError(f"Unknown rule: {self.name}")

        delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        new_data = np.clip(w + delta, 0.0, None)
        out = sparse.coo_matrix((new_data, (row, col)), shape=W.shape).tocsr()
        out = normalize_per_target(out)
        return out, float(np.abs(delta).sum()), float(np.sqrt(np.square(delta).sum()))


def _bcm_theta(rule: LocalRule, post: np.ndarray, theta_lr: float = 0.05) -> np.ndarray:
    theta = rule.state.get("theta")
    if theta is None or len(theta) != len(post):
        theta = np.zeros(len(post), dtype=np.float32)
    theta = (1.0 - theta_lr) * theta + theta_lr * np.square(post)
    rule.state["theta"] = theta.astype(np.float32)
    return rule.state["theta"]


def _basis_values(x: np.ndarray, y: np.ndarray, w: np.ndarray, reward: float, S: np.ndarray, B: np.ndarray) -> dict[str, np.ndarray]:
    xy = x * y
    oja = y * (x - y * w)
    return {
        "b1": xy,
        "b2": reward * xy,
        "b3": oja,
        "b4": reward * oja,
        "b5": xy * S,
        "b6": reward * xy * S,
        "b7": reward * oja * S,
        "b8": reward * xy / (EPS + B),
        "b9": reward * xy * S / (EPS + B),
        "b10": -w,
    }


BASELINE_RULES = [
    "no_plasticity",
    "hebbian",
    "reward_hebbian",
    "oja",
    "reward_oja",
    "bcm",
    "reward_bcm",
    "ambiguity_gated_hebbian",
    "ambiguity_gated_oja",
    "energy_normalized_ambiguity_oja",
]


def make_rule(name: str, active_basis: dict[str, float] | None = None) -> LocalRule:
    if name.startswith("symbolic:"):
        return LocalRule(name=name, active_basis=active_basis or {})
    if active_basis is not None:
        return LocalRule(name=name, active_basis=active_basis)
    if name not in BASELINE_RULES:
        raise ValueError(f"Unknown rule: {name}")
    return LocalRule(name=name)


def serialize_rule(rule: LocalRule) -> dict:
    return {"name": rule.name, "active_basis": rule.active_basis or {}}


def rule_formula(rule_spec: dict) -> str:
    basis = rule_spec.get("active_basis") or {}
    if not basis:
        return rule_spec.get("name", "")
    terms = [f"{coef:+g}*{name}" for name, coef in sorted(basis.items())]
    return "delta_w = lr * (" + " ".join(terms) + ") - decay*w"
