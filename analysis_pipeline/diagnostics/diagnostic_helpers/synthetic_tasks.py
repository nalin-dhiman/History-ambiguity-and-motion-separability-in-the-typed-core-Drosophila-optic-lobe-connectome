"""Synthetic visual tasks for pathway-specific learning probes."""
from __future__ import annotations

import re

import numpy as np

LAMINA_RE = re.compile(r"^L[1-5](?:$|[^0-9])")
MEDULLA_RE = re.compile(r"^(?:Mi|Tm|TmY)")
MOTION_RE = re.compile(r"^(?:T4|T5)")
OUTPUT_RE = re.compile(r"^(?:LC|LPLC|LT|DN)")


def _mask(types: list[str], regex: re.Pattern[str]) -> np.ndarray:
    return np.array([bool(regex.search(str(t))) for t in types], dtype=bool)


def _choose_template_groups(types: list[str], task_name: str, n_classes: int) -> list[np.ndarray]:
    types_arr = np.asarray(types, dtype=str)
    lamina = _mask(types, LAMINA_RE)
    medulla = _mask(types, MEDULLA_RE)
    motion = _mask(types, MOTION_RE)
    output = _mask(types, OUTPUT_RE)
    groups = []
    if task_name == "polarity_contrast":
        groups = [lamina | medulla, lamina | motion | output]
    elif task_name == "motion_direction":
        t4 = np.char.startswith(types_arr, "T4")
        t5 = np.char.startswith(types_arr, "T5")
        groups = [t4, t5, motion | medulla, motion | output]
    elif task_name == "global_pattern":
        groups = [lamina | medulla, medulla | motion, motion | output, output | lamina]
    while len(groups) < n_classes:
        groups.append(np.ones(len(types), dtype=bool))
    return groups[:n_classes]


def generate_task_data(graph, node_table, task_name: str, n_samples: int, noise: float, seed: int) -> dict:
    """Generate source-node activities and integer labels for a synthetic task."""

    rng = np.random.default_rng(seed)
    n_source = len(graph.source_nodes)
    if n_source == 0:
        raise ValueError("graph has no source nodes")
    if task_name == "polarity_contrast":
        n_classes = 2
    elif task_name in {"motion_direction", "global_pattern"}:
        n_classes = 4
    else:
        raise ValueError(f"Unknown task: {task_name}")
    y = rng.integers(0, n_classes, size=n_samples)
    X = rng.normal(0.0, noise, size=(n_samples, n_source)).astype(np.float32)
    templates = _choose_template_groups(graph.source_types, task_name, n_classes)
    for cls in range(n_classes):
        idx = y == cls
        active = templates[cls]
        if not active.any():
            active = np.ones(n_source, dtype=bool)
        X[np.ix_(idx, active)] += 1.0
        if task_name == "polarity_contrast" and cls == 0:
            X[np.ix_(idx, active)] -= 0.5
        if task_name == "global_pattern":
            alt = templates[(cls + 1) % n_classes]
            X[np.ix_(idx, alt)] += 0.35
    X = 1.0 / (1.0 + np.exp(-X))
    return {
        "X_source": X.astype(np.float32),
        "y": y.astype(np.int64),
        "source_nodes": graph.source_nodes,
        "target_nodes": graph.target_nodes,
        "task_metadata": {"task_name": task_name, "n_classes": n_classes, "noise": noise, "seed": seed},
    }
