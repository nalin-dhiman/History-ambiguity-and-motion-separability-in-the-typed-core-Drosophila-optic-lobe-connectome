"""Visualization helpers for thermodynamic connectomics."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def scatter_heatmap(
    metrics: pd.DataFrame,
    metadata: pd.DataFrame,
    output_png: str | Path,
    node_col_metrics: str = "post_id",
    node_col_meta: str = "root_id",
    value_col: str = "history_loss_bits_local",
    x_col: str = "x",
    y_col: str = "y",
    title: str = "Local incoming-history ambiguity",
) -> None:
    """Save a simple 2D scatter heatmap of node metrics."""
    df = metrics.merge(metadata, left_on=node_col_metrics, right_on=node_col_meta, how="left")
    df = df.dropna(subset=[x_col, y_col, value_col])
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(df[x_col], df[y_col], c=df[value_col], s=18, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.colorbar(sc, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
