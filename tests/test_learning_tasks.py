import numpy as np
import pandas as pd
import pytest
import subprocess
import sys
from pathlib import Path

from thermodynamic_connectomics.learning.subgraphs import learning_subgraph_from_edges
from thermodynamic_connectomics.learning.synthetic_tasks import generate_task_data


def _tiny_graph():
    edges = pd.DataFrame(
        {
            "src": [1, 2, 3, 4],
            "dst": [5, 5, 6, 6],
            "weight": [1.0, 2.0, 1.0, 3.0],
            "src_type": ["L1", "L2", "T4a", "Mi1"],
            "dst_type": ["Mi1", "Mi1", "T4a", "Tm1"],
        }
    )
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5, 6], "type": ["L1", "L2", "T4a", "Mi1", "Mi1", "Tm1"]})
    return learning_subgraph_from_edges("tiny", edges, nodes, max_nodes=10, max_edges=10)


def test_generate_task_data_shapes():
    graph = _tiny_graph()
    data = generate_task_data(graph, graph.node_table, "motion_direction", 32, 0.1, 1)
    assert data["X_source"].shape == (32, len(graph.source_nodes))
    assert data["y"].shape == (32,)
    assert np.isfinite(data["X_source"]).all()


def test_learning_smoke_cli_creates_expected_csvs(tmp_path):
    edges_path = Path("outputs/processed_typed_core/edges.parquet")
    nodes_path = Path("outputs/processed_typed_core/nodes.parquet")
    if not edges_path.exists() or not nodes_path.exists():
        pytest.skip("requires generated processed typed-core parquet files")
    out = tmp_path / "learn"
    cmd = [
        sys.executable,
        "scripts/09_physical_learning_efficiency.py",
        "--edges",
        str(edges_path),
        "--nodes",
        str(nodes_path),
        "--out-dir",
        str(out),
        "--pathways",
        "lamina_to_medulla",
        "--tasks",
        "polarity_contrast",
        "--conditions",
        "real",
        "--n-seeds",
        "1",
        "--n-samples",
        "80",
        "--epochs",
        "1",
        "--max-nodes",
        "80",
        "--max-edges",
        "400",
        "--matched-nodes",
        "80",
        "--matched-edges",
        "400",
        "--n-subgraph-samples",
        "1",
    ]
    subprocess.run(cmd, check=True)
    assert (out / "learning_curves.csv").exists()
    assert (out / "final_summary.csv").exists()
    final = pd.read_csv(out / "final_summary.csv")
    assert "sample_id" in final.columns
    assert (out / "paired_comparisons.csv").exists()
    assert (out / "matched_decisive_table.csv").exists()
