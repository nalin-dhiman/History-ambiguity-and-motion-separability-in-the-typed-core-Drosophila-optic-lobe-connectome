import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermodynamic_connectomics.learning.history_gates import gates_for_graph
from thermodynamic_connectomics.learning.rule_search import evaluate_rule, symbolic_rule_specs
from thermodynamic_connectomics.learning.rules import make_rule, rule_formula
from thermodynamic_connectomics.learning.subgraphs import learning_subgraph_from_edges
from thermodynamic_connectomics.learning.synthetic_tasks import generate_task_data


def _tiny_graph():
    edges = pd.DataFrame(
        {
            "src": [1, 2, 3, 4, 1, 3],
            "dst": [5, 5, 6, 6, 6, 5],
            "weight": [1.0, 2.0, 1.0, 3.0, 1.0, 1.0],
            "src_type": ["Mi1", "Tm1", "T4a", "T5b", "Mi1", "T4a"],
            "dst_type": ["T4a", "T4a", "T5b", "T5b", "T5b", "T4a"],
        }
    )
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5, 6], "type": ["Mi1", "Tm1", "T4a", "T5b", "T4a", "T5b"]})
    return learning_subgraph_from_edges("tiny", edges, nodes, max_nodes=10, max_edges=10)


def test_rules_preserve_nonnegative_normalized_weights():
    graph = _tiny_graph()
    gates = gates_for_graph(graph)
    rule = make_rule("ambiguity_gated_oja")
    rule.reset(graph, seed=1)
    W, l1, _ = rule.update(
        graph.W,
        pre=np.ones(len(graph.source_nodes), dtype=np.float32),
        post=np.ones(len(graph.target_nodes), dtype=np.float32),
        reward=1.0,
        gates=gates,
        lr=0.01,
        decay=0.001,
    )
    assert np.all(W.data >= 0)
    col_sums = np.asarray(W.sum(axis=0)).ravel()
    assert np.allclose(col_sums[col_sums > 0], 1.0)
    assert np.isfinite(l1)


def test_symbolic_rule_parser_and_representation_probe():
    graph = _tiny_graph()
    spec = symbolic_rule_specs(1)[0]
    rule = make_rule(spec["name"], active_basis=spec["active_basis"])
    assert "delta_w" in rule_formula({"name": rule.name, "active_basis": rule.active_basis})
    task = generate_task_data(graph, graph.node_table, "motion_direction", 50, 0.1, 2)
    out = evaluate_rule(graph, task, rule, "representation_probe", seed=3, epochs=1, readout_lr=0.01, plasticity_lr=0.001, decay=0.0001, probe_epochs=3)
    assert {"before_task_MI_bits", "final_task_MI_bits", "plasticity_delta_MI", "eta_plastic"}.issubset(out)
    assert np.isfinite(list(out.values())).all()


def test_rule_search_smoke_cli_creates_expected_files(tmp_path):
    edges_path = Path("outputs/processed_typed_core/edges.parquet")
    nodes_path = Path("outputs/processed_typed_core/nodes.parquet")
    if not edges_path.exists() or not nodes_path.exists():
        pytest.skip("requires generated processed typed-core parquet files")
    out = tmp_path / "rule_smoke"
    cmd = [
        sys.executable,
        "scripts/12_rule_search_learning_algorithm.py",
        "--edges",
        str(edges_path),
        "--nodes",
        str(nodes_path),
        "--out-dir",
        str(out),
        "--matched-nodes",
        "80",
        "--matched-edges",
        "300",
        "--n-samples",
        "80",
        "--epochs",
        "1",
        "--discovery-seeds",
        "0",
        "--discovery-subgraph-samples",
        "0",
        "--conditions",
        "real",
        "--rules",
        "no_plasticity",
        "ambiguity_gated_hebbian",
        "--protocols",
        "representation_probe",
    ]
    subprocess.run(cmd, check=True)
    assert (out / "history_gates.csv").exists()
    assert (out / "rule_search_results.csv").exists()
    assert (out / "best_candidate_rule.json").exists()
