#!/usr/bin/env python
"""Prepare canonical optic-lobe graph files from neuPrint exports."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermodynamic_connectomics.io.optic_lobe_loader import prepare_optic_lobe_tables, write_schema_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="optic-lobe-v1.1-neuprint-tables")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--typed-core", action="store_true", help="Keep only typed body IDs and edges among them.")
    parser.add_argument("--min-weight", type=float, default=1.0)
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--sample-nodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out_dir = args.out_dir or ("outputs/processed_typed_core" if args.typed_core else "outputs/processed")

    schema_path = Path(out_dir).parent / "schema_report.txt"
    write_schema_report(args.data_dir, schema_path)
    print(f"Wrote schema report to {schema_path}")
    prepare_optic_lobe_tables(
        args.data_dir,
        out_dir,
        min_weight=args.min_weight,
        max_edges=args.max_edges,
        sample_nodes=args.sample_nodes,
        seed=args.seed,
        typed_core=args.typed_core,
    )


if __name__ == "__main__":
    main()
