#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/logs

python scripts/01_prepare_optic_lobe.py \
  --data-dir optic-lobe-v1.1-neuprint-tables \
  --out-dir outputs/processed \
  2>&1 | tee outputs/logs/01_prepare.log

python scripts/01_prepare_optic_lobe.py \
  --data-dir optic-lobe-v1.1-neuprint-tables \
  --typed-core \
  --out-dir outputs/processed_typed_core \
  2>&1 | tee outputs/logs/01_prepare_typed_core.log

python scripts/02_compute_irreversibility.py \
  --edges outputs/processed/edges.parquet \
  --nodes outputs/processed/nodes.parquet \
  --out-dir outputs/metrics \
  --group-col type \
  --min-weight 1 \
  --min-degree 3 \
  2>&1 | tee outputs/logs/02_metrics.log

python scripts/02_compute_irreversibility.py \
  --edges outputs/processed_typed_core/edges.parquet \
  --nodes outputs/processed_typed_core/nodes.parquet \
  --out-dir outputs/metrics_typed_core \
  --group-col type \
  --min-weight 1 \
  --min-degree 3 \
  2>&1 | tee outputs/logs/02_metrics_typed_core.log

python scripts/03_run_nulls.py \
  --edges outputs/processed_typed_core/edges.parquet \
  --node-metrics outputs/metrics_typed_core/node_irreversibility.csv \
  --out-dir outputs/nulls_typed_core \
  --n-nulls "${N_NULLS:-100}" \
  --seed 42 \
  --group-col type \
  --max-edges "${NULL_MAX_EDGES:-500000}" \
  2>&1 | tee outputs/logs/03_nulls.log

python scripts/07_compare_variants.py \
  --full-processed-dir outputs/processed \
  --full-metrics-dir outputs/metrics \
  --typed-processed-dir outputs/processed_typed_core \
  --typed-metrics-dir outputs/metrics_typed_core \
  --out outputs/tables/full_vs_typed_core_comparison.csv \
  2>&1 | tee outputs/logs/07_compare_variants.log

python scripts/08_roi_pathway_analysis.py \
  --edges outputs/processed_typed_core/edges.parquet \
  --out-dir outputs/roi_pathway \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --n-nulls "${ROI_PATHWAY_N_NULLS:-30}" \
  --sample-edges "${ROI_PATHWAY_NULL_SAMPLE_EDGES:-1000000}" \
  2>&1 | tee outputs/logs/08_roi_pathway_analysis.log

python scripts/09_known_visual_pathway_validation.py \
  --edges outputs/processed_typed_core/edges.parquet \
  --out-dir outputs/known_visual_pathway_validation \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --n-nulls "${KNOWN_PATHWAY_N_NULLS:-20}" \
  --sample-edges "${KNOWN_PATHWAY_NULL_SAMPLE_EDGES:-1000000}" \
  2>&1 | tee outputs/logs/09_known_visual_pathway_validation.log

python scripts/04_make_figures.py \
  --processed-dir outputs/processed_typed_core \
  --metrics-dir outputs/metrics_typed_core \
  --nulls-dir outputs/nulls_typed_core \
  --out-dir outputs/figures \
  2>&1 | tee outputs/logs/04_figures.log

python scripts/05_write_report.py \
  --processed-dir outputs/processed_typed_core \
  --metrics-dir outputs/metrics_typed_core \
  --nulls-dir outputs/nulls_typed_core \
  --figures-dir outputs/figures \
  --comparison-table outputs/tables/full_vs_typed_core_comparison.csv \
  --roi-table outputs/tables/roi_comparison_table.csv \
  --pathway-table outputs/tables/pathway_comparison_table.csv \
  --roi-nulls outputs/roi_pathway/roi_pathway_null_summary.csv \
  --known-visual-validation outputs/known_visual_pathway_validation/known_visual_pathway_validation.csv \
  --known-visual-interpretation outputs/tables/known_visual_pathway_interpretation.csv \
  --out outputs/report.md \
  2>&1 | tee outputs/logs/05_report.log
