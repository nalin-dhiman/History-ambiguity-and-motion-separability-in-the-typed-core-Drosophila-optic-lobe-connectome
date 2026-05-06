# A cell-type channel architecture for history ambiguity and embedded motion separability

Clean analysis repository for the typed-core Drosophila optic-lobe connectome study. The code treats the typed connectome as a weighted source-to-target cell-type channel, computes history ambiguity \(H(S\mid D)\), evaluates null models and taxonomy/ROI sensitivity, and runs synthetic motion-separability and plasticity-rule controls.

This repository intentionally does **not** include the manuscript LaTeX, supplementary LaTeX, submission-package zips, LaTeX build files, or heavyweight processed graph exports. It does include the analysis pipeline, tests, compact result tables, and publication-ready PDF figures.

## Repository Layout

```text
thermodynamic_connectomics/   Core Python package
scripts/                      Pipeline, controls, and figure-generation scripts
tests/                        Unit tests for entropy, nulls, ROI parsing, and learning controls
outputs/tables/               Compact final CSV summary tables
outputs/*/                    Selected lightweight derived result tables
figures/main/                 Main publication figures as PDFs
figures/supplementary/        Supplementary figures as PDFs
run_optic_lobe_pipeline.sh    End-to-end core analysis runner
requirements.txt              Python dependency list
environment.yml               Conda environment
```

## What Is Excluded

The raw neuPrint export and generated processed edge/node tables are not committed. They are large and should be regenerated locally from the source optic-lobe v1.1 neuPrint tables.

Excluded generated/heavy artifacts include:

- `outputs/processed*/edges.parquet`
- `outputs/processed*/nodes.parquet`
- large node-level metric CSVs
- manuscript and supplement `.tex` files
- LaTeX auxiliary/build files
- submission zips
- Python caches and test caches

The included compact CSVs and PDFs are sufficient to inspect the reported results and figures without rebuilding the full graph pipeline.

## Install

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Using `conda`:

```bash
conda env create -f environment.yml
conda activate thermodynamic-connectomics
```

## Data Input

Place the optic-lobe v1.1 neuPrint table export at:

```text
optic-lobe-v1.1-neuprint-tables/
```

The preparation script expects the local export schema used by the analysis pipeline. The repository does not redistribute the raw connectome export.

## Run the Core Pipeline

```bash
bash run_optic_lobe_pipeline.sh
```

The runner performs:

1. full and typed-core graph preparation;
2. weighted history-ambiguity and mutual-information metrics;
3. degree-preserving/type-label null summaries;
4. ROI and visual-pathway analyses;
5. known-pathway validation;
6. core diagnostic figures and report tables.

Null sizes can be adjusted for local compute:

```bash
N_NULLS=20 ROI_PATHWAY_N_NULLS=10 KNOWN_PATHWAY_N_NULLS=10 bash run_optic_lobe_pipeline.sh
```

## Reviewer Controls and Final Figures

After the core pipeline has produced `outputs/processed_typed_core/`, run:

```bash
python scripts/18_reviewer_sensitivity_controls.py
python scripts/19_reviewer_revision_controls.py
python scripts/22_make_nature_level_figures.py
python scripts/23_make_connectome_backbone_network.py
python scripts/24_make_supplement_anatomical_projection.py
```

These scripts regenerate the final reviewer-facing tables and PDF figures, including:

- pathway and ROI history-ambiguity hierarchy figures;
- null-model and taxonomy/ROI sensitivity figures;
- synthetic motion separability controls;
- plasticity-rule negative-control figures;
- neuron-level connectome backbone visualization;
- anatomical-style ROI projection of optic-lobe fan-in to descending targets.

## Tests

```bash
pytest
```

The tests cover entropy/channel calculations, small-graph null behavior, ROI parsing, learning metrics, synthetic task helpers, local rules, and rule-search utilities.

## Scientific Scope

The supported claims are intentionally conservative:

- \(H(S\mid D)\) is a weighted anatomical cell-type channel statistic.
- The optic-lobe typed-core graph shows a structured hierarchy of source-history specificity and pooling.
- The synthetic medulla-to-lobula-plate motion probe supports representation separability under defined controls.
- The tested local plasticity rules do not improve representation beyond readout-only controls.

The repository does **not** claim measured thermodynamic cost, heat dissipation, biological channel capacity, or discovery of a general biological learning rule.

## Key Outputs

Useful entry points:

- `outputs/tables/final_master_claim_table.csv`
- `outputs/tables/authoritative_channel_metrics_with_ci.csv`
- `outputs/tables/pathway_comparison_table.csv`
- `outputs/tables/motion_probe_reviewer_control_summary.csv`
- `figures/main/fig6_connectome_backbone_network.pdf`
- `figures/supplementary/sfig19_anatomical_projection_target_fanin.pdf`

## Reproducibility Note

The PDF figures committed here were generated from the local processed typed-core graph and compact result tables. Figures that look anatomically spatial but are based on graph/ROI placement are labeled as projections, not physical soma/skeleton reconstructions.
