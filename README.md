# Typed connectome analysis workflows

Python tools for weighted cell-type channels, body-resolved input composition, sampling controls, matched-connectome comparison and spatial conditioning. This branch adds the `analysis_pipeline/` runner while retaining the existing package and scripts.

## Install and test

Python 3.12 is the tested environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r analysis_pipeline/requirements-lock.txt
python analysis_pipeline/run.py --stage test
```

The new test suite uses small synthetic fixtures and does not require external connectomes. Existing optional-dependency tests may skip when their dependency is unavailable.

## New pipeline

See [the pipeline guide](analysis_pipeline/README.md) for input setup, stage order, file schemas and numerical checks.

```bash
python analysis_pipeline/run.py --stage all --dry-run
python analysis_pipeline/prepare_inputs.py --male-export /path/to/optic-lobe-v1.1-neuprint-tables --female-dir /path/to/female_fruitfly
python analysis_pipeline/cross_specimen/scripts/acquire_annotations.py
python analysis_pipeline/run.py --stage all
```

Alternatively, `prepare_inputs.py --numeric-archive /path/to/local/numerical/archive` stages an explicit allowlist of numerical inputs from an existing local archive. It does not copy entire directories.

The default numerical workflow includes complete-input decomposition, weighting and sampling controls, type correspondence, count-matched and exact-margin references, spatially blocked prediction, conditional spatial references, validation and local diagnostic charts. Optional model and static-calibration stages are selected separately. Commands stop on failed input checks or numerical assertions and record their execution locally.

## Repository layout

| Path | Purpose |
|---|---|
| `analysis_pipeline/run.py` | Dependency-aware runner and dry-run command listing |
| `analysis_pipeline/prepare_inputs.py` | Local numerical input staging and schema checks |
| `analysis_pipeline/core/` | Body/type decomposition, reference and sampling algorithms |
| `analysis_pipeline/cross_specimen/` | Literal type matching, spatial partition and held-out prediction |
| `analysis_pipeline/diagnostics/` | Numerical helpers for optional static calibration |
| `analysis_pipeline/tests/` | Runner and input-staging tests |
| `thermodynamic_connectomics/`, `scripts/`, `tests/` | Established package, entry points and tests |
| `analysis_pipeline/ci/analysis-tests.yml.example` | CI template for tests without external datasets |

## Existing entry points

`bash run_optic_lobe_pipeline.sh` remains available for the established type-channel workflows. It reads the local neuPrint table export and writes processed inputs and analyses under `outputs/`. Its environment controls include `N_NULLS`, `ROI_PATHWAY_N_NULLS` and `KNOWN_PATHWAY_N_NULLS`. Existing tracked outputs and figures are retained as historical artifacts; the new workflow writes to separate, ignored runtime directories.

## Local artifacts

Raw datasets, generated results, local run logs, new figures and document/submission materials are excluded from this code update. The pipeline generates numerical outputs on the user's machine. Public input URLs and hashes identify the annotation snapshots required for reproducibility. See [data attribution](analysis_pipeline/DATA_SOURCES.md) for source releases and terms.
