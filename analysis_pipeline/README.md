# Numerical pipeline guide

Run commands from the repository root. The runner also resolves its own location when called from another directory. Python dependencies are recorded in `requirements-lock.txt`; the original repository requirements remain necessary for the established package and its tests.

## Input preparation

For raw inputs:

```bash
python analysis_pipeline/prepare_inputs.py \
  --male-export /path/to/optic-lobe-v1.1-neuprint-tables \
  --female-dir /path/to/female_fruitfly
python analysis_pipeline/cross_specimen/scripts/acquire_annotations.py
```

The male preparation delegates to the established typed-core loader and exports matching raw body metadata. The female preparation projects `proofread_connections_783.feather` to source root, target root, neuropil and positive integer count, checking the endpoints against `proofread_root_ids_783.npy`. It does not load the much larger synapse-level export.

A local numerical archive can be used instead:

```bash
python analysis_pipeline/prepare_inputs.py --numeric-archive /path/to/archive
```

The importer recognizes `reproducibility/outputs/processed_typed_core/`, `reproducibility/pnas_extension/data_external/` and `cross_specimen_extension/data_external/`. These are input compatibility paths. Only explicitly listed numerical/metadata files are copied. Optional model arrays and archived diagnostic graph inputs are staged if present. Generated outcome tables are not imported by this command.

Annotation downloads are explicit, checksum-pinned and cached locally. A changed upstream snapshot produces an error instead of silently changing the input version. The correspondence workbook is extracted by its exact basename; other archive members are not unpacked. Runtime data and checksums are git-ignored.

## Stage order

```bash
python analysis_pipeline/run.py --stage core
python analysis_pipeline/run.py --stage controls
python analysis_pipeline/run.py --stage replication
python analysis_pipeline/run.py --stage references
python analysis_pipeline/run.py --stage coarsening
python analysis_pipeline/run.py --stage spatial
python analysis_pipeline/run.py --stage verify
python analysis_pipeline/run.py --stage plot
```

`--stage all` runs this sequence. `--dry-run` prints commands without requiring inputs or executing them. `--stage test` runs all repository tests, independently of the external data.

| Stage | Operations |
|---|---|
| `core` | Exact-ID metadata audit; contact/equal-body decompositions; regional, label and weak-edge controls; independent sparse calculation |
| `controls` | Exact-margin allocations, contact rarefaction, edge/body observation experiments and alternate ROI selections |
| `replication` | Published one-to-one type mapping, shared active source/target roles, native/matched universes, weighting and threshold controls |
| `references` | Independent fixed-margin draws and per-body fixed-contact draws in comparative universes |
| `coarsening` | Conservative fully resolved bipartite correspondence components |
| `spatial` | Coordinate coverage, conditional partition, buffered spatial test bands and within-bin exact-margin draws |
| `verify` | Identities, masses, role sets, paired summaries, reference-draw summaries and held-out loss aggregation |
| `plot` | Data-driven charts from local outputs; no LaTeX installation required |

Optional stages have separate input requirements:

```bash
python analysis_pipeline/run.py --stage model
python analysis_pipeline/run.py --stage archived
python analysis_pipeline/run.py --stage probes
```

`model` reads exported nonnegative model connectivity and its metadata. `archived` recalculates the archived edge-sample and fixed-step static diagnostic. `probes` performs the configured development/confirmation static source-template calibration with training-only parameter selection. These stages do not run as part of `all`. The static generator's historical `motion_direction` key names a source-template task; it is not a temporal motion simulation.

## Input schemas

- `outputs/processed_typed_core/edges.parquet`: `src`, `dst`, `weight`, `src_type`, `dst_type`, `edge_major_roi`, `src_major_roi`, `dst_major_roi`.
- `outputs/processed_typed_core/nodes.parquet`: `bodyId`, `type`, `instance`, plus fields retained by the established loader.
- `core/data_external/raw_typed_metadata.parquet`: neuPrint fields `bodyId:long`, `type:string`, `instance:string`, `status:string`, `statusLabel:string`, `assignedOlHex1:float`, `assignedOlHex2:float`, `somaLocation:point{srid:9157}`.
- `cross_specimen/data_external/female_proofread_connections.parquet`: exact integer `src`, `dst`, `weight`, and `neuropil`.
- Female type/category annotations and column coordinates: exact snapshots in `cross_specimen/annotation_sources.json`; matching uses the published correspondence workbook.

Source/body identifiers stay integers. Female body-pair weights are summed across neuropils before thresholding. Comparative endpoint selection removes self-edges. Source and target type roles are matched separately and iterated to stability. Ambiguous correspondences and annotation mismatches receive explicit exclusion records.

## Quantities and settings

For source type S, target body J and its type D, the core estimators compute A = H(S|D), C = H(S|J), and R = I(S;J|D), with A = C + R. Contact and equal-body measures use their respective recomputed priors. Coarse spatial groups X refine R into I(S;X|D) + I(S;J|D,X) on the same coordinate-bearing target subset.

`core/configs/` and `cross_specimen/config.json` preserve numerical parameter presets and random seeds. Fixed-margin draws retain source-type and body totals; within-bin draws additionally condition source totals on spatial groups. They leave the specified support/geometry unconstrained. Prediction uses fixed smoothing/shrinkage, contiguous held-out spatial bands and a training exclusion buffer. Its losses are predictive cross-entropies, separate from the exact information components. Coordinate bases are used within each dataset without a cross-specimen registration.

The numerical estimator in `core/src/decomposition.py` is shared byte-for-byte with the comparative implementation. Tests cover exact limiting cases, mass/entropy invariants, sparse/grouped agreement, role matching, exact-ID aggregation, spatial partitions and training-only prediction. Validation derives expected quantities from saved data and settings, without hard-coded biological outcome targets.

## Outputs and provenance

The following runtime directories are excluded from version control:

- `core/results/`, `core/audit/`, `core/provenance/`;
- `cross_specimen/results/`, `cross_specimen/provenance/`, all `data_external/` directories;
- `diagnostics/data/`, `figures/`, `runs/`.

Outputs retain per-type/body denominators, every reference/rarefaction draw, mapping exclusions, coordinate coverage, fold assignments and losses. The runner records executed commands, exit status and elapsed time in `runs/latest.json`; input staging records SHA-256 hashes. Re-running a stage overwrites its generated products, so use a separate checkout if preserving multiple runs.

An optional independent raw-export audit is available:

```bash
python analysis_pipeline/cross_specimen/scripts/audit_local_inputs.py --female-dir /path/to/female_fruitfly
```

It streams the supplied synapse Feather file, normalizes the unassigned-region label and reports exact pair/neuropil differences. This audit requires the original raw files and is separate from the standard numerical workflow. It never modifies them.

## Optional GitHub Actions setup

`ci/analysis-tests.yml.example` preserves the CI configuration as a non-executing template. Local testing requires no Actions permissions. To enable it later, copy it to `.github/workflows/analysis-tests.yml` and commit with credentials that permit workflow creation. The template runs the data-free tests and command-plan check; it does not download connectome inputs or publish generated results.
