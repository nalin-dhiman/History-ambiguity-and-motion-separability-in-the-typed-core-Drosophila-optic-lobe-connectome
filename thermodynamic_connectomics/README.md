# Thermodynamic Connectomics / ThermoFly

This repository implements a conservative, testable plan for using FlyWire-style connectome data to study Landauer-style logical irreversibility.

**Core rule:** the code computes *lower-bound proxies* and information-theoretic irreversibility metrics. It does **not** claim actual biological heat dissipation unless you add a validated metabolic/biophysical model.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py --out demo_outputs
```

Optional PyTorch smoke test:

```bash
python scripts/run_demo.py --out demo_outputs --train-snn
```

## Expected FlyWire-style edge table

At minimum, provide a weighted directed edge table with columns equivalent to:

| column | meaning |
|---|---|
| `pre_id` or `pre_pt_root_id` | presynaptic neuron/root ID |
| `post_id` or `post_pt_root_id` | postsynaptic neuron/root ID |
| `weight`, `syn_count`, `Count` | synapse count or edge weight |

Example:

```csv
pre_id,post_id,weight,neuropil
720575940...,720575941...,5,ME_L
```

The loader also accepts Parquet and Feather when pandas has the needed backends installed.

## Compute a structural heat map

```bash
python scripts/compute_heatmap.py \
  --edges flywire_edges.csv \
  --metadata flywire_neuron_metadata.csv \
  --out heatmap_outputs
```

The metadata file is optional. If provided, it should include `root_id`, `x`, and `y` columns for 2-D plotting.

## Compare against degree-preserving nulls

```bash
python scripts/compare_nulls.py \
  --edges flywire_edges.csv \
  --n-nulls 50 \
  --out null_outputs
```

For full FlyWire-scale graphs, start with a selected optic-lobe subgraph or a cell-type/neuropil coarse-grained graph. The pure-Python rewire function is meant for prototypes and reviewer-facing logic, not massive production rewiring.

## Main metrics

- `history_loss_bits_local`: entropy of incoming sources for each postsynaptic neuron, `H(pre | post=j)`.
- `history_loss_bits_weighted`: local history loss weighted by the fraction of total incoming synaptic mass landing on that postsynaptic neuron.
- `landauer_relevant_bits`: `max(0, H(source) - H(target))` for a channel `P(target | source)`.
- `history_loss_bits`: `H(source | target)`, a non-invertibility measure. For deterministic many-to-one maps this equals entropy contraction; for noisy channels it is not automatically a heat bound.

## Code map

- `thermofly/landauer.py`: conversions between erased bits and `k_B T ln 2` joules.
- `thermofly/entropy.py`: entropy, mutual information, and channel metrics.
- `thermofly/data.py`: table loading, edge aggregation, sparse adjacency.
- `thermofly/graph_metrics.py`: node and channel irreversibility metrics.
- `thermofly/nulls.py`: degree-preserving edge-swap nulls and weight shuffles.
- `thermofly/activity.py`: activity-conditioned irreversibility estimation.
- `thermofly/snn.py`: minimal connectome-masked LIF network and differentiable thermodynamic proxy loss.

## Suggested reviewer-facing claim

A safe claim is: "We introduce an information-theoretic Landauer atlas of FlyWire circuits and test whether biological wiring concentrates entropy contraction differently from strict null connectomes."

An unsafe claim is: "We measured the fly brain's actual heat dissipation from the connectome." The current code does not support that.
