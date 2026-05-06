# Final Rescue Control: Medulla to Lobula Plate Motion Direction

Budget: 1500 nodes / 12000 edges. Runs: 5 seeds x 5 subgraph samples. Conditions: real, degree-preserving null, weight-shuffled real topology, uniform-weight real topology. Modes: readout-only and local-Hebbian.

## Subgraph realization
|   subgraph_samples |   nodes_mean |   nodes_min |   nodes_max |   edges_mean |   edges_min |   edges_max | exploratory_any   |
|-------------------:|-------------:|------------:|------------:|-------------:|------------:|------------:|:------------------|
|                  5 |       1499.4 |        1498 |        1500 |        12000 |       12000 |       12000 | True              |

## Mode summary
| condition                     | mode          |   n |   accuracy_mean |   accuracy_sd |   delta_MI_mean |   delta_MI_sd |   cost_mean |    eta_mean |
|:------------------------------|:--------------|----:|----------------:|--------------:|----------------:|--------------:|------------:|------------:|
| degree_preserving_null        | local_hebbian |  25 |         0.62358 |      0.114524 |        1.01475  |      0.320954 |     99.6456 | 0.010187    |
| degree_preserving_null        | readout_only  |  25 |         0.626   |      0.113972 |        1.02166  |      0.318407 |      0      | 1.02166e+12 |
| real                          | local_hebbian |  25 |         0.64476 |      0.122649 |        1.06899  |      0.315401 |    106.649  | 0.01003     |
| real                          | readout_only  |  25 |         0.6473  |      0.120837 |        1.07532  |      0.311348 |      0      | 1.07532e+12 |
| weight_shuffled_real_topology | local_hebbian |  25 |         0.65676 |      0.104603 |        1.07858  |      0.343573 |    106.659  | 0.010118    |
| weight_shuffled_real_topology | readout_only  |  25 |         0.66008 |      0.103859 |        1.08935  |      0.341602 |      0      | 1.08935e+12 |
| weight_uniform_real_topology  | local_hebbian |  25 |         0.60696 |      0.134829 |        0.955932 |      0.393923 |    106.793  | 0.008957    |
| weight_uniform_real_topology  | readout_only  |  25 |         0.60704 |      0.134942 |        0.956005 |      0.394138 |      0      | 9.56005e+11 |

## Decisive control table
| condition_comparison                  |   delta_accuracy |   delta_MI |   delta_cost |    delta_eta |   delta_plasticity_MI | bootstrap_CI                                                                                      | supported?   |
|:--------------------------------------|-----------------:|-----------:|-------------:|-------------:|----------------------:|:--------------------------------------------------------------------------------------------------|:-------------|
| real_vs_degree_preserving_null        |          0.02118 |   0.054248 |     7.00293  | -0.000157    |            nan        | Delta acc [-0.00758, 0.0494]; Delta MI [-0.03536, 0.1513]; Delta eta [-0.001035, 0.0007885]       | no           |
| real_vs_weight_shuffled_real_topology |         -0.012   |  -0.009588 |    -0.010217 | -8.8e-05     |            nan        | Delta acc [-0.05512, 0.02846]; Delta MI [-0.1143, 0.09146]; Delta eta [-0.001087, 0.0009175]      | no           |
| real_vs_weight_uniform_real_topology  |          0.0378  |   0.113061 |    -0.144573 |  0.001073    |            nan        | Delta acc [0.0004595, 0.07246]; Delta MI [-0.01585, 0.2421]; Delta eta [-0.0001161, 0.002302]     | no           |
| local_hebbian_minus_readout_only_real |         -0.00254 |  -0.006325 |   106.649    | -1.07532e+12 |             -0.006325 | Delta acc [-0.00402, -0.00122]; Delta MI [-0.0104, -0.002698]; Delta eta [-1.194e+12, -9.515e+11] | no           |

## Interpretation
- Real does not clearly beat the degree-preserving null under the support criterion.
- Real does not clearly beat the uniform-weight fixed-topology control.
- Local-Hebbian does not exceed readout-only: describe this as motion separability under the graph representation, not a learning-efficiency improvement.

## Guardrail
This remains a synthetic-task, connectome-constrained proxy. It is not a measurement of real fly learning, synaptic biophysics, or thermodynamic heat.
