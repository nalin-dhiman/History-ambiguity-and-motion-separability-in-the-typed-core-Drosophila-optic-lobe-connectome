# Learning-Signal Rescue: Medulla to Lobula Plate Motion Direction

Focused rescue run only: `medulla_to_lobula_plate` / `motion_direction`, real vs degree-preserving null.
Chance accuracy is 0.25 for the four-class motion task. Values aggregate 3 seeds x 3 subgraph samples.

## Budget realization
| budget       |   nodes_mean |   nodes_min |   edges_mean |   edges_min | exploratory_any   |
|:-------------|-------------:|------------:|-------------:|------------:|:------------------|
| 1000n_8000e  |      997.667 |         997 |         8000 |        8000 | True              |
| 1500n_12000e |     1499     |        1498 |        12000 |       12000 | True              |
| 500n_3000e   |      481     |         480 |         3000 |        3000 | True              |

All budgets reached the requested edge count. Node counts were close but slightly below target, so the current pipeline flags them as exploratory. Real and null are still paired within each exact sampled subgraph.

## Rescue summary
| budget       | mode          |   mean_real_accuracy |   real_accuracy_ci_low |   real_accuracy_ci_high |   runs_real_accuracy_above_chance |   mean_real_delta_MI |   real_delta_MI_ci_low |   real_delta_MI_ci_high |   runs_real_delta_MI_positive |   mean_delta_accuracy_vs_degree_null |   delta_accuracy_ci_low |   delta_accuracy_ci_high |   runs_real_beats_null_accuracy |   mean_delta_MI_vs_degree_null |   delta_MI_ci_low |   delta_MI_ci_high |   runs_real_beats_null_MI |   conditional_eta_gain |   conditional_eta_ci_low |   conditional_eta_ci_high | any_samples_exploratory   |
|:-------------|:--------------|---------------------:|-----------------------:|------------------------:|----------------------------------:|---------------------:|-----------------------:|------------------------:|------------------------------:|-------------------------------------:|------------------------:|-------------------------:|--------------------------------:|-------------------------------:|------------------:|-------------------:|--------------------------:|-----------------------:|-------------------------:|--------------------------:|:--------------------------|
| 500n_3000e   | local_hebbian |             0.258222 |               0.256389 |                0.260056 |                                 9 |            -0.169668 |              -0.388435 |                2e-06    |                             2 |                            -5.6e-05  |               -0.000167 |                 0        |                               0 |                      -0.04356  |         -0.161575 |           0.022984 |                         4 |           -0.010487    |             -0.039207    |               0.005742    | True                      |
| 500n_3000e   | readout_only  |             0.258222 |               0.256389 |                0.260111 |                                 9 |            -0.169711 |              -0.391172 |               -0.000393 |                             2 |                            -5.6e-05  |               -0.000167 |                 0        |                               0 |                      -0.044684 |         -0.163079 |           0.021967 |                         4 |           -4.4684e+10  |             -1.63618e+11 |               2.25319e+10 | True                      |
| 1000n_8000e  | local_hebbian |             0.283444 |               0.256944 |                0.332444 |                                 9 |             0.176572 |               0.021456 |                0.341882 |                             4 |                             0.006944 |               -0.043669 |                 0.066111 |                               4 |                       0.177417 |          0.000395 |           0.341939 |                         7 |            0.056993    |              0.009781    |               0.106975    | True                      |
| 1000n_8000e  | readout_only  |             0.283833 |               0.257167 |                0.333333 |                                 9 |             0.180879 |               0.020126 |                0.344997 |                             4 |                             0.007111 |               -0.04356  |                 0.066722 |                               4 |                       0.180499 |          0.005865 |           0.359014 |                         7 |            1.80499e+11 |              5.63023e+09 |               3.51943e+11 | True                      |
| 1500n_12000e | local_hebbian |             0.649278 |               0.582108 |                0.714558 |                                 9 |             1.11338  |               0.955387 |                1.27333  |                             9 |                             0.036667 |                0.002276 |                 0.077114 |                               6 |                       0.175768 |          0.05364  |           0.29823  |                         7 |            0.025187    |              0.008481    |               0.040956    | True                      |
| 1500n_12000e | readout_only  |             0.6515   |               0.585886 |                0.714389 |                                 9 |             1.11982  |               0.968888 |                1.28194  |                             9 |                             0.038556 |                0.001276 |                 0.083112 |                               6 |                       0.179016 |          0.050493 |           0.31322  |                         7 |            1.79016e+11 |              4.95243e+10 |               3.13905e+11 | True                      |

## Answers
- Real accuracy rose above chance with positive delta MI starting at `1000n_8000e` for local-Hebbian mode: accuracy 0.283, delta MI 0.177 bits.
- Real beats the degree-null most conservatively at `1500n_12000e`: delta accuracy 0.037 with CI [0.002, 0.077], delta MI 0.176 with CI [0.054, 0.298].
- `500n_3000e` local-Hebbian stability: accuracy above chance 9/9, positive delta MI 2/9, real beats null accuracy 0/9, real beats null MI 4/9.
- `1000n_8000e` local-Hebbian stability: accuracy above chance 9/9, positive delta MI 4/9, real beats null accuracy 4/9, real beats null MI 7/9.
- `1500n_12000e` local-Hebbian stability: accuracy above chance 9/9, positive delta MI 9/9, real beats null accuracy 6/9, real beats null MI 7/9.

## Smallest budget to use
Use `1500n_12000e` as the smallest budget with clear real-vs-degree-null support.

## Guardrail
This is a learning-signal rescue for a synthetic motion task under a simplified local-update proxy. It is not biological learning and not a thermodynamic heat measurement.
