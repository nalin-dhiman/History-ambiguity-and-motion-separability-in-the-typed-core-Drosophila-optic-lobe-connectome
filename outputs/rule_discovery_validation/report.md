# Rule Discovery: History-Ambiguity-Gated Plasticity

## 1. Motivation
The anatomical hierarchy showed structured history ambiguity. The rescue run showed a medulla-to-lobula-plate motion signal. Here we ask whether a local rule can exploit this better than standard local baselines and readout-only.

## 2. Candidate Rules
The search includes no-plasticity/readout-only, Hebbian, reward-Hebbian, Oja, reward-Oja, BCM, reward-BCM, ambiguity-gated Hebbian, ambiguity-gated Oja, and energy-normalized ambiguity-gated Oja. Symbolic rules are sparse combinations of local basis terms when requested.

## 3. Search Method
Discovery scores reward positive plasticity delta MI, real-minus-degree-null gains, and accuracy gains, while penalizing update cost, instability, and rule complexity. Eta is not rewarded when delta MI is non-positive.

Best discovery candidate:

```json
{
  "name": "no_plasticity",
  "active_basis": {},
  "score": 1.0137918784359907,
  "formula": "no_plasticity"
}
```

Formula:

`no_plasticity`

Top discovery rows:

| rule_name                       | protocol       |    score |   mean_plasticity_delta_MI |   mean_plasticity_delta_accuracy |   mean_real_minus_degree_delta_MI |   mean_real_minus_degree_accuracy |   mean_update_cost_L1 |   instability_penalty |   rule_complexity |
|:--------------------------------|:---------------|---------:|---------------------------:|---------------------------------:|----------------------------------:|----------------------------------:|----------------------:|----------------------:|------------------:|
| no_plasticity                   | online_readout | 1.01379  |                   0.92757  |                         0.320476 |                          0.260768 |                        -0.0921429 |               0       |              0.257332 |                 0 |
| energy_normalized_ambiguity_oja | online_readout | 0.954559 |                   0.92757  |                         0.320476 |                          0.260768 |                        -0.0921429 |               1.19167 |              0.257332 |                 1 |
| symbolic:0.1b1                  | online_readout | 0.852132 |                   0.92757  |                         0.320476 |                          0.259277 |                        -0.0921429 |              10.2266  |              0.257332 |                 2 |
| symbolic:0.1b3                  | online_readout | 0.851531 |                   0.92757  |                         0.320476 |                          0.259277 |                        -0.0921429 |              10.3625  |              0.257332 |                 2 |
| symbolic:0.1b2                  | online_readout | 0.851419 |                   0.92757  |                         0.320476 |                          0.259277 |                        -0.0921429 |              10.3878  |              0.257332 |                 2 |
| symbolic:-0.1b3                 | online_readout | 0.847914 |                   0.92757  |                         0.320476 |                          0.262028 |                        -0.0921429 |              11.5554  |              0.257332 |                 2 |
| symbolic:-0.1b2                 | online_readout | 0.846157 |                   0.92757  |                         0.320476 |                          0.262028 |                        -0.0921429 |              12.0044  |              0.257332 |                 2 |
| symbolic:-0.1b1                 | online_readout | 0.845541 |                   0.92757  |                         0.320476 |                          0.262028 |                        -0.0921429 |              12.1656  |              0.257332 |                 2 |
| symbolic:0.3b3                  | online_readout | 0.799812 |                   0.92757  |                         0.320476 |                          0.257941 |                        -0.0919048 |              30.572   |              0.257332 |                 2 |
| symbolic:0.3b1                  | online_readout | 0.799012 |                   0.92757  |                         0.320476 |                          0.257941 |                        -0.0919048 |              31.0813  |              0.257332 |                 2 |
| symbolic:0.3b2                  | online_readout | 0.798412 |                   0.92757  |                         0.320476 |                          0.257532 |                        -0.0921429 |              31.3058  |              0.257332 |                 2 |
| ambiguity_gated_oja             | online_readout | 0.798069 |                   0.928209 |                         0.320476 |                          0.257931 |                        -0.0921429 |              48.4102  |              0.25669  |                 1 |
| symbolic:-0.3b4                 | online_readout | 0.797443 |                   0.92586  |                         0.320476 |                          0.26101  |                        -0.0921429 |              31.8421  |              0.259078 |                 2 |
| symbolic:-0.3b3                 | online_readout | 0.797267 |                   0.92586  |                         0.320476 |                          0.26101  |                        -0.0921429 |              31.9577  |              0.259078 |                 2 |
| symbolic:-0.3b2                 | online_readout | 0.794889 |                   0.92586  |                         0.320476 |                          0.26101  |                        -0.0921429 |              33.5636  |              0.259078 |                 2 |

## 4. Validation Design
Validation uses held-out seeds and subgraph samples, the real graph, degree-preserving nulls, weight-shuffled real topology, uniform-weight real topology, and the representation probe as the stricter test.

## 5. Main Validation Result
Validation was skipped when the discovery screen failed the readout-only/plasticity-gain gate.

_No rows available._

## 6. Did We Discover a New Algorithm?
No new learning algorithm is supported; validation was skipped because discovery did not beat readout-only in the representation-probe protocol.

## 7. Limits
- Synthetic motion task.
- Simplified one-step message passing.
- No measured activity, stimulus movie, behavior, or synaptic biophysics.
- Approximate degree-preserving nulls.
- The result is a plasticity-efficiency proxy, not biological learning or thermodynamic heat.

## 8. Next Experiment
Use real visual stimulus/activity datasets or a biophysical SNN only after a rule clears readout-only and null controls in the representation-probe protocol.
