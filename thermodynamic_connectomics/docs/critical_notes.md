# Critical notes for the manuscript

1. **Fan-in is not erasure.** A high fan-in neuron may integrate redundant signals, correlated signals, or even expand entropy through noise. Use empirical entropy contraction and history loss, not raw input count.

2. **Landauer is a lower bound.** The `k_B T ln 2` conversion is an ideal thermodynamic scale. Biological neurons dissipate energy through ion pumps, vesicle release, maintenance currents, and glial processes. The code does not model those costs.

3. **Static connectome is insufficient.** The connectome gives possible pathways and approximate synaptic mass. It does not by itself give signs, time constants, short-term plasticity, or behavioral state.

4. **Nulls are central.** Any novelty claim must survive degree-preserving, weight-preserving, cell-type-preserving, and spatially constrained nulls.

5. **Negative results are publishable.** If the Landauer atlas collapses under strict nulls, the result can become a rigorous methods paper about what connectomes can and cannot tell us about thermodynamic computation.
