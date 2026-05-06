"""ThermoFly: Landauer-style irreversibility analysis for fly connectomes.

This package is intentionally conservative: it computes information-theoretic
lower-bound proxies and topology-conditioned audit metrics. It does not claim
actual biological heat dissipation unless paired with a validated biophysical
energy model.
"""

from .landauer import K_B, landauer_joules, bits_from_joules
from .entropy import shannon_bits, channel_metrics

__all__ = [
    "K_B",
    "landauer_joules",
    "bits_from_joules",
    "shannon_bits",
    "channel_metrics",
]
