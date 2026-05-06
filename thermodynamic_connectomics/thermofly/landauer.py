"""Landauer conversion utilities.

Landauer's bound gives k_B T ln(2) joules per erased bit for idealized
logically irreversible operations. Here we use it as an absolute lower-bound
scale for entropy contraction, not as an estimate of real neural metabolism.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

K_B: float = 1.380649e-23  # J/K, exact SI definition
LN2: float = math.log(2.0)


def landauer_joules(bits: float, temperature_kelvin: float = 293.15) -> float:
    """Convert erased bits to the Landauer lower-bound energy in joules.

    Parameters
    ----------
    bits:
        Entropy contraction in bits. Negative values are clipped to zero,
        because heat lower-bound proxies should not be negative.
    temperature_kelvin:
        Thermodynamic temperature. 293.15 K is room temperature.
    """
    return max(0.0, float(bits)) * K_B * float(temperature_kelvin) * LN2


def bits_from_joules(energy_joules: float, temperature_kelvin: float = 293.15) -> float:
    """Convert energy in joules to the equivalent number of Landauer bits."""
    denom = K_B * float(temperature_kelvin) * LN2
    if denom <= 0:
        raise ValueError("temperature_kelvin must be positive")
    return max(0.0, float(energy_joules)) / denom


@dataclass(frozen=True)
class LandauerScale:
    """Convenience object for repeated conversion at fixed temperature."""

    temperature_kelvin: float = 293.15

    @property
    def joules_per_bit(self) -> float:
        return K_B * self.temperature_kelvin * LN2

    def joules(self, bits: float) -> float:
        return landauer_joules(bits, self.temperature_kelvin)

    def bits(self, joules: float) -> float:
        return bits_from_joules(joules, self.temperature_kelvin)
