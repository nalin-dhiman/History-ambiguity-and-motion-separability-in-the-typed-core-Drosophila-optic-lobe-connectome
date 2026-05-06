"""Minimal PyTorch LIF network with connectome masks and thermodynamic proxies.

This is a proof-of-concept training scaffold, not a production-scale simulator.
For full FlyWire-scale experiments, use sampled subcircuits, cell-type graphs,
or a sparse PyTorch/Brian2 implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class FastSigmoidSpike(torch.autograd.Function):
    """Surrogate-gradient binary spike."""

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, slope: float = 25.0):
        ctx.save_for_backward(input_tensor)
        ctx.slope = slope
        return (input_tensor > 0).to(input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        slope = ctx.slope
        grad = grad_output / (slope * torch.abs(x) + 1.0) ** 2
        return grad, None


def spike_fn(x: torch.Tensor) -> torch.Tensor:
    return FastSigmoidSpike.apply(x)


class MaskedLinear(nn.Module):
    """Linear layer constrained by a fixed binary mask."""

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor, bias: bool = True):
        super().__init__()
        if mask.shape != (out_features, in_features):
            raise ValueError(f"mask must have shape {(out_features, in_features)}, got {tuple(mask.shape)}")
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer("mask", mask.float())
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


@dataclass
class ThermoLossWeights:
    spike: float = 1e-3
    landauer_proxy: float = 1e-2
    rate_entropy: float = 0.0


class ConnectomeMaskedLIF(nn.Module):
    """Small recurrent LIF network with a connectome-constrained recurrent mask."""

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        n_outputs: int,
        recurrent_mask: torch.Tensor,
        beta: float = 0.9,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.in_layer = nn.Linear(n_inputs, n_hidden)
        self.rec_layer = MaskedLinear(n_hidden, n_hidden, recurrent_mask, bias=False)
        self.out_layer = nn.Linear(n_hidden, n_outputs)
        self.beta = beta
        self.threshold = threshold

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        x_seq shape: (batch, time, n_inputs). Returns logits and stats.
        """
        batch, time, _ = x_seq.shape
        device = x_seq.device
        mem = torch.zeros(batch, self.rec_layer.weight.shape[0], device=device)
        spk = torch.zeros_like(mem)
        spikes = []
        for t in range(time):
            current = self.in_layer(x_seq[:, t]) + self.rec_layer(spk)
            mem = self.beta * mem + current - spk * self.threshold
            spk = spike_fn(mem - self.threshold)
            spikes.append(spk)
        spk_seq = torch.stack(spikes, dim=1)
        rates = spk_seq.mean(dim=1)
        logits = self.out_layer(rates)
        stats = {
            "spike_count": spk_seq.sum(dim=(1, 2)).mean(),
            "mean_rate": rates.mean(),
            "rates": rates,
            "spike_sequence": spk_seq,
        }
        return logits, stats


def bernoulli_entropy_torch(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))


def thermodynamic_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    input_rates: torch.Tensor,
    hidden_rates: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    weights: ThermoLossWeights = ThermoLossWeights(),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Cross-entropy plus spike and Landauer-style entropy-contraction proxy.

    The proxy uses Bernoulli entropies of input and hidden firing probabilities.
    It is differentiable and useful as a training regularizer, but should be
    audited post hoc with discrete empirical entropy estimators.
    """
    ce = F.cross_entropy(logits, labels)
    h_in = bernoulli_entropy_torch(input_rates).sum(dim=1).mean()
    h_hidden = bernoulli_entropy_torch(hidden_rates).sum(dim=1).mean()
    landauer_proxy_bits = F.relu(h_in - h_hidden)
    spike_penalty = stats["spike_count"]
    rate_entropy = bernoulli_entropy_torch(hidden_rates).mean()
    loss = ce + weights.spike * spike_penalty + weights.landauer_proxy * landauer_proxy_bits
    if weights.rate_entropy != 0:
        # Negative sign encourages high-entropy hidden rates if desired.
        loss = loss - weights.rate_entropy * rate_entropy
    parts = {
        "ce": ce.detach(),
        "spike_penalty": spike_penalty.detach(),
        "landauer_proxy_bits": landauer_proxy_bits.detach(),
        "h_input_bits": h_in.detach(),
        "h_hidden_bits": h_hidden.detach(),
    }
    return loss, parts
