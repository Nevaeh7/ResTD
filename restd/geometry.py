"""Numerically stable RQ codebook geometry used by ResTD."""

from __future__ import annotations

import torch


def _check_geometry_inputs(
    x: torch.Tensor,
    codebook: torch.Tensor,
    temperature: float,
) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must have shape [B, D], got {tuple(x.shape)}")
    if codebook.ndim != 2:
        raise ValueError(
            f"codebook must have shape [K, D], got {tuple(codebook.shape)}"
        )
    if x.shape[-1] != codebook.shape[-1]:
        raise ValueError(
            f"x/codebook dimension mismatch: {x.shape[-1]} != {codebook.shape[-1]}"
        )
    if codebook.shape[0] == 0:
        raise ValueError("codebook must contain at least one codeword")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")


def squared_l2_logits(
    x: torch.Tensor,
    codebook: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Return ``-||x-c||^2 / temperature`` in float32.

    ResTD intentionally preserves the Euclidean geometry learned by the RQ
    tokenizer.  Inputs are promoted to float32 and autocast is disabled for the
    distance calculation so a bf16/fp16 retriever cannot make the teacher
    distribution numerically unstable.
    """

    _check_geometry_inputs(x, codebook, temperature)
    with torch.autocast(device_type=x.device.type, enabled=False):
        x32 = x.float()
        codebook32 = codebook.detach().to(device=x.device, dtype=torch.float32)
        distance2 = (
            x32.square().sum(dim=-1, keepdim=True)
            + codebook32.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * x32 @ codebook32.t()
        )
        # Roundoff may make an exact zero very slightly negative.
        distance2 = distance2.clamp_min(0.0)
        return -distance2 / float(temperature)


def behavior_distribution(
    x: torch.Tensor,
    codebook: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Return the tokenizer-induced probability distribution over codewords."""

    return torch.softmax(squared_l2_logits(x, codebook, temperature), dim=-1)


def behavior_log_distribution(
    x: torch.Tensor,
    codebook: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Return log probabilities without a probability-to-log round trip."""

    return torch.log_softmax(squared_l2_logits(x, codebook, temperature), dim=-1)
