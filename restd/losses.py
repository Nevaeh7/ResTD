"""Training objectives for multi-horizon ResTD."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .geometry import behavior_distribution, behavior_log_distribution


def _validate_inputs(
    decoder_hidden_states: torch.Tensor,
    residual_targets: torch.Tensor,
    codebooks: Sequence[torch.Tensor],
    horizon_heads: Sequence[nn.Module],
    final_sid_targets: torch.Tensor | None,
    valid_sid_mask: torch.Tensor | None,
) -> tuple[int, int]:
    if decoder_hidden_states.ndim != 3:
        raise ValueError("decoder_hidden_states must have shape [B, T, d_model]")
    if residual_targets.ndim != 3:
        raise ValueError("residual_targets must have shape [B, L+1, rq_dim]")
    if decoder_hidden_states.shape[0] != residual_targets.shape[0]:
        raise ValueError("decoder and residual batch sizes differ")
    num_codebooks = len(codebooks)
    if num_codebooks == 0:
        raise ValueError("at least one RQ codebook is required")
    if residual_targets.shape[1] != num_codebooks + 1:
        raise ValueError(
            "residual trace length must equal number of codebooks + 1: "
            f"{residual_targets.shape[1]} != {num_codebooks + 1}"
        )
    if decoder_hidden_states.shape[1] < num_codebooks:
        raise ValueError(
            "decoder sequence is shorter than the Semantic ID: "
            f"{decoder_hidden_states.shape[1]} < {num_codebooks}"
        )
    if len(horizon_heads) == 0:
        raise ValueError("at least one ResTD horizon head is required")
    rq_dim = residual_targets.shape[-1]
    for level, codebook in enumerate(codebooks):
        if codebook.ndim != 2 or codebook.shape[-1] != rq_dim:
            raise ValueError(
                f"codebook {level} has incompatible shape {tuple(codebook.shape)}"
            )
    expected = (decoder_hidden_states.shape[0], num_codebooks)
    if final_sid_targets is not None and tuple(final_sid_targets.shape) != expected:
        raise ValueError(
            f"final_sid_targets must have shape {expected}, "
            f"got {tuple(final_sid_targets.shape)}"
        )
    if valid_sid_mask is not None and tuple(valid_sid_mask.shape) != expected:
        raise ValueError(
            f"valid_sid_mask must have shape {expected}, "
            f"got {tuple(valid_sid_mask.shape)}"
        )
    return num_codebooks, rq_dim


def multi_horizon_restd_loss(
    decoder_hidden_states: torch.Tensor,
    residual_targets: torch.Tensor,
    codebooks: Sequence[torch.Tensor],
    horizon_heads: Sequence[nn.Module],
    teacher_temperature: float,
    student_temperature: float,
    horizon_decay: float,
    valid_sid_mask: torch.Tensor | None = None,
    final_sid_targets: torch.Tensor | None = None,
    collision_epsilon: float = 0.1,
    adaptive_collision_margin: float = 0.001,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted, normalized multi-horizon behavioral KL.

    Zero-indexed alignment is deliberately explicit:

    * ``decoder_hidden_states[:, t]`` predicts SID position ``t``;
    * horizon ``s`` uses ``residual_targets[:, t+s]``;
    * the behavior distribution is evaluated on ``codebooks[t+s]``.

    The loss denominator is the sum of valid horizon weights.  Consequently,
    increasing the horizon does not silently multiply the auxiliary loss scale.
    """

    num_codebooks, _ = _validate_inputs(
        decoder_hidden_states,
        residual_targets,
        codebooks,
        horizon_heads,
        final_sid_targets,
        valid_sid_mask,
    )
    if not 0.0 < horizon_decay <= 1.0:
        raise ValueError("horizon_decay must be in (0, 1]")
    if not 0.0 <= collision_epsilon <= 1.0:
        raise ValueError("collision_epsilon must be in [0, 1]")
    if collision_epsilon > 0.0 and final_sid_targets is None:
        raise ValueError(
            "final_sid_targets are required when collision_epsilon is non-zero"
        )

    batch_size = decoder_hidden_states.shape[0]
    device = decoder_hidden_states.device
    if valid_sid_mask is None:
        valid_sid_mask = torch.ones(
            (batch_size, num_codebooks), dtype=torch.bool, device=device
        )
    else:
        valid_sid_mask = valid_sid_mask.to(device=device, dtype=torch.bool)
    if final_sid_targets is not None:
        final_sid_targets = final_sid_targets.to(device=device, dtype=torch.long)

    total_numerator = decoder_hidden_states.sum() * 0.0
    total_denominator = torch.zeros((), device=device, dtype=torch.float32)
    metrics: dict[str, torch.Tensor] = {}

    max_horizon = min(len(horizon_heads), num_codebooks)
    for horizon in range(max_horizon):
        horizon_numerator = decoder_hidden_states.sum() * 0.0
        horizon_count = torch.zeros((), device=device, dtype=torch.float32)
        teacher_correct = torch.zeros((), device=device, dtype=torch.float32)
        student_correct = torch.zeros((), device=device, dtype=torch.float32)

        for decoder_position in range(num_codebooks - horizon):
            future_position = decoder_position + horizon
            mask = valid_sid_mask[:, future_position]
            valid_count = mask.sum().to(dtype=torch.float32)
            if not bool(mask.any()):
                continue

            codebook = codebooks[future_position]
            teacher_residual = residual_targets[:, future_position].detach()
            teacher_probabilities = behavior_distribution(
                teacher_residual,
                codebook,
                teacher_temperature,
            ).detach()

            target = None
            if final_sid_targets is not None:
                target = final_sid_targets[:, future_position]
                if bool(
                    ((target[mask] < 0) | (target[mask] >= codebook.shape[0])).any()
                ):
                    raise ValueError(
                        f"final SID target is outside codebook {future_position}"
                    )
                safe_target = target.clamp(min=0, max=codebook.shape[0] - 1)
                teacher_probabilities, _ = apply_adaptive_target_correction(
                    teacher_probabilities,
                    safe_target,
                    epsilon_floor=collision_epsilon,
                    target_margin=adaptive_collision_margin,
                )

            student_residual = horizon_heads[horizon](
                decoder_hidden_states[:, decoder_position]
            )
            student_log_probabilities = behavior_log_distribution(
                student_residual,
                codebook,
                student_temperature,
            )
            per_example_kl = F.kl_div(
                student_log_probabilities,
                teacher_probabilities,
                reduction="none",
            ).sum(dim=-1)
            horizon_numerator = horizon_numerator + per_example_kl[mask].sum()
            horizon_count = horizon_count + valid_count

            if target is not None:
                teacher_correct = (
                    teacher_correct
                    + (teacher_probabilities.argmax(dim=-1)[mask] == target[mask])
                    .float()
                    .sum()
                )
                student_correct = (
                    student_correct
                    + (student_log_probabilities.argmax(dim=-1)[mask] == target[mask])
                    .float()
                    .sum()
                )

        if bool(horizon_count > 0):
            horizon_mean = horizon_numerator / horizon_count
            weight = float(horizon_decay**horizon)
            total_numerator = total_numerator + weight * horizon_numerator
            total_denominator = total_denominator + weight * horizon_count
            metrics[f"h{horizon + 1}_kl"] = horizon_mean.detach()
            if final_sid_targets is not None:
                metrics[f"h{horizon + 1}_teacher_top1"] = (
                    teacher_correct / horizon_count
                ).detach()
                metrics[f"h{horizon + 1}_student_top1"] = (
                    student_correct / horizon_count
                ).detach()

    if not bool(total_denominator > 0):
        metrics["valid_weight"] = total_denominator.detach()
        return decoder_hidden_states.sum() * 0.0, metrics

    loss = total_numerator / total_denominator
    metrics["valid_weight"] = total_denominator.detach()
    metrics["loss"] = loss.detach()
    return loss, metrics


def apply_adaptive_target_correction(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    epsilon_floor: float,
    target_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimally correct RQ behavior to respect collision-resolved SIDs.

    Choose the smallest per-example interpolation coefficient that places
    the stored SID target above all competing codes by the requested margin,
    subject to the minimum mixing weight ``epsilon_floor``.
    """

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N, K]")
    if targets.ndim != 1 or targets.shape[0] != probabilities.shape[0]:
        raise ValueError("targets must have shape [N]")
    if not 0.0 <= epsilon_floor <= 1.0:
        raise ValueError("epsilon_floor must be in [0, 1]")
    if not 0.0 <= target_margin < 1.0:
        raise ValueError("target_margin must be in [0, 1)")
    targets = targets.to(device=probabilities.device, dtype=torch.long)
    if bool(((targets < 0) | (targets >= probabilities.shape[1])).any()):
        raise ValueError("target is outside the teacher codebook")

    target_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
    competitors = probabilities.clone()
    competitors.scatter_(1, targets[:, None], -1.0)
    maximum_competitor = competitors.max(dim=-1).values
    numerator = maximum_competitor - target_probability + target_margin
    denominator = 1.0 - target_probability + maximum_competitor
    required = (numerator / denominator.clamp_min(1e-12)).clamp(0.0, 1.0)
    epsilon = torch.maximum(
        required,
        torch.full_like(required, float(epsilon_floor)),
    ).clamp_max(1.0)
    one_hot = F.one_hot(targets, num_classes=probabilities.shape[1]).to(
        dtype=probabilities.dtype
    )
    corrected = (1.0 - epsilon[:, None]) * probabilities + epsilon[:, None] * one_hot
    return corrected, epsilon


def capped_restd_lambda(
    lm_loss: torch.Tensor,
    restd_loss: torch.Tensor,
    *,
    scheduled_lambda: float,
    max_loss_ratio: float,
) -> torch.Tensor:
    """Return a detached auxiliary weight capped relative to SID LM loss.

    Keep the scheduled coefficient when the auxiliary contribution is below
    the configured cap. Otherwise enforce ``lambda * L_RT <= ratio * L_LM``
    per batch, with the scaling factor detached from the computation graph.
    """

    if scheduled_lambda < 0.0:
        raise ValueError("scheduled_lambda must be non-negative")
    if max_loss_ratio < 0.0:
        raise ValueError("max_loss_ratio must be non-negative")
    scheduled = torch.as_tensor(
        float(scheduled_lambda),
        device=restd_loss.device,
        dtype=torch.float32,
    )
    if max_loss_ratio == 0.0:
        return scheduled
    cap = (
        float(max_loss_ratio)
        * lm_loss.detach().float().clamp_min(0.0)
        / restd_loss.detach().float().clamp_min(1e-12)
    )
    return torch.minimum(scheduled, cap).detach()
