"""Aggregate product constraint scores and apply them to generated SID prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from transformers import LogitsProcessor


@dataclass(frozen=True)
class PrefixAssignment:
    key_to_row: dict[tuple[int, ...], int]
    item_to_row: np.ndarray


def build_prefix_assignments(sid_token_ids: np.ndarray) -> list[PrefixAssignment]:
    result = []
    for depth in range(1, sid_token_ids.shape[1] + 1):
        key_to_row = {}
        assignments = np.empty(sid_token_ids.shape[0], dtype=np.int32)
        for item_index, values in enumerate(sid_token_ids[:, :depth]):
            key = tuple(int(value) for value in values)
            row = key_to_row.get(key)
            if row is None:
                row = len(key_to_row)
                key_to_row[key] = row
            assignments[item_index] = row
        result.append(PrefixAssignment(key_to_row, assignments))
    return result


def aggregate_prefix_feasibility(
    item_scores: np.ndarray, assignments: Sequence[PrefixAssignment]
) -> list[np.ndarray]:
    result = []
    for assignment in assignments:
        values = np.full(len(assignment.key_to_row), -np.inf, dtype=np.float32)
        np.maximum.at(values, assignment.item_to_row, item_scores)
        values[~np.isfinite(values)] = 0.0
        result.append(values)
    return result


class PrefixScoreLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        *,
        assignments: Sequence[PrefixAssignment],
        prefix_scores: Sequence[np.ndarray],
        allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]],
        weight: float,
        decoder_start_token_id: int,
    ) -> None:
        self.assignments = list(assignments)
        self.prefix_scores = list(prefix_scores)
        self.allowed_tokens_fn = allowed_tokens_fn
        self.weight = float(weight)
        self.decoder_start_token_id = int(decoder_start_token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.weight == 0:
            return scores
        for beam_index in range(input_ids.shape[0]):
            sequence = input_ids[beam_index]
            generated = sequence
            if generated.numel() and generated[0].item() == self.decoder_start_token_id:
                generated = generated[1:]
            depth = int(generated.numel())
            if depth >= len(self.assignments):
                continue
            prefix = tuple(int(value) for value in generated.tolist())
            tokens = []
            values = []
            for token in self.allowed_tokens_fn(beam_index, sequence):
                row = self.assignments[depth].key_to_row.get(prefix + (int(token),))
                if row is not None:
                    tokens.append(int(token))
                    values.append(float(self.prefix_scores[depth][row]))
            if tokens:
                scores[beam_index, torch.tensor(tokens, device=scores.device)] += (
                    self.weight
                    * torch.tensor(values, device=scores.device, dtype=scores.dtype)
                )
        return scores
