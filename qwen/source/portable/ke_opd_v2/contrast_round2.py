"""Isolated Round-2 paired audio-condition contrastive objective.

This module deliberately has no imports from the V1/Round-1 training path.  It
contains the candidate objective and its audit surface only; wiring it into a
later round must be an explicit, separately reviewed change.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _contrast_compute_dtype(real: torch.Tensor, wrong: torch.Tensor) -> torch.dtype:
    """Use fp32 for low-precision inputs while retaining fp64 test precision."""

    dtype = torch.promote_types(real.dtype, wrong.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def paired_audio_condition_nce_rows(
    student_real_log_probs: torch.Tensor,
    student_wrong_log_probs: torch.Tensor,
    eligible_mask: torch.Tensor,
    token_word_ids: Sequence[Sequence[int | None]],
    *,
    tau: float = 1.0,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Compute complete-word, equal-word-weight paired audio-condition NCE.

    ``student_real_log_probs`` and ``student_wrong_log_probs`` are aligned
    sampled-token log probabilities for the same text under Real and Wrong
    audio.  A lexical word is eligible only when *all* of its BPE pieces are
    true in ``eligible_mask``.  Partially selected words fail closed.

    For each eligible word ``w``, the two scores are the means over its pieces,

    ``s_R(w) = mean_piece(log p(piece | Real audio))`` and
    ``s_W(w) = mean_piece(log p(piece | Wrong audio))``.

    The word loss is the two-way paired logistic/NCE objective

    ``softplus((s_W(w) - s_R(w)) / tau)``.

    Word losses, gaps, and Real-preference probabilities are averaged with
    equal word weight, irrespective of BPE length.  The Wrong branch is never
    detached.  When both branches have already been materialized, a row with no
    eligible complete word returns a differentiable zero connected to both
    graphs.  The formal trainer detects an empty gate before the extra Wrong
    forward and skips that branch entirely.

    Returns one loss per row and one JSON-friendly audit dictionary per row.
    ``paired_nce_mean_gap`` is ``mean_word(s_R - s_W)`` and
    ``paired_nce_mean_prob`` is the mean two-way probability assigned to Real,
    ``sigmoid((s_R - s_W) / tau)``.  Empty rows report zero for every statistic.
    """

    if student_real_log_probs.ndim != 2:
        raise ValueError("Student log probabilities must have shape [batch, positions]")
    if student_wrong_log_probs.shape != student_real_log_probs.shape:
        raise ValueError("Student Real/Wrong log probabilities differ in shape")
    if eligible_mask.shape != student_real_log_probs.shape:
        raise ValueError("eligible mask and Student log probabilities differ in shape")
    if eligible_mask.dtype != torch.bool:
        raise ValueError("eligible_mask must be a boolean tensor")
    if student_real_log_probs.device != student_wrong_log_probs.device:
        raise ValueError("Student Real/Wrong log probabilities must share a device")
    if eligible_mask.device != student_real_log_probs.device:
        raise ValueError("eligible_mask and Student log probabilities must share a device")
    if not student_real_log_probs.is_floating_point() or not student_wrong_log_probs.is_floating_point():
        raise ValueError("Student log probabilities must be floating-point tensors")

    try:
        tau_value = float(tau)
    except (TypeError, ValueError) as error:
        raise ValueError("tau must be finite and positive") from error
    if not math.isfinite(tau_value) or tau_value <= 0.0:
        raise ValueError("tau must be finite and positive")
    if not torch.isfinite(student_real_log_probs).all():
        raise FloatingPointError("non-finite Student-Real log probability")
    if not torch.isfinite(student_wrong_log_probs).all():
        raise FloatingPointError("non-finite Student-Wrong log probability")

    batch_size, position_count = student_real_log_probs.shape
    if len(token_word_ids) != batch_size:
        raise ValueError("word-ID rows do not match the tensor batch dimension")
    for row_word_ids in token_word_ids:
        if len(row_word_ids) != position_count:
            raise ValueError("word-ID row does not match the tensor position dimension")

    compute_dtype = _contrast_compute_dtype(
        student_real_log_probs, student_wrong_log_probs
    )
    real = student_real_log_probs.to(dtype=compute_dtype)
    wrong = student_wrong_log_probs.to(dtype=compute_dtype)
    row_losses: list[torch.Tensor] = []
    row_audits: list[dict[str, float]] = []

    for row_index, row_word_ids in enumerate(token_word_ids):
        # Insertion order follows the generated answer and does not require
        # word IDs to be sortable or numerically consecutive.
        by_word_indices: dict[int, list[int]] = {}
        for token_index, word_id in enumerate(row_word_ids):
            if word_id is not None:
                by_word_indices.setdefault(word_id, []).append(token_index)

        # One row transfer avoids a CPU/GPU synchronization for every word.
        row_eligible = eligible_mask[row_index].detach().to(device="cpu").tolist()
        eligible_word_indices = [
            indices
            for indices in by_word_indices.values()
            if all(bool(row_eligible[index]) for index in indices)
        ]

        if eligible_word_indices:
            flat_indices = [
                token_index
                for indices in eligible_word_indices
                for token_index in indices
            ]
            group_ids = [
                group_index
                for group_index, indices in enumerate(eligible_word_indices)
                for _ in indices
            ]
            token_index = torch.tensor(
                flat_indices, dtype=torch.long, device=real.device
            )
            group_index = torch.tensor(
                group_ids, dtype=torch.long, device=real.device
            )
            counts = torch.tensor(
                [len(indices) for indices in eligible_word_indices],
                dtype=compute_dtype,
                device=real.device,
            )
            real_scores = torch.zeros_like(counts).scatter_add(
                0, group_index, real[row_index].index_select(0, token_index)
            ) / counts
            wrong_scores = torch.zeros_like(counts).scatter_add(
                0, group_index, wrong[row_index].index_select(0, token_index)
            ) / counts
            word_gaps = real_scores - wrong_scores
            if not bool(torch.isfinite(word_gaps).all().item()):
                raise FloatingPointError("non-finite PAC-NCE word gap")
            scaled_gaps = word_gaps / tau_value
            word_probs = torch.sigmoid(scaled_gaps)
            word_losses = F.softplus(-scaled_gaps)
            if not bool(
                torch.isfinite(scaled_gaps).all().item()
                and torch.isfinite(word_probs).all().item()
                and torch.isfinite(word_losses).all().item()
            ):
                raise FloatingPointError("non-finite PAC-NCE intermediate")

            row_loss = word_losses.mean()
            mean_gap = word_gaps.mean()
            mean_prob = word_probs.mean()
            mean_gradient_factor = (1.0 - word_probs).mean()
            saturation_fraction = word_probs.ge(0.95).to(compute_dtype).mean()
            wrong_preferred_fraction = word_probs.lt(0.5).to(compute_dtype).mean()
            eligible_tokens = len(flat_indices)
            stats = torch.stack(
                (
                    mean_gap,
                    mean_prob,
                    row_loss,
                    mean_gradient_factor,
                    saturation_fraction,
                    wrong_preferred_fraction,
                )
            ).detach().to(device="cpu").tolist()
            if not all(math.isfinite(float(value)) for value in stats):
                raise FloatingPointError("non-finite PAC-NCE audit statistic")
            audit = {
                "paired_nce_eligible_words": float(len(eligible_word_indices)),
                "paired_nce_eligible_tokens": float(eligible_tokens),
                "paired_nce_mean_gap": float(stats[0]),
                "paired_nce_mean_prob": float(stats[1]),
                "paired_nce_mean_loss": float(stats[2]),
                "paired_nce_mean_gradient_factor": float(stats[3]),
                "paired_nce_saturation_fraction_preal_ge_0p95": float(stats[4]),
                "paired_nce_wrong_preferred_fraction": float(stats[5]),
            }
        else:
            # Both terms are required: optimizing an empty-gate batch must
            # materialize zero gradients on both condition branches rather
            # than silently dropping the Wrong-audio graph.
            row_loss = real[row_index].sum() * 0.0 + wrong[row_index].sum() * 0.0
            audit = {
                "paired_nce_eligible_words": 0.0,
                "paired_nce_eligible_tokens": 0.0,
                "paired_nce_mean_gap": 0.0,
                "paired_nce_mean_prob": 0.0,
                "paired_nce_mean_loss": 0.0,
                "paired_nce_mean_gradient_factor": 0.0,
                "paired_nce_saturation_fraction_preal_ge_0p95": 0.0,
                "paired_nce_wrong_preferred_fraction": 0.0,
            }
        row_losses.append(row_loss)
        row_audits.append(audit)

    if row_losses:
        losses = torch.stack(row_losses)
    else:
        # Preserve a valid autograd edge even for a degenerate empty batch.
        losses = real.sum(dim=1) * 0.0 + wrong.sum(dim=1) * 0.0
    return losses, row_audits


__all__ = ["paired_audio_condition_nce_rows"]
