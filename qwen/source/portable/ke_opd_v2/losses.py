"""Numerically explicit V2 losses and Audio-Advantage weighting."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence

import torch
import torch.nn.functional as F


def _select_valid_logits(logits: torch.Tensor, valid_token_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError("logits must end in a vocabulary dimension")
    if valid_token_ids.ndim != 1 or valid_token_ids.dtype != torch.long:
        raise ValueError("valid_token_ids must be a 1-D int64 tensor")
    selected = logits.float().index_select(-1, valid_token_ids.to(logits.device))
    if not torch.isfinite(selected).all():
        raise FloatingPointError("non-finite logits in the valid text vocabulary")
    return selected


def reverse_kl_per_position(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_token_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Exact D_KL(Student || Teacher) on the shared valid text vocabulary."""
    if student_logits.shape[:-1] != teacher_logits.shape[:-1]:
        raise ValueError("student and teacher prefix/position dimensions differ")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student = _select_valid_logits(student_logits, valid_token_ids) / temperature
    teacher = _select_valid_logits(teacher_logits, valid_token_ids) / temperature
    student_logp = F.log_softmax(student, dim=-1)
    teacher_logp = F.log_softmax(teacher, dim=-1)
    kl = (student_logp.exp() * (student_logp - teacher_logp)).sum(dim=-1)
    if torch.any(kl < -2e-5):
        raise FloatingPointError(f"reverse KL became negative: min={float(kl.min())}")
    return kl.clamp_min(0.0) * (temperature**2)


def weighted_position_mean(values: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over visited completion positions, with optional AA position weights."""
    if values.shape != weights.shape or values.shape != mask.shape:
        raise ValueError("values, weights, and mask must have identical shape")
    active_weights = weights.float() * mask.to(weights.dtype)
    numer = (values.float() * active_weights).sum(dim=-1)
    denom = active_weights.sum(dim=-1)
    if torch.any(denom <= 0):
        raise ValueError("every row must contain at least one valid completion position")
    return numer / denom


def answer_span_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = -100
) -> torch.Tensor:
    """CE on the already-masked answer span; prompt and think labels stay -100."""
    shifted_logits = logits[..., :-1, :].contiguous().float()
    shifted_labels = labels[..., 1:].contiguous()
    flat = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(shifted_labels)
    mask = shifted_labels.ne(ignore_index)
    counts = mask.sum(dim=-1)
    if torch.any(counts <= 0):
        raise ValueError("each row must supervise at least one answer token")
    return (flat * mask).sum(dim=-1) / counts


def sampled_token_audio_advantage(
    teacher_real_logits: torch.Tensor,
    teacher_wrong_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    valid_token_ids: torch.Tensor,
    *,
    support_top_k: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real-Wrong sampled-token advantage with absolute Real support gating.

    Probabilities are normalized on the same valid vocabulary used by RKLD.
    Returns ``(positive_advantage, support_mask)``.
    """
    log_ratio, supported = sampled_token_audio_log_ratio(
        teacher_real_logits,
        teacher_wrong_logits,
        sampled_token_ids,
        valid_token_ids,
        support_top_k=support_top_k,
    )
    advantage = torch.where(
        supported,
        log_ratio.clamp_min(0.0),
        torch.zeros_like(log_ratio),
    )
    return advantage, supported


def sampled_token_audio_log_ratio(
    teacher_real_logits: torch.Tensor,
    teacher_wrong_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    valid_token_ids: torch.Tensor,
    *,
    support_top_k: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw sampled-token ``log(P_real / P_wrong)`` and Real support.

    Negative piece-level values are retained so an exact whole-word joint
    likelihood ratio can sum all pieces before applying its threshold.
    """
    if teacher_real_logits.shape != teacher_wrong_logits.shape:
        raise ValueError("Real and Wrong Teacher logits must have identical shape")
    if teacher_real_logits.shape[:-1] != sampled_token_ids.shape:
        raise ValueError("sampled token IDs do not match prefix dimensions")
    valid_ids = valid_token_ids.to(teacher_real_logits.device)
    real = _select_valid_logits(teacher_real_logits, valid_ids)
    wrong = _select_valid_logits(teacher_wrong_logits, valid_ids)
    real_logp = F.log_softmax(real, dim=-1)
    wrong_logp = F.log_softmax(wrong, dim=-1)

    vocab_size = teacher_real_logits.shape[-1]
    id_to_local = torch.full((vocab_size,), -1, dtype=torch.long, device=real.device)
    id_to_local[valid_ids] = torch.arange(valid_ids.numel(), device=real.device)
    local_ids = id_to_local[sampled_token_ids]
    sampled_valid = local_ids.ge(0)
    safe_ids = local_ids.clamp_min(0).unsqueeze(-1)
    sampled_real = real_logp.gather(-1, safe_ids).squeeze(-1)
    sampled_wrong = wrong_logp.gather(-1, safe_ids).squeeze(-1)

    k = min(max(int(support_top_k), 1), valid_ids.numel())
    top_ids = real_logp.topk(k, dim=-1).indices
    supported = sampled_valid & top_ids.eq(safe_ids).any(dim=-1)
    return sampled_real - sampled_wrong, supported


def sampled_token_log_prob(
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    valid_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather sampled-token log probabilities on the valid text vocabulary.

    This is the Student-side counterpart to the sampled-token quantities used
    by AA.  Normalization excludes invalid/non-text vocabulary entries, exactly
    as RKLD and Teacher Real/Wrong ratios do.  A sampled token outside the
    declared valid vocabulary is a contract error rather than a masked value.
    """
    if logits.shape[:-1] != sampled_token_ids.shape:
        raise ValueError("sampled token IDs do not match prefix dimensions")
    if sampled_token_ids.dtype != torch.long:
        raise ValueError("sampled_token_ids must be an int64 tensor")
    valid_ids = valid_token_ids.to(logits.device)
    selected = _select_valid_logits(logits, valid_ids)
    log_probs = F.log_softmax(selected, dim=-1)

    vocab_size = logits.shape[-1]
    if torch.any(sampled_token_ids < 0) or torch.any(sampled_token_ids >= vocab_size):
        raise ValueError("sampled token ID lies outside the model vocabulary")
    id_to_local = torch.full(
        (vocab_size,), -1, dtype=torch.long, device=log_probs.device
    )
    id_to_local[valid_ids] = torch.arange(valid_ids.numel(), device=log_probs.device)
    local_ids = id_to_local[sampled_token_ids.to(log_probs.device)]
    if torch.any(local_ids < 0):
        raise ValueError("sampled token ID is absent from the valid text vocabulary")
    return log_probs.gather(-1, local_ids.unsqueeze(-1)).squeeze(-1)


def top_fraction_word_weights(
    token_advantages: Sequence[float],
    token_word_ids: Sequence[int | None],
    *,
    high_weight: float = 2.0,
    top_fraction: float = 0.20,
) -> list[float]:
    """Assign High to tokens in at most the top fraction of positive whole words.

    A word's raw AA is the mean of all its BPE pieces. There is no smoothing,
    quantile remapping, or per-row mean normalization.
    """
    if len(token_advantages) != len(token_word_ids):
        raise ValueError("token_advantages and token_word_ids differ in length")
    if not 0 < top_fraction <= 1 or high_weight < 1:
        raise ValueError("invalid top fraction or high weight")
    by_word_values: dict[int, list[float]] = {}
    for score, word_id in zip(token_advantages, token_word_ids):
        if word_id is None:
            continue
        by_word_values.setdefault(word_id, []).append(float(score))
    by_word = {
        word_id: sum(scores) / len(scores) for word_id, scores in by_word_values.items()
    }
    positive = [(score, word_id) for word_id, score in by_word.items() if score > 0]
    if not positive:
        return [1.0] * len(token_advantages)
    count = min(len(positive), max(1, math.ceil(len(by_word) * top_fraction)))
    selected = {word_id for _, word_id in sorted(positive, key=lambda item: (-item[0], item[1]))[:count]}
    return [high_weight if word_id in selected else 1.0 for word_id in token_word_ids]


def stopword_prefilter_top_fraction_word_weights(
    token_advantages: Sequence[float],
    token_word_ids: Sequence[int | None],
    word_texts: Mapping[int, str],
    stopwords: Collection[str],
    *,
    high_weight: float = 2.0,
    top_fraction: float = 0.20,
) -> tuple[list[float], dict[str, object]]:
    """Apply the frozen V1 Word-AA stopword rule before Top-20 selection.

    Word scores retain the V1.0 mean of non-negative, support-gated piece
    advantages. Exact Unicode-casefold stopword matches are removed from the
    positive candidate pool before ranking. The Top-20 limit is still computed
    from every lexical reasoning word, including blocked words.
    """
    if len(token_advantages) != len(token_word_ids):
        raise ValueError("token_advantages and token_word_ids differ in length")
    if not 0 < top_fraction <= 1 or high_weight < 1:
        raise ValueError("invalid top fraction or high weight")
    normalized_stopwords = {str(word).casefold() for word in stopwords}
    if not normalized_stopwords or "" in normalized_stopwords:
        raise ValueError("stopword set must contain non-empty lexical words")

    by_word_values: dict[int, list[float]] = {}
    by_word_indices: dict[int, list[int]] = {}
    for index, (score, word_id) in enumerate(zip(token_advantages, token_word_ids)):
        if word_id is None:
            continue
        by_word_values.setdefault(word_id, []).append(float(score))
        by_word_indices.setdefault(word_id, []).append(index)
    missing_text = sorted(set(by_word_values) - set(word_texts))
    if missing_text:
        raise ValueError(f"missing lexical text for word IDs: {missing_text}")

    by_word = {
        word_id: sum(scores) / len(scores) for word_id, scores in by_word_values.items()
    }
    normalized = {word_id: str(word_texts[word_id]).casefold() for word_id in by_word}
    blocked = {
        word_id for word_id, text in normalized.items() if text in normalized_stopwords
    }
    positive = {word_id for word_id, score in by_word.items() if score > 0}
    eligible = positive - blocked
    top20_limit = max(1, math.ceil(len(by_word) * top_fraction)) if by_word else 0

    original_selected = {
        word_id
        for word_id in sorted(positive, key=lambda item: (-by_word[item], item))[
            :top20_limit
        ]
    }
    selected = {
        word_id
        for word_id in sorted(eligible, key=lambda item: (-by_word[item], item))[
            :top20_limit
        ]
    }
    weights = [
        high_weight if word_id is not None and word_id in selected else 1.0
        for word_id in token_word_ids
    ]
    hit_counts = Counter(normalized[word_id] for word_id in blocked)
    partial_mismatch = sum(
        len({weights[index] for index in indices}) != 1
        for indices in by_word_indices.values()
    )
    audit: dict[str, object] = {
        "aa_lexical_words": len(by_word),
        "aa_positive_words_before_stopwords": len(positive),
        "aa_stopword_positive_words": len(positive & blocked),
        "aa_eligible_positive_words": len(eligible),
        "aa_top20_limit": top20_limit,
        "aa_selected_after_prefilter": len(selected),
        "aa_zero_high_rows_after_prefilter": int(not selected),
        "aa_stopword_hit_by_word": dict(sorted(hit_counts.items())),
        "aa_original_top20_stopword_hits": len(original_selected & blocked),
        "aa_promoted_nonstop_words_vs_original_top20": len(
            selected - original_selected
        ),
        "aa_high_bpe_after_prefilter": sum(weight > 1 for weight in weights),
        "answer_high_bpe": sum(
            word_id is None and weight > 1
            for word_id, weight in zip(token_word_ids, weights)
        ),
        "partial_word_weight_mismatch": partial_mismatch,
        "selected_word_ids": sorted(selected),
    }
    if len(eligible) != len(positive) - len(positive & blocked):
        raise AssertionError("stopword candidate accounting is inconsistent")
    if len(selected) != min(len(eligible), top20_limit):
        raise AssertionError("stopword Top-20 selection count is inconsistent")
    if selected & blocked or partial_mismatch:
        raise AssertionError("stopword prefilter produced an invalid High word")
    return weights, audit


def joint_top_fraction_word_weights(
    token_log_ratios: Sequence[float],
    token_support: Sequence[bool],
    token_word_ids: Sequence[int | None],
    *,
    high_weight: float = 2.0,
    top_fraction: float = 0.20,
) -> list[float]:
    """Rank whole words by a support-masked signed joint log ratio.

    Supported piece-level Real/Wrong log ratios retain their sign and are
    summed within each lexical word. Unsupported pieces contribute zero, as in
    the frozen V1.0 support semantics. At most ``top_fraction`` of all lexical
    words are selected from words whose joint score is strictly positive.
    """
    if not (
        len(token_log_ratios) == len(token_support) == len(token_word_ids)
    ):
        raise ValueError("log ratios, support, and word IDs differ in length")
    if not 0 < top_fraction <= 1 or high_weight < 1:
        raise ValueError("invalid top fraction or high weight")

    by_word: dict[int, float] = {}
    for score, supported, word_id in zip(
        token_log_ratios, token_support, token_word_ids
    ):
        if word_id is None:
            continue
        by_word.setdefault(word_id, 0.0)
        if supported:
            by_word[word_id] += float(score)

    positive = [(score, word_id) for word_id, score in by_word.items() if score > 0]
    if not positive:
        return [1.0] * len(token_log_ratios)
    count = min(len(positive), max(1, math.ceil(len(by_word) * top_fraction)))
    selected = {
        word_id
        for _, word_id in sorted(positive, key=lambda item: (-item[0], item[1]))[:count]
    }
    return [high_weight if word_id in selected else 1.0 for word_id in token_word_ids]


def token_top_fraction_weights(
    token_advantages: Sequence[float],
    token_word_ids: Sequence[int | None],
    *,
    high_weight: float = 2.0,
    top_fraction: float = 0.20,
) -> list[float]:
    """Select positive support-gated sampled BPE positions directly.

    ``token_word_ids`` is used only as the frozen lexical-reasoning mask:
    ``None`` positions (tags, answer, whitespace, punctuation) are ineligible.
    Scores are ranked per original sampled position, so ties are stable by
    generation order and High weight never spreads to sibling BPE pieces.
    """

    if len(token_advantages) != len(token_word_ids):
        raise ValueError("token advantages and lexical masks differ in length")
    if not 0 < top_fraction <= 1 or high_weight < 1:
        raise ValueError("invalid top fraction or high weight")
    eligible = [
        index for index, word_id in enumerate(token_word_ids) if word_id is not None
    ]
    positive = [
        index for index in eligible if float(token_advantages[index]) > 0
    ]
    if not positive:
        return [1.0] * len(token_advantages)
    count = min(len(positive), max(1, math.ceil(len(eligible) * top_fraction)))
    selected = set(
        sorted(
            positive,
            key=lambda index: (-float(token_advantages[index]), index),
        )[:count]
    )
    return [
        high_weight if index in selected else 1.0
        for index in range(len(token_advantages))
    ]


def joint_word_ratio_weights(
    token_log_ratios: Sequence[float],
    token_support: Sequence[bool],
    token_word_ids: Sequence[int | None],
    *,
    high_weight: float = 2.0,
    ratio_threshold: float = 2.0,
) -> list[float]:
    """Select a whole word by its joint Real/Wrong likelihood ratio.

    Every piece must pass the Real top-k support gate. The signed piece log
    ratios are then summed and compared strictly against
    ``log(ratio_threshold)``. All pieces of a selected word receive
    ``high_weight``; every other token retains weight 1.
    """
    if not (
        len(token_log_ratios) == len(token_support) == len(token_word_ids)
    ):
        raise ValueError("log ratios, support, and word IDs differ in length")
    if high_weight < 1:
        raise ValueError("high_weight must be at least 1")
    if ratio_threshold <= 1:
        raise ValueError("ratio_threshold must be greater than 1")

    by_word_indices: dict[int, list[int]] = {}
    for index, word_id in enumerate(token_word_ids):
        if word_id is not None:
            by_word_indices.setdefault(word_id, []).append(index)

    cutoff = math.log(float(ratio_threshold))
    selected: set[int] = set()
    for word_id, indices in by_word_indices.items():
        if all(bool(token_support[index]) for index in indices):
            joint_log_ratio = sum(float(token_log_ratios[index]) for index in indices)
            if joint_log_ratio > cutoff:
                selected.add(word_id)
    return [high_weight if word_id in selected else 1.0 for word_id in token_word_ids]


def strict_aa_word_contrast_targets(
    teacher_log_ratios: torch.Tensor,
    selected_mask: torch.Tensor,
    support_mask: torch.Tensor,
    token_word_ids: Sequence[Sequence[int | None]],
    *,
    margin_cap: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    """Build the frozen Round-1 complete-word contrast gate and margins.

    A lexical word is eligible only when every one of its BPE pieces is both
    selected by AA and supported by the Teacher-Real top-k gate.  Its signed
    Teacher Real/Wrong log ratios are summed before applying the strict
    ``joint ratio > 2`` test.  An eligible word receives the adaptive margin

    ``min(margin_cap, joint_teacher_log_ratio / number_of_pieces)``.

    The returned tensors have the input token shape.  ``eligible_tokens`` is
    true for every piece of an eligible complete word and ``token_margins``
    repeats that word's nat/token margin on all of its pieces.  The third item
    is one numeric audit dictionary per row so every contrastive arm can report
    the exact same gate population.  Teacher values are always detached; this
    helper defines targets rather than a trainable Teacher path.
    """
    if teacher_log_ratios.ndim != 2:
        raise ValueError("teacher log ratios must have shape [batch, positions]")
    if selected_mask.shape != teacher_log_ratios.shape:
        raise ValueError("selected mask and teacher log ratios differ in shape")
    if support_mask.shape != teacher_log_ratios.shape:
        raise ValueError("support mask and teacher log ratios differ in shape")
    if selected_mask.dtype != torch.bool or support_mask.dtype != torch.bool:
        raise ValueError("selected and support masks must be boolean tensors")
    if not 0 < float(margin_cap) <= 0.10:
        raise ValueError("margin_cap must be in (0, 0.10]")

    batch_size, position_count = teacher_log_ratios.shape
    if len(token_word_ids) != batch_size:
        raise ValueError("word-ID rows do not match the tensor batch dimension")
    for row_word_ids in token_word_ids:
        if len(row_word_ids) != position_count:
            raise ValueError("word-ID row does not match the tensor position dimension")

    # Gate in float64 so a supplied exact log(2) remains on the rejected side
    # of the strict boundary.  Margins returned to the loss remain float32.
    ratios = teacher_log_ratios.detach().to(dtype=torch.float64)
    if not torch.isfinite(ratios).all():
        raise FloatingPointError("non-finite Teacher Real/Wrong log ratio")
    eligible_tokens = torch.zeros_like(selected_mask, dtype=torch.bool)
    token_margins = torch.zeros_like(teacher_log_ratios, dtype=torch.float32)
    row_audits: list[dict[str, float]] = []
    cutoff = math.log(2.0)

    for row_index, row_word_ids in enumerate(token_word_ids):
        by_word_indices: dict[int, list[int]] = {}
        for token_index, word_id in enumerate(row_word_ids):
            if word_id is not None:
                by_word_indices.setdefault(word_id, []).append(token_index)

        complete_selected_words = 0
        complete_supported_words = 0
        joint_ratio_gt2_words = 0
        eligible_words = 0
        eligible_margins: list[float] = []
        for indices in by_word_indices.values():
            index = torch.tensor(indices, dtype=torch.long, device=ratios.device)
            complete_selected = bool(
                selected_mask[row_index].index_select(0, index).all()
            )
            complete_supported = bool(
                support_mask[row_index].index_select(0, index).all()
            )
            joint_log_ratio = ratios[row_index].index_select(0, index).sum()
            joint_ratio_gt2 = float(joint_log_ratio.item()) > cutoff
            complete_selected_words += int(complete_selected)
            complete_supported_words += int(complete_supported)
            joint_ratio_gt2_words += int(joint_ratio_gt2)
            if not (complete_selected and complete_supported and joint_ratio_gt2):
                continue
            margin = min(float(margin_cap), float(joint_log_ratio.item()) / len(indices))
            eligible_tokens[row_index, index] = True
            token_margins[row_index, index] = margin
            eligible_words += 1
            eligible_margins.append(margin)

        row_audits.append(
            {
                "contrast_lexical_words": float(len(by_word_indices)),
                "contrast_complete_selected_words": float(complete_selected_words),
                "contrast_complete_supported_words": float(complete_supported_words),
                "contrast_joint_ratio_gt2_words": float(joint_ratio_gt2_words),
                "contrast_eligible_words": float(eligible_words),
                "contrast_eligible_tokens": float(
                    eligible_tokens[row_index].sum().item()
                ),
                "contrast_mean_margin_nat_per_token": float(
                    sum(eligible_margins) / len(eligible_margins)
                    if eligible_margins
                    else 0.0
                ),
            }
        )

    return eligible_tokens, token_margins, row_audits


def aa_selected_word_contrastive_rows(
    student_real_log_probs: torch.Tensor,
    student_wrong_log_probs: torch.Tensor,
    teacher_log_ratios: torch.Tensor,
    selected_mask: torch.Tensor,
    support_mask: torch.Tensor,
    token_word_ids: Sequence[Sequence[int | None]],
    *,
    margin_cap: float = 0.10,
) -> torch.Tensor:
    """One-sided Student Real-vs-Wrong hinge on strict AA-selected words.

    Inputs are sampled-token log probabilities aligned to the same on-policy
    rollout.  For each eligible complete word ``w`` the loss is

    ``relu(m_w - (mean(log p_S(w | Real)) - stopgrad(mean(log p_S(w | Wrong)))))``.

    Eligible word hinges are averaged with equal word weight, then returned as
    one scalar per batch row.  Rows without an eligible word return a
    differentiable zero through the Real branch.  The Wrong branch is always
    detached and therefore never receives gradients from this loss.
    """
    if student_real_log_probs.ndim != 2:
        raise ValueError("Student log probabilities must have shape [batch, positions]")
    if student_wrong_log_probs.shape != student_real_log_probs.shape:
        raise ValueError("Student Real/Wrong log probabilities differ in shape")
    if teacher_log_ratios.shape != student_real_log_probs.shape:
        raise ValueError("Teacher ratios and Student log probabilities differ in shape")
    if not torch.isfinite(student_real_log_probs).all():
        raise FloatingPointError("non-finite Student-Real log probability")
    if not torch.isfinite(student_wrong_log_probs).all():
        raise FloatingPointError("non-finite Student-Wrong log probability")

    eligible_tokens, token_margins, _ = strict_aa_word_contrast_targets(
        teacher_log_ratios,
        selected_mask,
        support_mask,
        token_word_ids,
        margin_cap=margin_cap,
    )
    real = student_real_log_probs.float()
    wrong = student_wrong_log_probs.detach().float()
    row_losses: list[torch.Tensor] = []

    for row_index, row_word_ids in enumerate(token_word_ids):
        eligible_words = sorted(
            {
                word_id
                for token_index, word_id in enumerate(row_word_ids)
                if word_id is not None and bool(eligible_tokens[row_index, token_index])
            }
        )
        word_losses: list[torch.Tensor] = []
        for word_id in eligible_words:
            indices = [
                token_index
                for token_index, candidate in enumerate(row_word_ids)
                if candidate == word_id
            ]
            index = torch.tensor(indices, dtype=torch.long, device=real.device)
            student_gap = real[row_index].index_select(0, index).mean() - wrong[
                row_index
            ].index_select(0, index).mean()
            margin = token_margins[row_index, indices[0]]
            word_losses.append(torch.relu(margin - student_gap))
        if word_losses:
            row_losses.append(torch.stack(word_losses).mean())
        else:
            row_losses.append(real[row_index].sum() * 0.0)
    return torch.stack(row_losses)


def teacher_distribution_contrastive_rows(
    real_teacher_distance: torch.Tensor,
    wrong_teacher_distance: torch.Tensor,
    teacher_log_ratios: torch.Tensor,
    selected_mask: torch.Tensor,
    support_mask: torch.Tensor,
    token_word_ids: Sequence[Sequence[int | None]],
    *,
    margin: float = 0.02,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Rank Teacher-Real closer than Teacher-Wrong on strict AA words.

    Distances are per-position KL values sharing the same Student-Real anchor.
    Each eligible word contributes ``relu(D_real - D_wrong + margin)`` and
    words receive equal weight regardless of their BPE length.  Temperature
    is frozen to one and retained only so all contrast manifests expose a
    common, fail-closed parameter surface.
    """
    if real_teacher_distance.ndim != 2:
        raise ValueError("Teacher distances must have shape [batch, positions]")
    if wrong_teacher_distance.shape != real_teacher_distance.shape:
        raise ValueError("Teacher Real/Wrong distances differ in shape")
    if teacher_log_ratios.shape != real_teacher_distance.shape:
        raise ValueError("Teacher ratios and distances differ in shape")
    if not math.isfinite(float(margin)) or margin < 0:
        raise ValueError("distribution contrast margin must be non-negative")
    if not math.isclose(float(temperature), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("hinge distribution contrast freezes temperature at 1")
    if not torch.isfinite(real_teacher_distance).all() or not torch.isfinite(
        wrong_teacher_distance
    ).all():
        raise FloatingPointError("non-finite Teacher contrast distance")

    eligible_tokens, _, _ = strict_aa_word_contrast_targets(
        teacher_log_ratios,
        selected_mask,
        support_mask,
        token_word_ids,
        margin_cap=0.10,
    )
    real = real_teacher_distance.float()
    wrong = wrong_teacher_distance.detach().float()
    row_losses: list[torch.Tensor] = []
    for row_index, row_word_ids in enumerate(token_word_ids):
        eligible_words = sorted(
            {
                word_id
                for token_index, word_id in enumerate(row_word_ids)
                if word_id is not None and bool(eligible_tokens[row_index, token_index])
            }
        )
        word_losses: list[torch.Tensor] = []
        for word_id in eligible_words:
            indices = [
                token_index
                for token_index, candidate in enumerate(row_word_ids)
                if candidate == word_id
            ]
            index = torch.tensor(indices, dtype=torch.long, device=real.device)
            ranking_violation = (
                real[row_index].index_select(0, index).mean()
                - wrong[row_index].index_select(0, index).mean()
                + float(margin)
            )
            word_losses.append(torch.relu(ranking_violation))
        if word_losses:
            row_losses.append(torch.stack(word_losses).mean())
        else:
            row_losses.append(real[row_index].sum() * 0.0)
    return torch.stack(row_losses)


def combined_v2_loss(
    direct_ce_rows: torch.Tensor,
    rkl_rows: torch.Tensor | None,
    *,
    lambda_opd: float = 0.25,
) -> torch.Tensor:
    if direct_ce_rows.ndim != 1:
        raise ValueError("direct CE must contain one scalar per row")
    if rkl_rows is None:
        return direct_ce_rows.mean()
    if rkl_rows.shape != direct_ce_rows.shape:
        raise ValueError("direct CE and RKL row dimensions differ")
    return (direct_ce_rows + float(lambda_opd) * rkl_rows).mean()
