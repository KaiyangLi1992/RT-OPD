"""Isolated PAC-2S primitives for the K3B six-dimensional BO campaign.

The sixth axis (Student learning rate) belongs to the launcher.  This module
owns the other five candidate axes without widening or mutating the frozen
Round-12 trainer:

``beta, top_fraction, ratio_gate, tau, ramp_end``.

At the historical default point ``.025/.30/2/1/80`` the selector and paired
logistic objective delegate to the already-audited implementations.  The
ratio gate is reproduced in this namespace because the legacy helper freezes
its cutoff at ``log(2)``.  Its float64 cutoff/boundary and complete-word
semantics intentionally match that helper exactly when ``ratio_gate == 2``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import torch

from .contrast_round2 import paired_audio_condition_nce_rows
from .losses import joint_top_fraction_word_weights


BO6D_PAC_CONFIG_VERSION: Final = "k3b_pac_bo6d_config_v1"
BO6D_PAC_SELECTOR_VERSION: Final = "joint_top_fraction_parameterized_v1"
BO6D_PAC_GATE_VERSION: Final = (
    "strict_aa_complete_word_joint_ratio_parameterized_v2"
)
BO6D_PAC_OBJECTIVE_VERSION: Final = "paired_audio_condition_nce_v1"

BO6D_TOP_FRACTIONS: Final = (0.15, 0.20, 0.25, 0.30)
BO6D_RATIO_GATES: Final = (1.25, 1.50, 2.00, 2.50, 3.00)
BO6D_RAMP_ENDS: Final = (48, 64, 80, 96, 112)
BO6D_BETA_BOUNDS: Final = (0.00625, 0.050)
BO6D_TAU_BOUNDS: Final = (0.50, 2.00)
BO6D_RAMP_START: Final = 32
BO6D_AA_HIGH_WEIGHT: Final = 2.0
BO6D_MARGIN_CAP: Final = 0.10


def _finite_float(value: object, label: str) -> float:
    # Candidate manifests are scientific evidence, so numeric-looking strings
    # and booleans must not be silently coerced into accepted coordinates.
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _canonical_discrete(value: float, allowed: Sequence[float], label: str) -> float:
    for candidate in allowed:
        if math.isclose(value, candidate, rel_tol=0.0, abs_tol=1e-15):
            return float(candidate)
    raise ValueError(f"{label} must be one of {tuple(allowed)}")


@dataclass(frozen=True)
class BO6DPACConfig:
    """Fail-closed five-axis PAC configuration for one BO candidate.

    A disabled, ``beta=0`` configuration is admitted for the later matched
    PAC-off confirmation.  It is not an observation in the positive-beta BO
    search space.  PAC-2S, high-weight 2, margin cap .10, and ramp start 32
    remain frozen experimental invariants rather than hidden search axes.
    """

    beta: float = 0.025
    top_fraction: float = 0.30
    ratio_gate: float = 2.0
    tau: float = 1.0
    ramp_end: int = 80
    enabled: bool = True

    @property
    def ramp_start(self) -> int:
        """Frozen campaign ramp start, exposed for trainer compatibility."""

        return BO6D_RAMP_START

    @property
    def wrong_gradient_mode(self) -> str:
        """The BO campaign searches PAC-2S only."""

        return "bidirectional"

    @property
    def engineering_calibration(self) -> bool:
        """BO observations are formal trajectories, never calibration probes."""

        return False

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be an exact boolean")

        beta = _finite_float(self.beta, "beta")
        top_fraction = _canonical_discrete(
            _finite_float(self.top_fraction, "top_fraction"),
            BO6D_TOP_FRACTIONS,
            "top_fraction",
        )
        ratio_gate = _canonical_discrete(
            _finite_float(self.ratio_gate, "ratio_gate"),
            BO6D_RATIO_GATES,
            "ratio_gate",
        )
        tau = _finite_float(self.tau, "tau")
        if not BO6D_TAU_BOUNDS[0] <= tau <= BO6D_TAU_BOUNDS[1]:
            raise ValueError(f"tau must be within {BO6D_TAU_BOUNDS}")

        if isinstance(self.ramp_end, bool) or not isinstance(self.ramp_end, int):
            raise ValueError("ramp_end must be an integer BO level")
        if self.ramp_end not in BO6D_RAMP_ENDS:
            raise ValueError(f"ramp_end must be one of {BO6D_RAMP_ENDS}")

        if self.enabled:
            if not BO6D_BETA_BOUNDS[0] <= beta <= BO6D_BETA_BOUNDS[1]:
                raise ValueError(
                    f"enabled beta must be within {BO6D_BETA_BOUNDS}"
                )
        elif beta != 0.0:
            raise ValueError("disabled PAC confirmation control requires beta=0")

        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "top_fraction", top_fraction)
        object.__setattr__(self, "ratio_gate", ratio_gate)
        object.__setattr__(self, "tau", tau)

    @classmethod
    def from_manifest_axes(
        cls, axes: Mapping[str, object], *, enabled: bool = True
    ) -> "BO6DPACConfig":
        """Parse exactly the five non-LR axes from a candidate manifest."""

        expected = {"beta", "top_fraction", "ratio_gate", "tau", "ramp_end"}
        observed = set(axes)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"BO6D PAC axes changed; missing={missing}, extra={extra}"
            )
        ramp_end = axes["ramp_end"]
        if isinstance(ramp_end, bool) or not isinstance(ramp_end, int):
            raise ValueError("ramp_end must be an integer BO level")
        return cls(
            beta=_finite_float(axes["beta"], "beta"),
            top_fraction=_finite_float(axes["top_fraction"], "top_fraction"),
            ratio_gate=_finite_float(axes["ratio_gate"], "ratio_gate"),
            tau=_finite_float(axes["tau"], "tau"),
            ramp_end=ramp_end,
            enabled=enabled,
        )

    def manifest_axes(self) -> dict[str, float | int]:
        """Return the canonical five-axis fragment; LR remains launcher-owned."""

        return {
            "beta": float(self.beta),
            "top_fraction": float(self.top_fraction),
            "ratio_gate": float(self.ratio_gate),
            "tau": float(self.tau),
            "ramp_end": int(self.ramp_end),
        }

    def effective_beta(self, global_step: int) -> float:
        """Coefficient before the next update, with the frozen start at 32."""

        if isinstance(global_step, bool) or not isinstance(global_step, int):
            raise ValueError("global_step must be a non-negative integer")
        if global_step < 0:
            raise ValueError("global_step must be a non-negative integer")
        if not self.enabled or global_step < BO6D_RAMP_START:
            return 0.0
        if global_step >= self.ramp_end:
            return float(self.beta)
        progress = (global_step - BO6D_RAMP_START) / (
            self.ramp_end - BO6D_RAMP_START
        )
        return float(self.beta) * progress

    def audit(self) -> dict[str, Any]:
        return {
            "bo6d_pac_config_version": BO6D_PAC_CONFIG_VERSION,
            "bo6d_pac_selector_version": BO6D_PAC_SELECTOR_VERSION,
            "bo6d_pac_gate_version": BO6D_PAC_GATE_VERSION,
            "bo6d_pac_objective_version": BO6D_PAC_OBJECTIVE_VERSION,
            "bo6d_pac_enabled": bool(self.enabled),
            "bo6d_pac_axes": self.manifest_axes(),
            "bo6d_lr_owner": "launcher_manifest",
            "bo6d_pac_wrong_gradient_mode": "bidirectional",
            "bo6d_pac_ramp_start": BO6D_RAMP_START,
            "bo6d_aa_high_weight": BO6D_AA_HIGH_WEIGHT,
            "bo6d_margin_cap": BO6D_MARGIN_CAP,
        }


@dataclass(frozen=True)
class BO6DPACExecutionDecision:
    """Decision to materialize the extra Student-Wrong forward."""

    execute_wrong: bool
    effective_beta: float
    eligible_tokens: int
    skip_reason: str

    def audit(self) -> dict[str, object]:
        # Preserve the legacy audit names so downstream aggregation need not
        # reinterpret default-point runs.
        return {
            "paired_nce_beta_effective": float(self.effective_beta),
            "paired_nce_eligible_tokens": float(self.eligible_tokens),
            "paired_nce_wrong_forward_executed": float(self.execute_wrong),
            "paired_nce_skip_reason": self.skip_reason,
        }


def decide_bo6d_pac_execution(
    config: BO6DPACConfig,
    *,
    global_step: int,
    eligible_mask: torch.Tensor,
) -> BO6DPACExecutionDecision:
    """Skip PAC-off, zero-ramp, and empty-gate Wrong forwards explicitly."""

    if eligible_mask.dtype != torch.bool or eligible_mask.ndim != 2:
        raise ValueError("BO6D PAC eligible mask must be boolean [batch, positions]")
    eligible_tokens = int(eligible_mask.sum().item())
    effective_beta = config.effective_beta(global_step)
    if not config.enabled:
        return BO6DPACExecutionDecision(
            False, effective_beta, eligible_tokens, "control_beta_zero"
        )
    if effective_beta == 0.0:
        return BO6DPACExecutionDecision(
            False, effective_beta, eligible_tokens, "ramp_beta_zero"
        )
    if eligible_tokens == 0:
        return BO6DPACExecutionDecision(
            False, effective_beta, eligible_tokens, "empty_complete_word_gate"
        )
    return BO6DPACExecutionDecision(
        True, effective_beta, eligible_tokens, "executed"
    )


def bo6d_joint_top_fraction_word_weights(
    token_log_ratios: Sequence[float],
    token_support: Sequence[bool],
    token_word_ids: Sequence[int | None],
    *,
    config: BO6DPACConfig,
) -> list[float]:
    """Run the audited Word-Joint selector at the candidate Top fraction."""

    return joint_top_fraction_word_weights(
        token_log_ratios,
        token_support,
        token_word_ids,
        high_weight=BO6D_AA_HIGH_WEIGHT,
        top_fraction=config.top_fraction,
    )


def bo6d_strict_aa_word_contrast_targets(
    teacher_log_ratios: torch.Tensor,
    selected_mask: torch.Tensor,
    support_mask: torch.Tensor,
    token_word_ids: Sequence[Sequence[int | None]],
    *,
    config: BO6DPACConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    """Build complete-word eligibility using the candidate ratio gate.

    The legacy fixed ``ratio>2`` count remains as a diagnostic, while
    ``contrast_joint_ratio_gate_words`` records the population at the actual
    candidate cutoff.  At ``ratio_gate=2`` the returned mask and margins are
    numerically identical to ``strict_aa_word_contrast_targets``.
    """

    if teacher_log_ratios.ndim != 2:
        raise ValueError("teacher log ratios must have shape [batch, positions]")
    if selected_mask.shape != teacher_log_ratios.shape:
        raise ValueError("selected mask and teacher log ratios differ in shape")
    if support_mask.shape != teacher_log_ratios.shape:
        raise ValueError("support mask and teacher log ratios differ in shape")
    if selected_mask.dtype != torch.bool or support_mask.dtype != torch.bool:
        raise ValueError("selected and support masks must be boolean tensors")

    batch_size, position_count = teacher_log_ratios.shape
    if len(token_word_ids) != batch_size:
        raise ValueError("word-ID rows do not match the tensor batch dimension")
    for row_word_ids in token_word_ids:
        if len(row_word_ids) != position_count:
            raise ValueError(
                "word-ID row does not match the tensor position dimension"
            )

    ratios = teacher_log_ratios.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(ratios).all().item()):
        raise FloatingPointError("non-finite Teacher Real/Wrong log ratio")
    eligible_tokens = torch.zeros_like(selected_mask, dtype=torch.bool)
    token_margins = torch.zeros_like(teacher_log_ratios, dtype=torch.float32)
    row_audits: list[dict[str, float]] = []
    configured_cutoff = math.log(float(config.ratio_gate))
    legacy_cutoff = math.log(2.0)

    for row_index, row_word_ids in enumerate(token_word_ids):
        by_word_indices: dict[int, list[int]] = {}
        for token_index, word_id in enumerate(row_word_ids):
            if word_id is not None:
                by_word_indices.setdefault(word_id, []).append(token_index)

        complete_selected_words = 0
        complete_supported_words = 0
        joint_ratio_gt2_words = 0
        joint_ratio_gate_words = 0
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
            joint_value = float(joint_log_ratio.item())
            joint_ratio_gt2 = joint_value > legacy_cutoff
            joint_ratio_gate = joint_value > configured_cutoff
            complete_selected_words += int(complete_selected)
            complete_supported_words += int(complete_supported)
            joint_ratio_gt2_words += int(joint_ratio_gt2)
            joint_ratio_gate_words += int(joint_ratio_gate)
            if not (
                complete_selected and complete_supported and joint_ratio_gate
            ):
                continue
            margin = min(BO6D_MARGIN_CAP, joint_value / len(indices))
            eligible_tokens[row_index, index] = True
            token_margins[row_index, index] = margin
            eligible_words += 1
            eligible_margins.append(margin)

        row_audits.append(
            {
                "contrast_lexical_words": float(len(by_word_indices)),
                "contrast_complete_selected_words": float(
                    complete_selected_words
                ),
                "contrast_complete_supported_words": float(
                    complete_supported_words
                ),
                "contrast_joint_ratio_gt2_words": float(joint_ratio_gt2_words),
                "contrast_joint_ratio_gate_words": float(
                    joint_ratio_gate_words
                ),
                "contrast_ratio_gate": float(config.ratio_gate),
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


def bo6d_paired_audio_condition_nce_rows(
    student_real_log_probs: torch.Tensor,
    student_wrong_log_probs: torch.Tensor,
    eligible_mask: torch.Tensor,
    token_word_ids: Sequence[Sequence[int | None]],
    *,
    config: BO6DPACConfig,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Run the existing stable PAC-2S objective at the candidate temperature."""

    return paired_audio_condition_nce_rows(
        student_real_log_probs,
        student_wrong_log_probs,
        eligible_mask,
        token_word_ids,
        tau=config.tau,
    )


__all__ = [
    "BO6D_AA_HIGH_WEIGHT",
    "BO6D_BETA_BOUNDS",
    "BO6D_MARGIN_CAP",
    "BO6D_PAC_CONFIG_VERSION",
    "BO6D_PAC_GATE_VERSION",
    "BO6D_PAC_OBJECTIVE_VERSION",
    "BO6D_PAC_SELECTOR_VERSION",
    "BO6D_RATIO_GATES",
    "BO6D_RAMP_ENDS",
    "BO6D_RAMP_START",
    "BO6D_TAU_BOUNDS",
    "BO6D_TOP_FRACTIONS",
    "BO6DPACConfig",
    "BO6DPACExecutionDecision",
    "bo6d_joint_top_fraction_word_weights",
    "bo6d_paired_audio_condition_nce_rows",
    "bo6d_strict_aa_word_contrast_targets",
    "decide_bo6d_pac_execution",
]
