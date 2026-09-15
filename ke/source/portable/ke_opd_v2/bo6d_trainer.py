"""Versioned PAC-2S trainer for the K3B 50-run six-dimensional BO campaign.

This module is intentionally separate from ``trainer_round2.py``.  It reuses
that trainer's audited paired-forward/RNG machinery, while owning the new
candidate-dependent selector, ratio gate, temperature, beta, and ramp path.
The legacy Round-12 class is neither modified nor monkey-patched here.
"""

from __future__ import annotations

import math
from typing import Any, Final

import torch

from .bo6d_pac import (
    BO6D_AA_HIGH_WEIGHT,
    BO6D_MARGIN_CAP,
    BO6D_PAC_GATE_VERSION,
    BO6D_PAC_OBJECTIVE_VERSION,
    BO6DPACConfig,
    bo6d_joint_top_fraction_word_weights,
    bo6d_strict_aa_word_contrast_targets,
    decide_bo6d_pac_execution,
)
from .losses import (
    reverse_kl_per_position,
    sampled_token_audio_log_ratio,
    weighted_position_mean,
)
from .modeling import (
    append_completion,
    completion_word_ids,
    load_audio,
    match_waveform_length,
    prepare_prompt_inputs,
)
from .trainer import KeOPDV2Trainer
from .trainer_round2 import (
    PAC_NCE_REDUCTION_VERSION,
    PACCalibrationRowCapture,
    PairedNCEResult,
    KeOPDRound2Trainer,
    _paired_input_contract,
    pac_nce_from_paired_forward,
    paired_student_condition_forward,
    validate_round2_teacher_contract,
)


BO6D_TRAINER_VERSION: Final = "k3b_pac2s_bo6d_trainer_v1"


def validate_bo6d_aa_contract(
    contract: Any, config: BO6DPACConfig
) -> dict[str, bool]:
    """Freeze all inherited AA/OPD fields except candidate Top fraction."""

    checks = {
        "aa_top_fraction_matches_candidate": math.isclose(
            float(contract.aa_top_fraction),
            float(config.top_fraction),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "aa_high_weight_2p0": math.isclose(
            float(contract.aa_high_weight),
            BO6D_AA_HIGH_WEIGHT,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "lambda_opd_0p25": math.isclose(
            float(contract.lambda_opd), 0.25, rel_tol=0.0, abs_tol=1e-15
        ),
        "kl_temperature_1p0": math.isclose(
            float(contract.kl_temperature), 1.0, rel_tol=0.0, abs_tol=1e-15
        ),
        "teacher_support_top_k_128": int(contract.teacher_support_top_k) == 128,
    }
    if not all(checks.values()):
        raise ValueError(f"BO6D PAC-2S AA/OPD contract changed: {checks}")
    return checks


class KeOPDBO6DPACTrainerV1(KeOPDRound2Trainer):
    """Formal PAC-2S trainer whose five non-LR axes come from one manifest."""

    PAC_WRONG_GRADIENT_MODE: Final = "bidirectional"

    def __init__(
        self,
        *args: Any,
        bo6d_config: BO6DPACConfig,
        **kwargs: Any,
    ) -> None:
        for reserved in (
            "contrast_mode",
            "contrast_beta",
            "contrast_margin",
            "contrast_teacher_ratio_threshold",
            "contrast_temperature",
            "contrast_ramp_start",
            "contrast_ramp_end",
        ):
            if reserved in kwargs:
                raise ValueError(
                    f"BO6D trainer owns {reserved}; candidate manifest is authoritative"
                )
        if kwargs.get("audit_every_microbatch") is not True:
            raise ValueError("formal BO6D trainer requires every-microbatch audit")

        # Call the stable generic trainer directly.  The Round-2 initializer
        # freezes Top=.30/tau=1/ramp=80 and therefore must not mediate BO axes.
        KeOPDV2Trainer.__init__(
            self,
            *args,
            contrast_mode="none",
            contrast_beta=0.0,
            contrast_margin=BO6D_MARGIN_CAP,
            # The generic legacy initializer freezes this unused Round-1 knob
            # at 2.  BO eligibility uses bo6d_config.ratio_gate below.
            contrast_teacher_ratio_threshold=2.0,
            contrast_temperature=bo6d_config.tau,
            contrast_ramp_start=bo6d_config.ramp_start,
            contrast_ramp_end=bo6d_config.ramp_end,
            **kwargs,
        )
        if self.v2_arm != "AA":
            raise ValueError("BO6D PAC-2S requires the AA arm")
        if self.v2_aa_selector != "joint_top_fraction":
            raise ValueError("BO6D PAC-2S requires Word-Joint Top-Fraction AA")
        if not bo6d_config.enabled:
            raise ValueError("50-run BO observations require enabled positive-beta PAC")

        self.v2_bo6d_aa_contract_checks = validate_bo6d_aa_contract(
            self.v2_contract, bo6d_config
        )
        self.v2_round2_teacher_contract_checks = validate_round2_teacher_contract(
            self.model, self.v2_teacher
        )
        for sample_id, gate_value in self.v2_teacher_gate.items():
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("BO6D Teacher gate IDs must be non-empty strings")
            if type(gate_value) is not bool:
                raise ValueError(
                    "BO6D Teacher gate values must be exact booleans; "
                    f"id={sample_id!r}"
                )

        wrong_gradient_mode = str(type(self).PAC_WRONG_GRADIENT_MODE)
        if wrong_gradient_mode not in {"bidirectional", "detach_wrong"}:
            raise ValueError(
                "BO6D PAC wrong-gradient mode must be bidirectional or detach_wrong"
            )
        self.v2_pac_nce_config = bo6d_config
        self.v2_bo6d_config = bo6d_config
        self.v2_pac_wrong_gradient_mode = wrong_gradient_mode
        self._v2_calibration_rollout_provider = None
        self._v2_calibration_current_sample_id: str | None = None
        self.v2_contrast_mode = BO6D_PAC_OBJECTIVE_VERSION
        self.v2_contrast_beta = float(bo6d_config.beta)
        self.v2_contrast_temperature = float(bo6d_config.tau)
        self.v2_contrast_ramp_start = int(bo6d_config.ramp_start)
        self.v2_contrast_ramp_end = int(bo6d_config.ramp_end)
        self.v2_aa_ratio_threshold = float(bo6d_config.ratio_gate)
        self._v2_pac_calibration_rows: list[PACCalibrationRowCapture] | None = None
        self._v2_pac_gradient_audited_steps: set[int] = set()
        self._v2_pac_gradient_pending_steps: set[int] = set()

    def _effective_contrast_beta(self) -> float:
        return self.v2_bo6d_config.effective_beta(int(self.state.global_step))

    def _opd_loss(
        self,
        model: torch.nn.Module,
        row: dict[str, Any],
        real_waveform: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Compute matched AA-RKLD plus parameterized PAC-2S for one row."""

        config = self.v2_bo6d_config
        real_prompt = prepare_prompt_inputs(
            self.processor,
            row["canonical_user_text"],
            real_waveform,
            self.v2_sampling_rate,
            self.args.device,
        )
        rollout = self._generate_rollout(model, real_prompt)
        real_sequence = append_completion(real_prompt, rollout)
        prompt_length = int(real_prompt["input_ids"].shape[1])
        logit_start = prompt_length - 1
        logit_stop = logit_start + int(rollout.shape[1])

        wrong_waveform = load_audio(row["wrong_audio_path"], self.v2_sampling_rate)
        wrong_waveform = match_waveform_length(wrong_waveform, len(real_waveform))
        wrong_prompt = prepare_prompt_inputs(
            self.processor,
            row["canonical_user_text"],
            wrong_waveform,
            self.v2_sampling_rate,
            self.args.device,
        )
        wrong_sequence = append_completion(wrong_prompt, rollout)
        prefix_equal, same_rollout, _ = _paired_input_contract(
            model,
            real_sequence,
            wrong_sequence,
            prompt_length=prompt_length,
        )

        teacher_contract = validate_round2_teacher_contract(model, self.v2_teacher)
        assert self.v2_teacher is not None
        with torch.inference_mode():
            teacher_real = self.v2_teacher(
                **real_sequence, use_cache=False, return_dict=True
            ).logits[:, logit_start:logit_stop].clone()
            teacher_wrong = self.v2_teacher(
                **wrong_sequence, use_cache=False, return_dict=True
            ).logits[:, logit_start:logit_stop].clone()
        if not bool(torch.isfinite(teacher_real).all().item()):
            raise FloatingPointError("non-finite BO6D Teacher-Real logits")
        if not bool(torch.isfinite(teacher_wrong).all().item()):
            raise FloatingPointError("non-finite BO6D Teacher-Wrong logits")

        word_ids = completion_word_ids(
            self.processor.tokenizer, rollout[0].tolist()
        )
        log_ratios, support = sampled_token_audio_log_ratio(
            teacher_real,
            teacher_wrong,
            rollout,
            self.v2_valid_ids,
            support_top_k=self.v2_contract.teacher_support_top_k,
        )
        raw_weights = bo6d_joint_top_fraction_word_weights(
            log_ratios[0].tolist(),
            support[0].tolist(),
            word_ids,
            config=config,
        )
        advantages = torch.where(support, log_ratios, torch.zeros_like(log_ratios))
        weights = torch.tensor(
            raw_weights, dtype=torch.float32, device=log_ratios.device
        ).unsqueeze(0)
        selected_mask = weights.gt(1)
        eligible_mask, _, gate_audits = bo6d_strict_aa_word_contrast_targets(
            log_ratios,
            selected_mask,
            support,
            [word_ids],
            config=config,
        )
        decision = decide_bo6d_pac_execution(
            config,
            global_step=int(self.state.global_step),
            eligible_mask=eligible_mask,
        )
        del teacher_wrong

        paired = paired_student_condition_forward(
            model,
            real_sequence,
            wrong_sequence,
            prompt_length=prompt_length,
            logit_start=logit_start,
            logit_stop=logit_stop,
            execute_wrong=decision.execute_wrong,
            wrong_requires_grad=(
                self.v2_pac_wrong_gradient_mode == "bidirectional"
            ),
            get_rng_state=self._rng_state,
            set_rng_state=self._set_rng_state,
        )
        rkl = reverse_kl_per_position(
            paired.real_logits,
            teacher_real,
            self.v2_valid_ids,
            temperature=self.v2_contract.kl_temperature,
        )
        if not bool(torch.isfinite(rkl).all().item()):
            raise FloatingPointError("non-finite BO6D AA-RKLD")
        del teacher_real

        contrast = paired.real_logits.sum().reshape(1) * 0.0
        captured_nce: PairedNCEResult | None = None
        paired_stats: dict[str, float | None] = {
            "paired_nce_eligible_words": float(
                gate_audits[0]["contrast_eligible_words"]
            ),
            "paired_nce_eligible_tokens": float(
                gate_audits[0]["contrast_eligible_tokens"]
            ),
            "paired_nce_mean_gap": None,
            "paired_nce_mean_prob": None,
            "paired_nce_mean_loss": None,
            "paired_nce_mean_gradient_factor": None,
            "paired_nce_saturation_fraction_preal_ge_0p95": None,
            "paired_nce_wrong_preferred_fraction": None,
        }
        if decision.execute_wrong:
            nce = pac_nce_from_paired_forward(
                paired,
                rollout,
                self.v2_valid_ids,
                eligible_mask,
                [word_ids],
                tau=config.tau,
                detach_wrong=(
                    self.v2_pac_wrong_gradient_mode == "detach_wrong"
                ),
            )
            contrast = nce.losses
            captured_nce = nce
            self._register_pac_gradient_audit(
                sample_id=str(row["id"]), result=nce
            )
            nce_audit = nce.row_audits[0]
            if (
                nce_audit["paired_nce_eligible_words"]
                != paired_stats["paired_nce_eligible_words"]
                or nce_audit["paired_nce_eligible_tokens"]
                != paired_stats["paired_nce_eligible_tokens"]
            ):
                raise AssertionError(
                    "BO6D objective and Teacher gate populations differ"
                )
            paired_stats.update(nce_audit)
        if not bool(torch.isfinite(contrast).all().item()):
            raise FloatingPointError("non-finite BO6D paired NCE")

        calibration_rows = getattr(self, "_v2_pac_calibration_rows", None)
        if calibration_rows is not None:
            calibration_rows.append(
                PACCalibrationRowCapture(
                    sample_id=str(row["id"]),
                    teacher_correct=True,
                    rollout_token_ids=tuple(
                        int(value) for value in rollout[0].tolist()
                    ),
                    complete_word_mask=eligible_mask,
                    token_word_ids=tuple(word_ids),
                    paired_result=captured_nce,
                    wrong_graph_executed=bool(decision.execute_wrong),
                )
            )

        by_word_advantages: dict[int, list[float]] = {}
        by_word_weights: dict[int, set[float]] = {}
        for word_id, advantage, weight in zip(
            word_ids, advantages[0].tolist(), raw_weights
        ):
            if word_id is None:
                continue
            by_word_advantages.setdefault(word_id, []).append(float(advantage))
            by_word_weights.setdefault(word_id, set()).add(float(weight))
        word_scores = {
            word_id: sum(values) / len(values)
            for word_id, values in by_word_advantages.items()
        }
        positive_words = {
            word_id for word_id, score in word_scores.items() if score > 0
        }
        selected_words = {
            word_id
            for word_id, word_weights in by_word_weights.items()
            if any(weight > 1 for weight in word_weights)
        }
        partial_word_weight_mismatch = sum(
            len(word_weights) != 1 for word_weights in by_word_weights.values()
        )
        lexical = torch.tensor(
            [[word_id is not None for word_id in word_ids]],
            dtype=torch.bool,
            device=advantages.device,
        )
        positive_lexical = advantages.gt(0) & lexical
        supported_lexical = support & lexical
        lexical_tokens = int(lexical.sum().item())
        selected_tokens = int(weights.gt(1).sum().item())
        lexical_words = len(word_scores)
        top_limit = (
            max(1, math.ceil(lexical_words * config.top_fraction))
            if lexical_words
            else 0
        )
        rollout_text = self.processor.tokenizer.decode(
            rollout[0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        metrics: dict[str, Any] = {
            **gate_audits[0],
            **decision.audit(),
            **paired.audit,
            **paired_stats,
            **config.audit(),
            "bo6d_pac_wrong_gradient_mode": self.v2_pac_wrong_gradient_mode,
            "bo6d_trainer_version": BO6D_TRAINER_VERSION,
            "contrast_objective_version": BO6D_PAC_OBJECTIVE_VERSION,
            "paired_nce_gate_version": BO6D_PAC_GATE_VERSION,
            "paired_nce_enabled": 1.0,
            "paired_nce_tau": float(config.tau),
            "paired_nce_wrong_gradient_mode": self.v2_pac_wrong_gradient_mode,
            "paired_nce_beta_target": float(config.beta),
            "paired_nce_engineering_calibration": 0.0,
            "paired_nce_computed": float(decision.execute_wrong),
            "paired_nce_statistics_defined": float(decision.execute_wrong),
            "paired_nce_reduction_version": PAC_NCE_REDUCTION_VERSION,
            "paired_nce_reduction_semantics": (
                "equal_complete_words_within_row_then_all_local_microbatch_rows;"
                "empty_gate_and_teacher_false_rows_are_zero"
            ),
            "paired_nce_objective_row_computed": float(decision.execute_wrong),
            "contrast_eligible_row": float(eligible_mask.any().item()),
            "contrast_active_row": float(decision.execute_wrong),
            "contrast_active_row_deprecated": 1.0,
            "contrast_active_row_semantics": (
                "deprecated_alias_of_paired_nce_objective_row_computed_not_hinge_activity"
            ),
            "contrast_loss": float(contrast.detach().item()),
            "contrast_mode": BO6D_PAC_OBJECTIVE_VERSION,
            "paired_prefix_equal": float(prefix_equal),
            "paired_same_rollout": float(same_rollout),
            "rollout_tokens": float(rollout.numel()),
            "rollout_complete_think": float(
                "<think>" in rollout_text.lower()
                and "</think>" in rollout_text.lower()
            ),
            "rollout_has_answer": float("<answer" in rollout_text.lower()),
            "rollout_preview": rollout_text[:240],
            "aa_high_tokens": float(selected_tokens),
            "aa_high_words": float(len(selected_words)),
            "aa_lexical_words": float(lexical_words),
            "aa_positive_words": float(len(positive_words)),
            "aa_top_fraction": float(config.top_fraction),
            "aa_top_limit": float(top_limit),
            "aa_selected_words": float(len(selected_words)),
            "aa_selected_tokens": float(selected_tokens),
            "aa_zero_high_rows": float(selected_tokens == 0),
            "aa_high_bpe": float(selected_tokens),
            "answer_high_bpe": float(
                sum(
                    word_id is None and weight > 1
                    for word_id, weight in zip(word_ids, raw_weights)
                )
            ),
            "partial_word_weight_mismatch": float(partial_word_weight_mismatch),
            "aa_selector": "joint_top_fraction",
            "aa_selection_unit": "word",
            "aa_top_denominator_units": float(lexical_words),
            "aa_ratio_threshold": float(config.ratio_gate),
            "aa_lexical_tokens": float(lexical_tokens),
            "aa_positive_tokens": float(positive_lexical.sum().item()),
            "support_tokens": float(supported_lexical.sum().item()),
            "support_fraction": float(
                supported_lexical.sum().item() / max(lexical_tokens, 1)
            ),
            "aa_high_token_fraction": float(
                selected_tokens / max(lexical_tokens, 1)
            ),
            "mean_positive_aa": float(
                advantages[positive_lexical].mean().item()
                if positive_lexical.any()
                else 0.0
            ),
            "paired_teacher_distinct_student": float(
                bool(teacher_contract["paired_teacher_distinct_student"])
            ),
            "paired_teacher_all_parameters_frozen": float(
                bool(teacher_contract["paired_teacher_all_parameters_frozen"])
            ),
            "paired_teacher_recursive_eval": float(
                bool(teacher_contract["paired_teacher_recursive_eval"])
            ),
        }
        mask = torch.ones_like(rkl, dtype=torch.bool)
        opd = weighted_position_mean(rkl, weights, mask)
        if not bool(torch.isfinite(opd).all().item()):
            raise FloatingPointError("non-finite BO6D weighted AA-RKLD")
        return opd, contrast, metrics


__all__ = [
    "BO6D_TRAINER_VERSION",
    "KeOPDBO6DPACTrainerV1",
    "validate_bo6d_aa_contract",
]
