"""Objective-isolated trainers for the V1.0 Rounded-X15 seed-50 ablation."""

from __future__ import annotations

import json
import math
from typing import Any, Final, Mapping

import torch

from .bo6d_pac import (
    BO6D_PAC_GATE_VERSION,
    BO6D_PAC_OBJECTIVE_VERSION,
    BO6DPACConfig,
    bo6d_joint_top_fraction_word_weights,
    bo6d_strict_aa_word_contrast_targets,
    decide_bo6d_pac_execution,
)
from .bo6d_trainer import KeOPDBO6DPACTrainerV1
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
from .phaseb_pac_all import (
    PHASEB_PAC_ALL_ELIGIBILITY_VERSION,
    pac_all_complete_word_targets,
    sampled_token_teacher_real_support,
)
from .trainer import KeOPDV2Trainer
from .trainer_round2 import (
    PAC_NCE_REDUCTION_VERSION,
    PACCalibrationRowCapture,
    PairedNCEResult,
    _paired_input_contract,
    pac_nce_from_paired_forward,
    paired_student_condition_forward,
    validate_round2_teacher_contract,
)


TRAINER_VERSION: Final = "ke_opd_v10_rounded_x15_seed50_ablation_trainers_v1"
PAC_ALL_OBJECTIVE_VERSION: Final = "paired_audio_condition_nce_v1"


_GENERIC_OWNED = (
    "arm",
    "aa_selector",
    "contrast_mode",
    "contrast_beta",
    "contrast_margin",
    "contrast_teacher_ratio_threshold",
    "contrast_temperature",
    "contrast_ramp_start",
    "contrast_ramp_end",
)


def _generic_kwargs(kwargs: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    result = dict(kwargs)
    for name in _GENERIC_OWNED:
        result.pop(name, None)
    result.update(
        arm=arm,
        aa_selector="joint_top_fraction",
        contrast_mode="none",
        contrast_beta=0.0,
        contrast_margin=0.10,
        contrast_teacher_ratio_threshold=2.0,
        contrast_temperature=1.5,
        contrast_ramp_start=32,
        contrast_ramp_end=112,
        audit_every_microbatch=True,
    )
    return result


class KeOPDV10UniformOPDTrainerV1(KeOPDV2Trainer):
    """Direct CE plus Uniform OPD; no AA and no Student-Wrong graph."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **_generic_kwargs(kwargs, arm="U"))

    def _opd_loss(self, model: torch.nn.Module, row: dict[str, Any], real_waveform: Any):
        opd, contrast, metrics = super()._opd_loss(model, row, real_waveform)
        metrics.update(
            phaseb_arm_id="uniform_opd",
            aa_enabled=0.0,
            aa_top_fraction_used=0.0,
            aa_ratio_gate_used=0.0,
            opd_token_weighting_uniform=1.0,
            teacher_forward_calls_observed=1,
            teacher_wrong_forward_executed=0.0,
            student_wrong_forward_executed=0.0,
            paired_nce_enabled=0.0,
            paired_nce_computed=0.0,
        )
        return opd, contrast, metrics


class KeOPDV10AAOnlyTrainerV1(KeOPDV2Trainer):
    """Direct CE plus AA-weighted OPD; PAC remains completely disabled."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **_generic_kwargs(kwargs, arm="AA"))

    def _opd_loss(self, model: torch.nn.Module, row: dict[str, Any], real_waveform: Any):
        opd, contrast, metrics = super()._opd_loss(model, row, real_waveform)
        metrics.update(
            phaseb_arm_id="aa_only",
            aa_enabled=1.0,
            aa_top_fraction_used=1.0,
            aa_ratio_gate_used=0.0,
            opd_token_weighting_uniform=0.0,
            teacher_forward_calls_observed=2,
            teacher_wrong_forward_executed=1.0,
            student_wrong_forward_executed=0.0,
            paired_nce_enabled=0.0,
            paired_nce_computed=0.0,
        )
        return opd, contrast, metrics


class KeOPDV10PACSelectedTrainerV1(KeOPDBO6DPACTrainerV1):
    """Uniform OPD plus PAC on the exact final-method AA-selected mask."""

    def __init__(self, *args: Any, bo6d_config: BO6DPACConfig, **kwargs: Any) -> None:
        super().__init__(*args, bo6d_config=bo6d_config, **kwargs)

    def _build_pac_targets(self, log_ratios, support, word_ids, rollout):
        """Historical word selector; opt-in subclasses may change target units."""
        del rollout
        config = self.v2_bo6d_config
        raw_weights = bo6d_joint_top_fraction_word_weights(
            log_ratios[0].tolist(), support[0].tolist(), word_ids, config=config)
        selected = torch.tensor(raw_weights, dtype=torch.float32,
                                device=log_ratios.device).unsqueeze(0).gt(1)
        eligible, _, audits = bo6d_strict_aa_word_contrast_targets(
            log_ratios, selected, support, [word_ids], config=config)
        return raw_weights, eligible, audits, [word_ids], {}

    def _opd_loss(
        self,
        model: torch.nn.Module,
        row: dict[str, Any],
        real_waveform: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
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
            raise FloatingPointError("non-finite selected-PAC Teacher-Real logits")
        if not bool(torch.isfinite(teacher_wrong).all().item()):
            raise FloatingPointError("non-finite selected-PAC Teacher-Wrong logits")

        word_ids = completion_word_ids(self.processor.tokenizer, rollout[0].tolist())
        log_ratios, support = sampled_token_audio_log_ratio(
            teacher_real,
            teacher_wrong,
            rollout,
            self.v2_valid_ids,
            support_top_k=self.v2_contract.teacher_support_top_k,
        )
        raw_weights, eligible_mask, gate_audits, loss_unit_ids, selector_metrics = (
            self._build_pac_targets(log_ratios, support, word_ids, rollout)
        )
        advantages = torch.where(support, log_ratios, torch.zeros_like(log_ratios))
        aa_weights = torch.tensor(
            raw_weights, dtype=torch.float32, device=log_ratios.device
        ).unsqueeze(0)
        selected_mask = aa_weights.gt(1)
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
            raise FloatingPointError("non-finite selected-PAC Uniform RKLD")
        del teacher_real

        contrast = paired.real_logits.sum().reshape(1) * 0.0
        captured_nce: PairedNCEResult | None = None
        paired_stats: dict[str, float | None] = {
            "paired_nce_eligible_words": float(gate_audits[0]["contrast_eligible_words"]),
            "paired_nce_eligible_tokens": float(gate_audits[0]["contrast_eligible_tokens"]),
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
                loss_unit_ids,
                tau=config.tau,
                detach_wrong=(
                    self.v2_pac_wrong_gradient_mode == "detach_wrong"
                ),
            )
            contrast = nce.losses
            captured_nce = nce
            self._register_pac_gradient_audit(sample_id=str(row["id"]), result=nce)
            nce_audit = nce.row_audits[0]
            if (
                nce_audit["paired_nce_eligible_words"]
                != paired_stats["paired_nce_eligible_words"]
                or nce_audit["paired_nce_eligible_tokens"]
                != paired_stats["paired_nce_eligible_tokens"]
            ):
                raise AssertionError("selected-PAC objective and Teacher gate populations differ")
            paired_stats.update(nce_audit)
        if not bool(torch.isfinite(contrast).all().item()):
            raise FloatingPointError("non-finite selected-PAC loss")

        calibration_rows = getattr(self, "_v2_pac_calibration_rows", None)
        if calibration_rows is not None:
            calibration_rows.append(
                PACCalibrationRowCapture(
                    sample_id=str(row["id"]),
                    teacher_correct=True,
                    rollout_token_ids=tuple(int(value) for value in rollout[0].tolist()),
                    complete_word_mask=eligible_mask,
                    token_word_ids=tuple(loss_unit_ids[0]),
                    paired_result=captured_nce,
                    wrong_graph_executed=bool(decision.execute_wrong),
                )
            )

        by_word_advantages: dict[int, list[float]] = {}
        by_word_weights: dict[int, set[float]] = {}
        for word_id, advantage, weight in zip(
            word_ids, advantages[0].tolist(), raw_weights, strict=True
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
            for word_id, values in by_word_weights.items()
            if any(value > 1 for value in values)
        }
        lexical = torch.tensor(
            [[word_id is not None for word_id in word_ids]],
            dtype=torch.bool,
            device=advantages.device,
        )
        positive_lexical = advantages.gt(0) & lexical
        supported_lexical = support & lexical
        lexical_tokens = int(lexical.sum().item())
        selected_tokens = int(selected_mask.sum().item())
        lexical_words = len(word_scores)
        top_limit = max(1, math.ceil(lexical_words * config.top_fraction)) if lexical_words else 0
        rollout_text = self.processor.tokenizer.decode(
            rollout[0], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        metrics: dict[str, Any] = {
            **gate_audits[0],
            **decision.audit(),
            **paired.audit,
            **paired_stats,
            **config.audit(),
            "bo6d_pac_wrong_gradient_mode": self.v2_pac_wrong_gradient_mode,
            "rounded_x15_ablation_trainer_version": TRAINER_VERSION,
            "phaseb_arm_id": "pac_aa_mask",
            "contrast_objective_version": BO6D_PAC_OBJECTIVE_VERSION,
            "paired_nce_gate_version": BO6D_PAC_GATE_VERSION,
            "paired_nce_enabled": 1.0,
            "paired_nce_tau": float(config.tau),
            "paired_nce_wrong_gradient_mode": self.v2_pac_wrong_gradient_mode,
            "paired_nce_beta_target": float(config.beta),
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
            "contrast_loss": float(contrast.detach().item()),
            "contrast_mode": BO6D_PAC_OBJECTIVE_VERSION,
            "paired_prefix_equal": float(prefix_equal),
            "paired_same_rollout": float(same_rollout),
            "rollout_tokens": float(rollout.numel()),
            "rollout_complete_think": float(
                "<think>" in rollout_text.lower() and "</think>" in rollout_text.lower()
            ),
            "rollout_has_answer": float("<answer" in rollout_text.lower()),
            "rollout_preview": rollout_text[:240],
            "aa_enabled": 1.0,
            "aa_high_tokens": float(selected_tokens),
            "aa_high_words": float(len(selected_words)),
            "aa_lexical_words": float(lexical_words),
            "aa_positive_words": float(len(positive_words)),
            "aa_top_fraction": float(config.top_fraction),
            "aa_top_limit": float(top_limit),
            "aa_selected_words": float(len(selected_words)),
            "aa_selected_tokens": float(selected_tokens),
            "aa_zero_high_rows": float(selected_tokens == 0),
            "aa_selector": "joint_top_fraction_for_pac_mask_only",
            "aa_selection_unit": "word",
            "aa_top_denominator_units": float(lexical_words),
            "aa_ratio_threshold": float(config.ratio_gate),
            "aa_lexical_tokens": float(lexical_tokens),
            "aa_positive_tokens": float(positive_lexical.sum().item()),
            "support_tokens": float(supported_lexical.sum().item()),
            "support_fraction": float(supported_lexical.sum().item() / max(lexical_tokens, 1)),
            "aa_high_token_fraction": float(selected_tokens / max(lexical_tokens, 1)),
            "mean_positive_aa": float(
                advantages[positive_lexical].mean().item()
                if positive_lexical.any()
                else 0.0
            ),
            "aa_top_fraction_used": 1.0,
            "aa_ratio_gate_used": 1.0,
            "opd_token_weighting_uniform": 1.0,
            "teacher_forward_calls_observed": 2,
            "teacher_wrong_forward_executed": 1.0,
            "student_wrong_forward_executed": float(decision.execute_wrong),
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
        uniform_weights = torch.ones_like(rkl, dtype=torch.float32)
        opd = weighted_position_mean(rkl, uniform_weights, mask)
        if not bool(torch.isfinite(opd).all().item()):
            raise FloatingPointError("non-finite selected-PAC Uniform OPD loss")
        metrics.update(selector_metrics)
        return opd, contrast, metrics


class KeOPDV10PACAllTrainerV1(KeOPDV2Trainer):
    """Uniform OPD plus PAC on every fully Teacher-Real-supported lexical word."""

    def __init__(self, *args: Any, pac_config: BO6DPACConfig, **kwargs: Any) -> None:
        if type(pac_config) is not BO6DPACConfig or not pac_config.enabled:
            raise ValueError("PAC-all requires the exact enabled Rounded-X15 config")
        KeOPDV2Trainer.__init__(self, *args, **_generic_kwargs(kwargs, arm="U"))
        self.v2_pac_all_config = pac_config
        self.v2_round2_teacher_contract_checks = validate_round2_teacher_contract(
            self.model, self.v2_teacher
        )
        if not math.isclose(float(self.v2_contract.lambda_opd), 0.25, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("PAC-all lambda OPD changed")
        if int(self.v2_contract.teacher_support_top_k) != 128:
            raise ValueError("PAC-all Teacher support top-k changed")
        self.v2_contrast_mode = PAC_ALL_OBJECTIVE_VERSION
        self.v2_contrast_beta = float(pac_config.beta)
        self.v2_contrast_temperature = float(pac_config.tau)
        self.v2_contrast_ramp_start = int(pac_config.ramp_start)
        self.v2_contrast_ramp_end = int(pac_config.ramp_end)
        self._v2_pac_all_gradient_pending_steps: set[int] = set()
        self.v2_pac_all_gradient_audits: list[dict[str, object]] = []

    def _require_teacher_gate_value(self, row: Mapping[str, Any]) -> bool:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or sample_id not in self.v2_teacher_gate:
            raise ValueError("PAC-all row lacks a bound Teacher gate value")
        value = self.v2_teacher_gate[sample_id]
        if type(value) is not bool:
            raise ValueError("PAC-all Teacher gate value must be an exact boolean")
        return value

    def _effective_contrast_beta(self) -> float:
        return self.v2_pac_all_config.effective_beta(int(self.state.global_step))

    def compute_loss(self, model: torch.nn.Module, inputs: list[dict[str, Any]], **kwargs: Any):
        for row in inputs:
            self._require_teacher_gate_value(row)
        self.v2_round2_teacher_contract_checks = validate_round2_teacher_contract(
            model, self.v2_teacher
        )
        return super().compute_loss(model, inputs, **kwargs)

    def _register_pac_all_gradient_audit(
        self, *, sample_id: str, result: PairedNCEResult
    ) -> None:
        step = int(self.state.global_step)
        if (
            not self.is_world_process_zero()
            or step in self._v2_pac_all_gradient_pending_steps
            or any(row.get("global_step_before_update") == step for row in self.v2_pac_all_gradient_audits)
        ):
            return
        self._v2_pac_all_gradient_pending_steps.add(step)
        observed: dict[str, dict[str, float]] = {}

        def emit_if_complete() -> None:
            if set(observed) != {"real", "wrong"}:
                return
            real, wrong = observed["real"], observed["wrong"]
            passed = bool(
                real["finite"] == real["nonzero"] == 1.0
                and wrong["finite"] == wrong["nonzero"] == 1.0
                and result.wrong_log_probs.requires_grad
            )
            audit: dict[str, object] = {
                "scope": "rank0_first_active_row_per_optimizer_step",
                "run_name": self.v2_audit_run_name,
                "attempt_id": self.v2_audit_attempt_id,
                "global_step_before_update": step,
                "optimizer_step": step + 1,
                "id": sample_id,
                "wrong_gradient_mode": "bidirectional",
                "real_sampled_logprob_gradient_finite": real["finite"],
                "real_sampled_logprob_gradient_nonzero": real["nonzero"],
                "real_sampled_logprob_gradient_norm": real["norm"],
                "wrong_sampled_logprob_requires_grad": float(result.wrong_log_probs.requires_grad),
                "wrong_sampled_logprob_gradient_finite": wrong["finite"],
                "wrong_sampled_logprob_gradient_nonzero": wrong["nonzero"],
                "wrong_sampled_logprob_gradient_norm": wrong["norm"],
                "variant_gradient_contract_passed": float(passed),
            }
            self.v2_pac_all_gradient_audits.append(audit)
            self._v2_pac_all_gradient_pending_steps.discard(step)
            print("ROUNDED_X15_PAC_ALL_GRAD_AUDIT=" + json.dumps(audit, allow_nan=False, sort_keys=True), flush=True)

        def capture(branch: str):
            def hook(gradient: torch.Tensor) -> torch.Tensor:
                detached = gradient.detach()
                finite = bool(torch.isfinite(detached).all().item())
                norm = float(detached.float().norm().item()) if finite else float("nan")
                observed[branch] = {
                    "finite": float(finite),
                    "nonzero": float(finite and norm > 0.0),
                    "norm": norm,
                }
                emit_if_complete()
                return gradient
            return hook

        result.real_log_probs.register_hook(capture("real"))
        result.wrong_log_probs.register_hook(capture("wrong"))

    def _opd_loss(
        self,
        model: torch.nn.Module,
        row: dict[str, Any],
        real_waveform: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self._require_teacher_gate_value(row) is not True:
            raise RuntimeError("PAC-all OPD/PAC path requires a Teacher-correct row")
        config = self.v2_pac_all_config
        real_prompt = prepare_prompt_inputs(
            self.processor, row["canonical_user_text"], real_waveform,
            self.v2_sampling_rate, self.args.device,
        )
        rollout = self._generate_rollout(model, real_prompt)
        real_sequence = append_completion(real_prompt, rollout)
        prompt_length = int(real_prompt["input_ids"].shape[1])
        logit_start = prompt_length - 1
        logit_stop = logit_start + int(rollout.shape[1])
        wrong_waveform = load_audio(row["wrong_audio_path"], self.v2_sampling_rate)
        wrong_waveform = match_waveform_length(wrong_waveform, len(real_waveform))
        wrong_prompt = prepare_prompt_inputs(
            self.processor, row["canonical_user_text"], wrong_waveform,
            self.v2_sampling_rate, self.args.device,
        )
        wrong_sequence = append_completion(wrong_prompt, rollout)
        prefix_equal, same_rollout, _ = _paired_input_contract(
            model, real_sequence, wrong_sequence, prompt_length=prompt_length
        )
        teacher_contract = validate_round2_teacher_contract(model, self.v2_teacher)
        assert self.v2_teacher is not None
        with torch.inference_mode():
            teacher_real = self.v2_teacher(
                **real_sequence, use_cache=False, return_dict=True
            ).logits[:, logit_start:logit_stop].clone()
        if not bool(torch.isfinite(teacher_real).all().item()):
            raise FloatingPointError("non-finite PAC-all Teacher-Real logits")
        word_ids = completion_word_ids(self.processor.tokenizer, rollout[0].tolist())
        support = sampled_token_teacher_real_support(
            teacher_real,
            rollout,
            self.v2_valid_ids,
            support_top_k=self.v2_contract.teacher_support_top_k,
        )
        eligible_mask, eligibility_audits = pac_all_complete_word_targets(
            support, [word_ids]
        )
        decision = decide_bo6d_pac_execution(
            config,
            global_step=int(self.state.global_step),
            eligible_mask=eligible_mask,
        )
        paired = paired_student_condition_forward(
            model,
            real_sequence,
            wrong_sequence,
            prompt_length=prompt_length,
            logit_start=logit_start,
            logit_stop=logit_stop,
            execute_wrong=decision.execute_wrong,
            wrong_requires_grad=True,
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
            raise FloatingPointError("non-finite PAC-all Uniform RKLD")
        contrast = paired.real_logits.sum().reshape(1) * 0.0
        paired_stats: dict[str, float | None] = {
            "paired_nce_eligible_words": eligibility_audits[0]["pac_all_eligible_words"],
            "paired_nce_eligible_tokens": eligibility_audits[0]["pac_all_eligible_tokens"],
            "paired_nce_mean_gap": None,
            "paired_nce_mean_prob": None,
            "paired_nce_mean_loss": None,
            "paired_nce_mean_gradient_factor": None,
            "paired_nce_saturation_fraction_preal_ge_0p95": None,
            "paired_nce_wrong_preferred_fraction": None,
        }
        if decision.execute_wrong:
            nce = pac_nce_from_paired_forward(
                paired, rollout, self.v2_valid_ids, eligible_mask, [word_ids],
                tau=config.tau, detach_wrong=False,
            )
            contrast = nce.losses
            nce_audit = nce.row_audits[0]
            if (
                nce_audit["paired_nce_eligible_words"]
                != paired_stats["paired_nce_eligible_words"]
                or nce_audit["paired_nce_eligible_tokens"]
                != paired_stats["paired_nce_eligible_tokens"]
            ):
                raise AssertionError("PAC-all objective and support populations differ")
            paired_stats.update(nce_audit)
            self._register_pac_all_gradient_audit(sample_id=str(row["id"]), result=nce)
        if not bool(torch.isfinite(contrast).all().item()):
            raise FloatingPointError("non-finite PAC-all paired NCE")
        lexical_tokens = sum(word_id is not None for word_id in word_ids)
        supported_lexical_tokens = sum(
            word_id is not None and bool(support[0, index].item())
            for index, word_id in enumerate(word_ids)
        )
        rollout_text = self.processor.tokenizer.decode(
            rollout[0], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        metrics: dict[str, Any] = {
            **eligibility_audits[0],
            **decision.audit(),
            **paired.audit,
            **paired_stats,
            **config.audit(),
            "rounded_x15_ablation_trainer_version": TRAINER_VERSION,
            "phaseb_arm_id": "pac_all",
            "contrast_objective_version": PAC_ALL_OBJECTIVE_VERSION,
            "paired_nce_gate_version": PHASEB_PAC_ALL_ELIGIBILITY_VERSION,
            "paired_nce_reduction_version": PAC_NCE_REDUCTION_VERSION,
            "paired_nce_reduction_semantics": (
                "equal_complete_words_within_row_then_all_local_microbatch_rows;"
                "empty_gate_and_teacher_false_rows_are_zero"
            ),
            "paired_nce_enabled": 1.0,
            "paired_nce_tau": float(config.tau),
            "paired_nce_wrong_gradient_mode": "bidirectional",
            "paired_nce_beta_target": float(config.beta),
            "paired_nce_computed": float(decision.execute_wrong),
            "paired_nce_statistics_defined": float(decision.execute_wrong),
            "paired_nce_objective_row_computed": float(decision.execute_wrong),
            "contrast_eligible_row": float(eligible_mask.any().item()),
            "contrast_active_row": float(decision.execute_wrong),
            "contrast_loss": float(contrast.detach().item()),
            "contrast_mode": PAC_ALL_OBJECTIVE_VERSION,
            "paired_prefix_equal": float(prefix_equal),
            "paired_same_rollout": float(same_rollout),
            "rollout_tokens": float(rollout.numel()),
            "rollout_complete_think": float(
                "<think>" in rollout_text.lower() and "</think>" in rollout_text.lower()
            ),
            "rollout_has_answer": float("<answer" in rollout_text.lower()),
            "rollout_preview": rollout_text[:240],
            "aa_enabled": 0.0,
            "aa_selector": "none_pac_all",
            "aa_high_tokens": 0.0,
            "aa_high_words": 0.0,
            "aa_selected_words": 0.0,
            "aa_selected_tokens": 0.0,
            "aa_high_token_fraction": 0.0,
            "aa_top_fraction_used": 0.0,
            "aa_ratio_gate_used": 0.0,
            "aa_lexical_tokens": float(lexical_tokens),
            "support_tokens": float(supported_lexical_tokens),
            "support_fraction": float(supported_lexical_tokens / max(lexical_tokens, 1)),
            "opd_token_weighting_uniform": 1.0,
            "teacher_forward_calls_observed": 1,
            "teacher_wrong_forward_executed": 0.0,
            "student_wrong_forward_executed": float(decision.execute_wrong),
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
        opd = weighted_position_mean(rkl, torch.ones_like(rkl, dtype=torch.float32), mask)
        if not bool(torch.isfinite(opd).all().item()):
            raise FloatingPointError("non-finite PAC-all Uniform OPD loss")
        return opd, contrast, metrics


def trainer_class_for_arm(arm_id: str):
    mapping = {
        "uniform_opd": KeOPDV10UniformOPDTrainerV1,
        "aa_only": KeOPDV10AAOnlyTrainerV1,
        "pac_aa_mask": KeOPDV10PACSelectedTrainerV1,
        "pac_all": KeOPDV10PACAllTrainerV1,
        "aa_pac": KeOPDBO6DPACTrainerV1,
    }
    try:
        return mapping[arm_id]
    except KeyError as error:
        raise ValueError(f"unknown Rounded-X15 ablation arm: {arm_id}") from error


__all__ = [
    "KeOPDV10AAOnlyTrainerV1",
    "KeOPDV10PACAllTrainerV1",
    "KeOPDV10PACSelectedTrainerV1",
    "KeOPDV10UniformOPDTrainerV1",
    "TRAINER_VERSION",
    "trainer_class_for_arm",
]
