"""Additive PAC-all objective for the V1.2 Phase-B matched ablation.

PAC-all intentionally keeps Uniform OPD and changes only the PAC mask.  On a
Teacher-correct row, every lexical word emitted by the frozen completion-word
mapper is eligible when every mapped BPE piece is in the Teacher-Real top-128
support set.  The mask never reads a Teacher Real/Wrong ratio, a top fraction,
or a ratio threshold.

The paired Student loss delegates to the already-audited PAC-2S reduction:
equal-weight complete words within each row, followed by the inherited mean
over every row in the local microbatch.  Teacher-false and empty-mask rows are
therefore explicit zeros in the full-row denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Final, Mapping, Sequence

import torch

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
    PairedNCEResult,
    _paired_input_contract,
    pac_nce_from_paired_forward,
    paired_student_condition_forward,
    validate_round2_teacher_contract,
)


PHASEB_PAC_ALL_CONFIG_VERSION: Final = "ke_opd_v12_phaseb_pac_all_config_v1"
PHASEB_PAC_ALL_TRAINER_VERSION: Final = "ke_opd_v12_phaseb_pac_all_trainer_v1"
PHASEB_PAC_ALL_ELIGIBILITY_VERSION: Final = (
    "teacher_correct_complete_lexical_teacher_real_top128_v1"
)
PHASEB_PAC_ALL_OBJECTIVE_VERSION: Final = "paired_audio_condition_nce_v1"
PHASEB_PAC_ALL_ALLOWED_BETAS: Final = (0.005, 0.010, 0.020)
PHASEB_PAC_ALL_TEMPERATURE: Final = 1.50
PHASEB_PAC_ALL_RAMP_START: Final = 63
PHASEB_PAC_ALL_RAMP_END: Final = 219
PHASEB_PAC_ALL_SUPPORT_TOP_K: Final = 128


def _finite_number(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _canonical(value: object, expected: float, label: str) -> float:
    result = _finite_number(value, label)
    if not math.isclose(result, expected, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"{label} is frozen at {expected}")
    return float(expected)


def _canonical_beta(value: object) -> float:
    result = _finite_number(value, "beta")
    for beta in PHASEB_PAC_ALL_ALLOWED_BETAS:
        if math.isclose(result, beta, rel_tol=0.0, abs_tol=1e-15):
            return float(beta)
    raise ValueError(
        f"beta must be one of {PHASEB_PAC_ALL_ALLOWED_BETAS}"
    )


@dataclass(frozen=True)
class PhaseBPACAllConfig:
    """Fail-closed PAC-all configuration late-bound from the Phase-A winner."""

    beta: float
    tau: float = PHASEB_PAC_ALL_TEMPERATURE
    ramp_start: int = PHASEB_PAC_ALL_RAMP_START
    ramp_end: int = PHASEB_PAC_ALL_RAMP_END
    teacher_support_top_k: int = PHASEB_PAC_ALL_SUPPORT_TOP_K
    engineering_smoke: bool = False

    def __post_init__(self) -> None:
        if type(self.engineering_smoke) is not bool:
            raise ValueError("engineering_smoke must be an exact boolean")
        beta = _canonical_beta(self.beta)
        tau = _canonical(self.tau, PHASEB_PAC_ALL_TEMPERATURE, "tau")
        if type(self.teacher_support_top_k) is not int:
            raise ValueError("teacher_support_top_k must be an exact integer")
        if self.teacher_support_top_k != PHASEB_PAC_ALL_SUPPORT_TOP_K:
            raise ValueError("teacher_support_top_k is frozen at 128")
        if type(self.ramp_start) is not int or type(self.ramp_end) is not int:
            raise ValueError("ramp bounds must be exact integers")
        expected_ramp = (
            (0, 1)
            if self.engineering_smoke
            else (PHASEB_PAC_ALL_RAMP_START, PHASEB_PAC_ALL_RAMP_END)
        )
        if (self.ramp_start, self.ramp_end) != expected_ramp:
            raise ValueError(
                "PAC-all ramp changed; "
                f"expected={expected_ramp}, observed={(self.ramp_start, self.ramp_end)}"
            )
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "tau", tau)

    @classmethod
    def formal(cls, beta: float) -> "PhaseBPACAllConfig":
        return cls(beta=beta)

    @classmethod
    def smoke(cls, beta: float) -> "PhaseBPACAllConfig":
        return cls(beta=beta, ramp_start=0, ramp_end=1, engineering_smoke=True)

    @property
    def wrong_gradient_mode(self) -> str:
        return "bidirectional"

    def effective_beta(self, global_step: int) -> float:
        if type(global_step) is not int or global_step < 0:
            raise ValueError("global_step must be a non-negative integer")
        if self.engineering_smoke:
            return float(self.beta)
        if global_step < self.ramp_start:
            return 0.0
        if global_step >= self.ramp_end:
            return float(self.beta)
        progress = (global_step - self.ramp_start) / (
            self.ramp_end - self.ramp_start
        )
        return float(self.beta) * progress

    def audit(self) -> dict[str, object]:
        return {
            "phaseb_pac_all_config_version": PHASEB_PAC_ALL_CONFIG_VERSION,
            "phaseb_pac_all_eligibility_version": (
                PHASEB_PAC_ALL_ELIGIBILITY_VERSION
            ),
            "phaseb_pac_all_objective_version": (
                PHASEB_PAC_ALL_OBJECTIVE_VERSION
            ),
            "phaseb_pac_all_enabled": True,
            "phaseb_pac_all_beta": float(self.beta),
            "phaseb_pac_all_temperature": float(self.tau),
            "phaseb_pac_all_ramp_start": int(self.ramp_start),
            "phaseb_pac_all_ramp_end": int(self.ramp_end),
            "phaseb_pac_all_teacher_support_top_k": int(
                self.teacher_support_top_k
            ),
            "phaseb_pac_all_wrong_gradient_mode": self.wrong_gradient_mode,
            "phaseb_pac_all_engineering_smoke": bool(self.engineering_smoke),
            "phaseb_pac_all_eligible_for_formal_schedule": not bool(
                self.engineering_smoke
            ),
            "phaseb_pac_all_uses_aa_top_fraction": False,
            "phaseb_pac_all_uses_teacher_real_wrong_ratio": False,
            "phaseb_pac_all_uses_ratio_gate": False,
        }


@dataclass(frozen=True)
class PACAllExecutionDecision:
    execute_wrong: bool
    effective_beta: float
    eligible_tokens: int
    skip_reason: str

    def audit(self) -> dict[str, object]:
        return {
            "paired_nce_beta_effective": float(self.effective_beta),
            "paired_nce_eligible_tokens": float(self.eligible_tokens),
            "paired_nce_wrong_forward_executed": float(self.execute_wrong),
            "paired_nce_skip_reason": self.skip_reason,
        }


def decide_pac_all_execution(
    config: PhaseBPACAllConfig,
    *,
    global_step: int,
    eligible_mask: torch.Tensor,
) -> PACAllExecutionDecision:
    if type(config) is not PhaseBPACAllConfig:
        raise ValueError("PAC-all decision requires an exact PhaseBPACAllConfig")
    if eligible_mask.dtype != torch.bool or eligible_mask.ndim != 2:
        raise ValueError("PAC-all eligible mask must be boolean [batch, positions]")
    eligible_tokens = int(eligible_mask.sum().item())
    effective_beta = config.effective_beta(global_step)
    if effective_beta == 0.0:
        return PACAllExecutionDecision(
            False, effective_beta, eligible_tokens, "ramp_beta_zero"
        )
    if eligible_tokens == 0:
        return PACAllExecutionDecision(
            False, effective_beta, eligible_tokens, "empty_complete_word_gate"
        )
    return PACAllExecutionDecision(
        True, effective_beta, eligible_tokens, "executed"
    )


def sampled_token_teacher_real_support(
    teacher_real_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    valid_token_ids: torch.Tensor,
    *,
    support_top_k: int = PHASEB_PAC_ALL_SUPPORT_TOP_K,
) -> torch.Tensor:
    """Return the exact inherited Teacher-Real top-k support mask.

    Passing the same Teacher-Real tensor into both slots of the inherited
    helper guarantees support parity with AA+PAC while requiring no
    Teacher-Wrong forward and exposing no Real/Wrong ratio to PAC-all.
    """

    if type(support_top_k) is not int or support_top_k != 128:
        raise ValueError("PAC-all Teacher support top-k is frozen at 128")
    zero_ratio, support = sampled_token_audio_log_ratio(
        teacher_real_logits,
        teacher_real_logits,
        sampled_token_ids,
        valid_token_ids,
        support_top_k=support_top_k,
    )
    if not bool(torch.equal(zero_ratio, torch.zeros_like(zero_ratio))):
        raise AssertionError("identical Teacher-Real support path produced a ratio")
    return support


def pac_all_complete_word_targets(
    support_mask: torch.Tensor,
    token_word_ids: Sequence[Sequence[int | None]],
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Select all and only complete lexical words with full Teacher support."""

    if support_mask.dtype != torch.bool or support_mask.ndim != 2:
        raise ValueError("support_mask must be boolean [batch, positions]")
    batch_size, position_count = support_mask.shape
    if len(token_word_ids) != batch_size:
        raise ValueError("word-ID rows do not match the tensor batch dimension")
    eligible = torch.zeros_like(support_mask, dtype=torch.bool)
    audits: list[dict[str, float]] = []
    for row_index, row_word_ids in enumerate(token_word_ids):
        if len(row_word_ids) != position_count:
            raise ValueError(
                "word-ID row does not match the tensor position dimension"
            )
        by_word: dict[int, list[int]] = {}
        for token_index, word_id in enumerate(row_word_ids):
            if word_id is not None:
                by_word.setdefault(word_id, []).append(token_index)
        supported_words = 0
        partial_support_words = 0
        for indices in by_word.values():
            index = torch.tensor(
                indices, dtype=torch.long, device=support_mask.device
            )
            pieces = support_mask[row_index].index_select(0, index)
            if bool(pieces.all().item()):
                eligible[row_index, index] = True
                supported_words += 1
            elif bool(pieces.any().item()):
                partial_support_words += 1
        audits.append(
            {
                "pac_all_lexical_words": float(len(by_word)),
                "pac_all_complete_supported_words": float(supported_words),
                "pac_all_partial_support_words": float(partial_support_words),
                "pac_all_eligible_words": float(supported_words),
                "pac_all_eligible_tokens": float(
                    eligible[row_index].sum().item()
                ),
            }
        )
    return eligible, audits


def validate_pac_all_training_contract(
    contract: Any, config: PhaseBPACAllConfig
) -> dict[str, bool]:
    checks = {
        "lambda_opd_0p25": math.isclose(
            float(contract.lambda_opd), 0.25, rel_tol=0.0, abs_tol=1e-15
        ),
        "kl_temperature_1p0": math.isclose(
            float(contract.kl_temperature), 1.0, rel_tol=0.0, abs_tol=1e-15
        ),
        "teacher_support_top_k_128": (
            int(contract.teacher_support_top_k)
            == config.teacher_support_top_k
            == 128
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Phase-B PAC-all training contract changed: {checks}")
    return checks


class KeOPDPhaseBPACAllTrainerV1(KeOPDV2Trainer):
    """Uniform OPD plus PAC on every supported complete lexical word."""

    def __init__(
        self,
        *args: Any,
        phaseb_pac_all_config: PhaseBPACAllConfig,
        **kwargs: Any,
    ) -> None:
        if type(phaseb_pac_all_config) is not PhaseBPACAllConfig:
            raise ValueError("PAC-all trainer requires an exact PhaseBPACAllConfig")
        for reserved in (
            "arm",
            "aa_selector",
            "contrast_mode",
            "contrast_beta",
            "contrast_margin",
            "contrast_teacher_ratio_threshold",
            "contrast_temperature",
            "contrast_ramp_start",
            "contrast_ramp_end",
        ):
            if reserved in kwargs:
                raise ValueError(f"PAC-all trainer owns {reserved}")
        if kwargs.get("audit_every_microbatch") is not True:
            raise ValueError("formal PAC-all trainer requires every-microbatch audit")
        KeOPDV2Trainer.__init__(
            self,
            *args,
            arm="U",
            aa_selector="joint_top_fraction",
            contrast_mode="none",
            contrast_beta=0.0,
            contrast_margin=0.10,
            contrast_teacher_ratio_threshold=2.0,
            contrast_temperature=phaseb_pac_all_config.tau,
            contrast_ramp_start=phaseb_pac_all_config.ramp_start,
            contrast_ramp_end=phaseb_pac_all_config.ramp_end,
            **kwargs,
        )
        self.v2_pac_all_config = phaseb_pac_all_config
        self.v2_pac_all_contract_checks = validate_pac_all_training_contract(
            self.v2_contract, phaseb_pac_all_config
        )
        self.v2_round2_teacher_contract_checks = validate_round2_teacher_contract(
            self.model, self.v2_teacher
        )
        for sample_id, gate_value in self.v2_teacher_gate.items():
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("PAC-all Teacher gate IDs must be non-empty strings")
            if type(gate_value) is not bool:
                raise ValueError(
                    "PAC-all Teacher gate values must be exact booleans; "
                    f"id={sample_id!r}"
                )
        self.v2_contrast_mode = PHASEB_PAC_ALL_OBJECTIVE_VERSION
        self.v2_contrast_beta = float(phaseb_pac_all_config.beta)
        self.v2_contrast_temperature = float(phaseb_pac_all_config.tau)
        self.v2_contrast_ramp_start = int(phaseb_pac_all_config.ramp_start)
        self.v2_contrast_ramp_end = int(phaseb_pac_all_config.ramp_end)
        self._v2_pac_all_gradient_pending_steps: set[int] = set()
        self.v2_pac_all_gradient_audits: list[dict[str, object]] = []

    def _require_teacher_gate_value(self, row: Mapping[str, Any]) -> bool:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("PAC-all row ID must be a non-empty string")
        if sample_id not in self.v2_teacher_gate:
            raise ValueError(f"PAC-all Teacher gate is missing id={sample_id!r}")
        value = self.v2_teacher_gate[sample_id]
        if type(value) is not bool:
            raise ValueError(
                "PAC-all Teacher gate values must be exact booleans; "
                f"id={sample_id!r}"
            )
        return value

    def _effective_contrast_beta(self) -> float:
        return self.v2_pac_all_config.effective_beta(int(self.state.global_step))

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: list[dict[str, Any]],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for row in inputs:
            if not isinstance(row, dict):
                raise ValueError("PAC-all training inputs must be row dictionaries")
            self._require_teacher_gate_value(row)
        self.v2_round2_teacher_contract_checks = validate_round2_teacher_contract(
            model, self.v2_teacher
        )
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def _register_pac_all_gradient_audit(
        self, *, sample_id: str, result: PairedNCEResult
    ) -> None:
        step = int(self.state.global_step)
        if (
            not self.is_world_process_zero()
            or step in self._v2_pac_all_gradient_pending_steps
            or any(
                row.get("global_step_before_update") == step
                for row in self.v2_pac_all_gradient_audits
            )
        ):
            return
        self._v2_pac_all_gradient_pending_steps.add(step)
        observed: dict[str, dict[str, float]] = {}

        def emit_if_complete() -> None:
            if set(observed) != {"real", "wrong"}:
                return
            real = observed["real"]
            wrong = observed["wrong"]
            passed = bool(
                real["finite"] == 1.0
                and real["nonzero"] == 1.0
                and wrong["finite"] == 1.0
                and wrong["nonzero"] == 1.0
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
                "wrong_sampled_logprob_requires_grad": float(
                    result.wrong_log_probs.requires_grad
                ),
                "wrong_sampled_logprob_gradient_finite": wrong["finite"],
                "wrong_sampled_logprob_gradient_nonzero": wrong["nonzero"],
                "wrong_sampled_logprob_gradient_norm": wrong["norm"],
                "variant_gradient_contract_passed": float(passed),
            }
            self.v2_pac_all_gradient_audits.append(audit)
            self._v2_pac_all_gradient_pending_steps.discard(step)
            print(
                "PHASEB_PAC_ALL_GRAD_AUDIT="
                + json.dumps(audit, allow_nan=False, sort_keys=True),
                flush=True,
            )

        def capture(branch: str):
            def hook(gradient: torch.Tensor) -> torch.Tensor:
                detached = gradient.detach()
                finite = bool(torch.isfinite(detached).all().item())
                norm = (
                    float(detached.float().norm().item())
                    if finite
                    else float("nan")
                )
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
        if not bool(torch.isfinite(teacher_real).all().item()):
            raise FloatingPointError("non-finite PAC-all Teacher-Real logits")

        word_ids = completion_word_ids(
            self.processor.tokenizer, rollout[0].tolist()
        )
        support = sampled_token_teacher_real_support(
            teacher_real,
            rollout,
            self.v2_valid_ids,
            support_top_k=config.teacher_support_top_k,
        )
        eligible_mask, eligibility_audits = pac_all_complete_word_targets(
            support, [word_ids]
        )
        decision = decide_pac_all_execution(
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
            "paired_nce_eligible_words": eligibility_audits[0][
                "pac_all_eligible_words"
            ],
            "paired_nce_eligible_tokens": eligibility_audits[0][
                "pac_all_eligible_tokens"
            ],
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
                detach_wrong=False,
            )
            contrast = nce.losses
            nce_audit = nce.row_audits[0]
            if (
                nce_audit["paired_nce_eligible_words"]
                != paired_stats["paired_nce_eligible_words"]
                or nce_audit["paired_nce_eligible_tokens"]
                != paired_stats["paired_nce_eligible_tokens"]
            ):
                raise AssertionError(
                    "PAC-all objective and Teacher-support populations differ"
                )
            paired_stats.update(nce_audit)
            self._register_pac_all_gradient_audit(
                sample_id=str(row["id"]), result=nce
            )
        if not bool(torch.isfinite(contrast).all().item()):
            raise FloatingPointError("non-finite PAC-all paired NCE")

        lexical_tokens = sum(word_id is not None for word_id in word_ids)
        supported_lexical_tokens = sum(
            word_id is not None and bool(support[0, index].item())
            for index, word_id in enumerate(word_ids)
        )
        rollout_text = self.processor.tokenizer.decode(
            rollout[0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        metrics: dict[str, Any] = {
            **config.audit(),
            **eligibility_audits[0],
            **decision.audit(),
            **paired.audit,
            **paired_stats,
            "phaseb_pac_all_trainer_version": PHASEB_PAC_ALL_TRAINER_VERSION,
            "contrast_objective_version": PHASEB_PAC_ALL_OBJECTIVE_VERSION,
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
            "contrast_mode": PHASEB_PAC_ALL_OBJECTIVE_VERSION,
            "paired_prefix_equal": float(prefix_equal),
            "paired_same_rollout": float(same_rollout),
            "rollout_tokens": float(rollout.numel()),
            "rollout_complete_think": float(
                "<think>" in rollout_text.lower()
                and "</think>" in rollout_text.lower()
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
            "support_fraction": float(
                supported_lexical_tokens / max(lexical_tokens, 1)
            ),
            "opd_token_weighting_uniform": 1.0,
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
        weights = torch.ones_like(rkl, dtype=torch.float32)
        opd = weighted_position_mean(rkl, weights, mask)
        if not bool(torch.isfinite(opd).all().item()):
            raise FloatingPointError("non-finite PAC-all Uniform OPD loss")
        return opd, contrast, metrics


__all__ = [
    "KeOPDPhaseBPACAllTrainerV1",
    "PACAllExecutionDecision",
    "PHASEB_PAC_ALL_ALLOWED_BETAS",
    "PHASEB_PAC_ALL_CONFIG_VERSION",
    "PHASEB_PAC_ALL_ELIGIBILITY_VERSION",
    "PHASEB_PAC_ALL_OBJECTIVE_VERSION",
    "PHASEB_PAC_ALL_RAMP_END",
    "PHASEB_PAC_ALL_RAMP_START",
    "PHASEB_PAC_ALL_SUPPORT_TOP_K",
    "PHASEB_PAC_ALL_TEMPERATURE",
    "PHASEB_PAC_ALL_TRAINER_VERSION",
    "PhaseBPACAllConfig",
    "decide_pac_all_execution",
    "pac_all_complete_word_targets",
    "sampled_token_teacher_real_support",
    "validate_pac_all_training_contract",
]
