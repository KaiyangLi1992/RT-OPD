"""Isolated Round-2 trainer path for paired audio-condition NCE.

The active Round-1 trainer is intentionally not modified by this module.  The
new class inherits the stable outer ``compute_loss``/scheduler machinery, but
owns the complete AA/OPD row path that needs a trainable Student-Wrong graph.

No launcher imports this module yet.  It is the first, deliberately isolated
integration layer for a later, separately preregistered round.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .contrast_round2 import paired_audio_condition_nce_rows
from .losses import (
    combined_v2_loss,
    joint_top_fraction_word_weights,
    reverse_kl_per_position,
    sampled_token_audio_log_ratio,
    sampled_token_log_prob,
    strict_aa_word_contrast_targets,
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


PAC_NCE_OBJECTIVE_VERSION = "paired_audio_condition_nce_v1"
PAC_NCE_GATE_VERSION = "strict_aa_complete_word_joint_ratio_gt2_v1"
PAC_NCE_REDUCTION_VERSION = (
    "row_balanced_word_mean_then_full_microbatch_row_mean_v1"
)
PAC_NCE_AUDIO_INPUT_KEYS = frozenset({"input_features"})

RNGState = tuple[torch.Tensor, torch.Tensor | None]


def validate_round2_aa_contract(contract: Any) -> dict[str, bool]:
    """Freeze the inherited AA anchor that PAC-NCE is allowed to augment."""

    checks = {
        "aa_top_fraction_0p30": math.isclose(
            float(contract.aa_top_fraction), 0.30, rel_tol=0.0, abs_tol=1e-15
        ),
        "aa_high_weight_2p0": math.isclose(
            float(contract.aa_high_weight), 2.0, rel_tol=0.0, abs_tol=1e-15
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
        raise ValueError(f"Round-2 PAC-NCE AA anchor contract changed: {checks}")
    return checks


@dataclass(frozen=True)
class PairedAudioConditionNCEConfig:
    """Immutable objective and ramp configuration for one Round-2 arm."""

    enabled: bool
    beta: float
    tau: float = 1.0
    ramp_start: int = 32
    ramp_end: int = 80
    engineering_calibration: bool = False
    wrong_gradient_mode: str = "bidirectional"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("PAC-NCE enabled flag must be boolean")
        if not isinstance(self.engineering_calibration, bool):
            raise ValueError("PAC-NCE engineering_calibration flag must be boolean")
        if self.wrong_gradient_mode not in {"detach_wrong", "bidirectional"}:
            raise ValueError(
                "PAC-NCE wrong_gradient_mode must be detach_wrong or bidirectional"
            )
        beta = float(self.beta)
        tau = float(self.tau)
        if not math.isfinite(beta) or beta < 0.0:
            raise ValueError("PAC-NCE beta must be finite and non-negative")
        if self.enabled and beta <= 0.0:
            raise ValueError("enabled PAC-NCE requires beta>0")
        if not self.enabled and beta != 0.0:
            raise ValueError("PAC-NCE control requires beta=0")
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError("PAC-NCE tau must be finite and positive")
        if not math.isclose(tau, 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("formal Round-2 PAC-NCE freezes tau=1.0")
        if not (
            isinstance(self.ramp_start, int)
            and isinstance(self.ramp_end, int)
            and 0 <= self.ramp_start < self.ramp_end
        ):
            raise ValueError("invalid PAC-NCE beta ramp")
        expected_ramp = (0, 1) if self.engineering_calibration else (32, 80)
        if (self.ramp_start, self.ramp_end) != expected_ramp:
            raise ValueError(
                "Round-2 PAC-NCE freezes beta ramp to "
                f"{expected_ramp[0]}->{expected_ramp[1]} for this execution mode"
            )

    def effective_beta(self, global_step: int) -> float:
        """Return the preregistered coefficient before the next update."""

        step = int(global_step)
        if step < 0:
            raise ValueError("global step must be non-negative")
        if not self.enabled or step < self.ramp_start:
            return 0.0
        if step >= self.ramp_end:
            return float(self.beta)
        progress = (step - self.ramp_start) / (self.ramp_end - self.ramp_start)
        return float(self.beta) * progress


@dataclass(frozen=True)
class PACNCEExecutionDecision:
    """Fail-closed decision about materializing the Student-Wrong graph."""

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


def decide_pac_nce_execution(
    config: PairedAudioConditionNCEConfig,
    *,
    global_step: int,
    eligible_mask: torch.Tensor,
) -> PACNCEExecutionDecision:
    """Skip control, zero-ramp, and empty-gate Wrong forwards explicitly."""

    if eligible_mask.dtype != torch.bool or eligible_mask.ndim != 2:
        raise ValueError("PAC-NCE eligible mask must be boolean [batch, positions]")
    eligible_tokens = int(eligible_mask.sum().item())
    effective_beta = config.effective_beta(global_step)
    if not config.enabled:
        return PACNCEExecutionDecision(
            False, effective_beta, eligible_tokens, "control_beta_zero"
        )
    if effective_beta == 0.0:
        return PACNCEExecutionDecision(
            False, effective_beta, eligible_tokens, "ramp_beta_zero"
        )
    if eligible_tokens == 0:
        return PACNCEExecutionDecision(
            False, effective_beta, eligible_tokens, "empty_complete_word_gate"
        )
    return PACNCEExecutionDecision(True, effective_beta, eligible_tokens, "executed")


@dataclass(frozen=True)
class PACCalibrationRowCapture:
    """Attached tensors from one calibration-only row in a single forward."""

    sample_id: str
    teacher_correct: bool
    rollout_token_ids: tuple[int, ...]
    complete_word_mask: torch.Tensor | None
    token_word_ids: tuple[int | None, ...]
    paired_result: "PairedNCEResult | None"
    wrong_graph_executed: bool


@dataclass(frozen=True)
class PACCalibrationComponents:
    """Unweighted B/P components from one full local microbatch graph."""

    base_loss: torch.Tensor
    pac_loss: torch.Tensor
    rows: tuple[PACCalibrationRowCapture, ...]
    full_denominator_rows: int
    wrong_graph_rows: int


def _clone_rng_state(state: RNGState) -> RNGState:
    cpu, cuda = state
    return cpu.clone(), None if cuda is None else cuda.clone()


def _validate_rng_state_scope(
    state: RNGState,
    *,
    input_device: torch.device,
    label: str,
) -> None:
    cpu, cuda = state
    if not isinstance(cpu, torch.Tensor) or cpu.device.type != "cpu":
        raise RuntimeError(f"PAC-NCE {label} RNG snapshot lacks a CPU RNG tensor")
    if input_device.type == "cuda":
        if not isinstance(cuda, torch.Tensor):
            raise RuntimeError(
                f"PAC-NCE {label} RNG snapshot lacks the local CUDA RNG state"
            )
    elif cuda is not None:
        raise RuntimeError(
            f"PAC-NCE {label} CPU execution unexpectedly captured CUDA RNG state"
        )


def _rng_state_components_equal(
    observed: RNGState, expected: RNGState
) -> tuple[bool, bool | None]:
    cpu_ok = torch.equal(observed[0], expected[0])
    if observed[1] is None and expected[1] is None:
        cuda_ok: bool | None = None
    elif observed[1] is None or expected[1] is None:
        cuda_ok = False
    else:
        cuda_ok = torch.equal(observed[1], expected[1])
    return cpu_ok, cuda_ok


def _paired_text_contract(
    real_inputs: dict[str, torch.Tensor],
    wrong_inputs: dict[str, torch.Tensor],
    *,
    prompt_length: int,
) -> tuple[bool, bool]:
    try:
        real_ids = real_inputs["input_ids"]
        wrong_ids = wrong_inputs["input_ids"]
    except KeyError as error:
        raise ValueError("paired Student inputs require input_ids") from error
    if real_ids.ndim != 2 or wrong_ids.ndim != 2 or real_ids.shape != wrong_ids.shape:
        raise ValueError("paired Student input_ids must have one identical 2-D shape")
    if not 0 < int(prompt_length) <= int(real_ids.shape[1]):
        raise ValueError("prompt length is outside paired Student input_ids")
    prefix_equal = torch.equal(
        real_ids[:, :prompt_length], wrong_ids[:, :prompt_length]
    )
    same_rollout = torch.equal(
        real_ids[:, prompt_length:], wrong_ids[:, prompt_length:]
    )
    if not prefix_equal:
        raise ValueError("PAC-NCE Real/Wrong serialized prefixes differ")
    if not same_rollout:
        raise ValueError("PAC-NCE Real/Wrong branches do not share one rollout")
    return prefix_equal, same_rollout


def _paired_input_contract(
    model: torch.nn.Module,
    real_inputs: dict[str, torch.Tensor],
    wrong_inputs: dict[str, torch.Tensor],
    *,
    prompt_length: int,
) -> tuple[bool, bool, dict[str, object]]:
    """Validate one-process/one-device paired conditioning inputs.

    The only tensor whose *value* may differ is Qwen's ``input_features``
    audio tensor.  Its metadata must still align exactly.  Every other tensor
    is part of the shared serialized/text condition and must be value-identical.

    This contract is local to one process and one device.  It is compatible
    with one-device-per-process DDP, but deliberately rejects model/pipeline
    parallel execution in which one process spans multiple CUDA devices.
    """

    real_keys = set(real_inputs)
    wrong_keys = set(wrong_inputs)
    if real_keys != wrong_keys:
        raise ValueError(
            "PAC-NCE Real/Wrong input tensor key sets differ: "
            f"real_only={sorted(real_keys - wrong_keys)}, "
            f"wrong_only={sorted(wrong_keys - real_keys)}"
        )
    missing_audio = PAC_NCE_AUDIO_INPUT_KEYS - real_keys
    if missing_audio:
        raise ValueError(
            "PAC-NCE paired inputs are missing the frozen audio tensor key(s): "
            f"{sorted(missing_audio)}"
        )

    input_devices: set[torch.device] = set()
    non_audio_count = 0
    for key in sorted(real_keys):
        real_value = real_inputs[key]
        wrong_value = wrong_inputs[key]
        if not isinstance(real_value, torch.Tensor) or not isinstance(
            wrong_value, torch.Tensor
        ):
            raise ValueError(f"PAC-NCE paired input {key!r} must be a tensor")
        if real_value.shape != wrong_value.shape:
            raise ValueError(
                f"PAC-NCE Real/Wrong input {key!r} differs in shape"
            )
        if real_value.dtype != wrong_value.dtype:
            raise ValueError(
                f"PAC-NCE Real/Wrong input {key!r} differs in dtype"
            )
        if real_value.device != wrong_value.device:
            raise ValueError(
                f"PAC-NCE Real/Wrong input {key!r} differs in device"
            )
        input_devices.add(real_value.device)
        if key not in PAC_NCE_AUDIO_INPUT_KEYS:
            non_audio_count += 1
            # ``input_ids`` is checked just below with more actionable
            # prefix-vs-rollout diagnostics; together those two slices still
            # prove full-tensor equality.
            if key != "input_ids" and not torch.equal(real_value, wrong_value):
                raise ValueError(
                    "PAC-NCE Real/Wrong non-audio tensor values differ for "
                    f"{key!r}"
                )

    if len(input_devices) != 1:
        raise ValueError(
            "PAC-NCE paired forward requires one process-local tensor device; "
            f"observed={sorted(str(device) for device in input_devices)}"
        )
    input_device = next(iter(input_devices))
    if input_device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "PAC-NCE paired forward supports only CPU tests or one local CUDA device"
        )
    if input_device.type == "cuda" and input_device.index is None:
        raise ValueError("PAC-NCE CUDA tensors require one explicit local device index")

    model_devices = {
        parameter.device
        for parameter in model.parameters()
        if parameter.device.type != "meta"
    }
    model_devices.update(
        buffer.device for buffer in model.buffers() if buffer.device.type != "meta"
    )
    if len(model_devices) > 1 or (
        model_devices and model_devices != {input_device}
    ):
        raise ValueError(
            "PAC-NCE paired forward rejects multi-device/off-device Student tensors "
            f"parameters: inputs={input_device}, "
            f"model_tensors={sorted(str(device) for device in model_devices)}"
        )

    prefix_equal, same_rollout = _paired_text_contract(
        real_inputs, wrong_inputs, prompt_length=prompt_length
    )
    audit: dict[str, object] = {
        "paired_input_contract_version": "all_non_audio_equal_audio_metadata_aligned_v1",
        "paired_execution_scope": "one_process_one_local_device",
        "paired_input_device": str(input_device),
        "paired_input_single_device": 1.0,
        "paired_input_audio_keys": ",".join(sorted(PAC_NCE_AUDIO_INPUT_KEYS)),
        "paired_input_audio_metadata_aligned": 1.0,
        "paired_input_non_audio_tensor_count": float(non_audio_count),
        "paired_input_non_audio_values_equal": 1.0,
    }
    return prefix_equal, same_rollout, audit


def validate_round2_teacher_contract(
    student: torch.nn.Module,
    teacher: torch.nn.Module | None,
) -> dict[str, object]:
    """Require a distinct, fully frozen, recursively-eval Teacher."""

    if teacher is None:
        raise ValueError("Round-2 PAC-NCE requires a Teacher")
    if teacher is student or any(module is teacher for module in student.modules()):
        raise ValueError("Round-2 Teacher must be a different object from Student")

    teacher_parameters = tuple(teacher.parameters())
    if any(parameter.requires_grad for parameter in teacher_parameters):
        raise ValueError("Round-2 Teacher must have every parameter frozen")
    if any(module.training for module in teacher.modules()):
        raise ValueError("Round-2 Teacher must be recursively in eval mode")

    student_parameter_ids = {id(parameter) for parameter in student.parameters()}
    shared_parameter_count = sum(
        id(parameter) in student_parameter_ids for parameter in teacher_parameters
    )
    if shared_parameter_count:
        raise ValueError("Round-2 Teacher and Student must not share parameters")

    return {
        "paired_teacher_distinct_student": True,
        "paired_teacher_all_parameters_frozen": True,
        "paired_teacher_recursive_eval": True,
        "paired_teacher_parameter_count": len(teacher_parameters),
        "paired_teacher_shared_parameter_count": 0,
    }


@dataclass
class PairedStudentForward:
    """Sliced Real/Wrong Student logits plus JSON-safe engineering audit."""

    real_logits: torch.Tensor
    wrong_logits: torch.Tensor | None
    audit: dict[str, object]


def paired_student_condition_forward(
    model: torch.nn.Module,
    real_inputs: dict[str, torch.Tensor],
    wrong_inputs: dict[str, torch.Tensor],
    *,
    prompt_length: int,
    logit_start: int,
    logit_stop: int,
    execute_wrong: bool,
    wrong_requires_grad: bool = True,
    get_rng_state: Callable[[], RNGState],
    set_rng_state: Callable[[RNGState], None],
) -> PairedStudentForward:
    """Run paired trainable Student branches with replayed dropout RNG.

    The global CPU/local-CUDA trajectory advances exactly as one Real forward:
    the state immediately before Real is replayed for Wrong, then the state
    immediately after Real is restored even if Wrong raises.  Real always
    retains an autograd graph; Wrong does so only for the bidirectional variant.
    """

    prefix_equal, same_rollout, input_audit = _paired_input_contract(
        model, real_inputs, wrong_inputs, prompt_length=prompt_length
    )
    input_device = real_inputs["input_ids"].device
    if not 0 <= int(logit_start) < int(logit_stop):
        raise ValueError("invalid paired Student logit slice")

    def snapshot_rng(label: str) -> RNGState:
        state = get_rng_state()
        _validate_rng_state_scope(state, input_device=input_device, label=label)
        return _clone_rng_state(state)

    def run_branch(
        inputs: dict[str, torch.Tensor], label: str, *, requires_grad: bool,
    ) -> torch.Tensor:
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            outputs = model(**inputs, use_cache=False, return_dict=True)
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError(f"PAC-NCE Student-{label} returned invalid logits")
        if int(logit_stop) > int(logits.shape[1]):
            raise ValueError("paired Student logit slice exceeds sequence length")
        # Materialize only completion-position logits.  A plain view would keep
        # the full audio-prefix vocabulary tensor alive in each retained
        # autograd graph, which is especially costly once Wrong also has grad.
        selected = logits[:, logit_start:logit_stop].clone()
        if not selected.is_floating_point():
            raise ValueError(f"PAC-NCE Student-{label} logits are not floating point")
        if not bool(torch.isfinite(selected).all().item()):
            raise FloatingPointError(f"non-finite PAC-NCE Student-{label} logits")
        if requires_grad:
            if not selected.requires_grad or selected.grad_fn is None:
                raise RuntimeError(f"PAC-NCE Student-{label} graph is detached")
        elif selected.requires_grad or selected.grad_fn is not None:
            raise RuntimeError(f"PAC-NCE Student-{label} unexpectedly retained a graph")
        return selected

    rng_before_real = snapshot_rng("before-Real")
    real_logits = run_branch(real_inputs, "Real", requires_grad=True)
    rng_after_real = snapshot_rng("after-Real")

    wrong_logits: torch.Tensor | None = None
    cpu_restored: bool | None = None
    cuda_restored: bool | None = None
    trajectory_cpu_equal: bool | None = None
    trajectory_cuda_equal: bool | None = None
    trajectory_exact: bool | None = None
    replay_verified = False
    if execute_wrong:
        set_rng_state(rng_before_real)
        try:
            observed_before_wrong = snapshot_rng("before-Wrong")
            replay_cpu_ok, replay_cuda_ok = _rng_state_components_equal(
                observed_before_wrong, rng_before_real
            )
            replay_verified = replay_cpu_ok and replay_cuda_ok is not False
            if not replay_verified:
                raise RuntimeError("paired Student-Wrong RNG replay was not installed")
            wrong_logits = run_branch(
                wrong_inputs, "Wrong", requires_grad=wrong_requires_grad
            )
            # Check *before* restoration.  Replaying the same pre-forward RNG
            # is insufficient if one branch consumes a different number of
            # random draws; that would give the two conditions different
            # dropout/stochastic-depth realizations while a later restore
            # falsely made the global trajectory look correct.
            rng_after_wrong = snapshot_rng("after-Wrong")
            trajectory_cpu_equal, trajectory_cuda_equal = (
                _rng_state_components_equal(rng_after_wrong, rng_after_real)
            )
            trajectory_exact = bool(
                trajectory_cpu_equal and trajectory_cuda_equal is not False
            )
            if not trajectory_exact:
                raise RuntimeError(
                    "paired Student-Wrong RNG trajectory differs from Student-Real"
                )
        finally:
            # Restoration is unconditional so a failed/non-finite Wrong branch
            # cannot silently perturb later rollout, dropout, or sampler state.
            set_rng_state(rng_after_real)
        observed = snapshot_rng("after-restore")
        cpu_ok, cuda_ok = _rng_state_components_equal(observed, rng_after_real)
        cpu_restored = cpu_ok
        cuda_restored = cuda_ok
        if not cpu_ok or cuda_ok is False:
            raise RuntimeError("paired Student-Wrong forward did not restore RNG")

    restored = bool(
        execute_wrong and cpu_restored is True and cuda_restored is not False
    )
    audit: dict[str, object] = {
        **input_audit,
        "paired_rng_scope": "global_cpu_and_one_process_local_cuda_device",
        "paired_rng_cpu_state_present": 1.0,
        "paired_rng_local_cuda_state_present": float(input_device.type == "cuda"),
        "paired_prefix_equal": float(prefix_equal),
        "paired_same_rollout": float(same_rollout),
        "paired_student_real_requires_grad": float(real_logits.requires_grad),
        "paired_student_wrong_forward": float(execute_wrong),
        "additional_student_wrong_forward": float(execute_wrong),
        "paired_student_wrong_requires_grad": float(
            wrong_logits is not None and wrong_logits.requires_grad
        ),
        "paired_student_wrong_gradient_mode": (
            "bidirectional" if wrong_requires_grad else "detach_wrong"
        ),
        "paired_student_dropout_rng_replayed": float(replay_verified),
        "paired_student_wrong_cpu_rng_matches_real": (
            None if trajectory_cpu_equal is None else float(trajectory_cpu_equal)
        ),
        "paired_student_wrong_cuda_rng_matches_real": (
            None if trajectory_cuda_equal is None else float(trajectory_cuda_equal)
        ),
        "paired_student_wrong_rng_trajectory_exact": (
            None if trajectory_exact is None else float(trajectory_exact)
        ),
        "paired_student_wrong_cpu_rng_restored": (
            None if cpu_restored is None else float(cpu_restored)
        ),
        "paired_student_wrong_cuda_rng_restored": (
            None if cuda_restored is None else float(cuda_restored)
        ),
        "paired_student_wrong_rng_restored": float(restored),
    }
    return PairedStudentForward(real_logits, wrong_logits, audit)


@dataclass
class PairedNCEResult:
    losses: torch.Tensor
    row_audits: list[dict[str, float]]
    real_log_probs: torch.Tensor
    wrong_log_probs: torch.Tensor


def pac_nce_from_paired_forward(
    paired: PairedStudentForward,
    rollout: torch.Tensor,
    valid_token_ids: torch.Tensor,
    eligible_mask: torch.Tensor,
    token_word_ids: Sequence[Sequence[int | None]],
    *,
    tau: float,
    detach_wrong: bool = False,
) -> PairedNCEResult:
    """Convert paired Student logits into the isolated PAC-NCE loss."""

    if paired.wrong_logits is None:
        raise ValueError("PAC-NCE loss requires an executed Student-Wrong forward")
    real_log_probs = sampled_token_log_prob(
        paired.real_logits, rollout, valid_token_ids
    )
    wrong_log_probs = sampled_token_log_prob(
        paired.wrong_logits, rollout, valid_token_ids
    )
    if not real_log_probs.requires_grad:
        raise RuntimeError("PAC-NCE Student-Real sampled-token graph is detached")
    if detach_wrong:
        wrong_log_probs = wrong_log_probs.detach()
        if wrong_log_probs.requires_grad:
            raise RuntimeError("PAC-1S Student-Wrong sampled-token graph was not detached")
    elif not wrong_log_probs.requires_grad:
        raise RuntimeError("PAC-2S Student-Wrong sampled-token graph is detached")
    losses, row_audits = paired_audio_condition_nce_rows(
        real_log_probs,
        wrong_log_probs,
        eligible_mask,
        token_word_ids,
        tau=tau,
    )
    if not bool(torch.isfinite(losses).all().item()):
        raise FloatingPointError("non-finite PAC-NCE row loss")
    return PairedNCEResult(
        losses, row_audits, real_log_probs, wrong_log_probs
    )


class KeOPDRound2Trainer(KeOPDV2Trainer):
    """AA/OPD trainer with an opt-in, trainable paired audio NCE branch.

    This class intentionally supports only the exact Word-Joint AA path needed
    by the proposed Round 2.  The legacy selector/mode matrix remains owned by
    :class:`KeOPDV2Trainer` and is not widened here.
    """

    def __init__(
        self,
        *args: Any,
        pac_nce_config: PairedAudioConditionNCEConfig,
        calibration_rollout_provider: Callable[[str], Sequence[int]] | None = None,
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
                    f"Round-2 trainer owns {reserved}; do not pass a Round-1 contrast option"
                )
        if kwargs.get("audit_every_microbatch") is not True:
            raise ValueError("formal Round-2 trainer requires every-microbatch audit")
        super().__init__(
            *args,
            contrast_mode="none",
            contrast_beta=0.0,
            contrast_margin=0.10,
            contrast_teacher_ratio_threshold=2.0,
            contrast_temperature=1.0,
            contrast_ramp_start=pac_nce_config.ramp_start,
            contrast_ramp_end=pac_nce_config.ramp_end,
            **kwargs,
        )
        if self.v2_arm != "AA":
            raise ValueError("Round-2 PAC-NCE requires the AA arm")
        if self.v2_aa_selector != "joint_top_fraction":
            raise ValueError("Round-2 PAC-NCE requires Word-Joint Top-Fraction AA")
        self.v2_round2_aa_contract_checks = validate_round2_aa_contract(
            self.v2_contract
        )
        self.v2_round2_teacher_contract_checks = validate_round2_teacher_contract(
            self.model, self.v2_teacher
        )
        for sample_id, gate_value in self.v2_teacher_gate.items():
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("Round-2 Teacher gate IDs must be non-empty strings")
            if type(gate_value) is not bool:
                raise ValueError(
                    "Round-2 Teacher gate values must be exact booleans; "
                    f"id={sample_id!r}"
                )
        self.v2_pac_nce_config = pac_nce_config
        self.v2_pac_wrong_gradient_mode = pac_nce_config.wrong_gradient_mode
        if calibration_rollout_provider is not None and not pac_nce_config.engineering_calibration:
            raise ValueError("a frozen rollout provider is calibration-only")
        self._v2_calibration_rollout_provider = calibration_rollout_provider
        self._v2_calibration_current_sample_id: str | None = None
        self.v2_contrast_mode = PAC_NCE_OBJECTIVE_VERSION
        self.v2_contrast_beta = float(pac_nce_config.beta)
        self.v2_contrast_temperature = float(pac_nce_config.tau)
        self.v2_contrast_ramp_start = int(pac_nce_config.ramp_start)
        self.v2_contrast_ramp_end = int(pac_nce_config.ramp_end)
        # Populated only by ``calibration_loss_components``.  Formal training
        # never reads or mutates this protected capture channel.
        self._v2_pac_calibration_rows: list[PACCalibrationRowCapture] | None = None
        self._v2_pac_gradient_audited_steps: set[int] = set()
        self._v2_pac_gradient_pending_steps: set[int] = set()

    def _register_pac_gradient_audit(
        self, *, sample_id: str, result: PairedNCEResult,
    ) -> None:
        """Emit one rank-0, post-backward Real/Wrong gradient proof per step."""

        step = int(self.state.global_step)
        if (
            not self.is_world_process_zero()
            or step in self._v2_pac_gradient_audited_steps
            or step in self._v2_pac_gradient_pending_steps
        ):
            return
        self._v2_pac_gradient_pending_steps.add(step)
        observed: dict[str, dict[str, float]] = {}

        # Preserve the historical two-sided default for legacy audit fixtures
        # instantiated before ``wrong_gradient_mode`` existed.
        config = getattr(self, "v2_pac_nce_config", None)
        mode = getattr(
            self,
            "v2_pac_wrong_gradient_mode",
            getattr(config, "wrong_gradient_mode", "bidirectional"),
        )

        def emit_if_complete() -> None:
            expected = {"real", "wrong"} if mode == "bidirectional" else {"real"}
            if set(observed) != expected:
                return
            real = observed["real"]
            wrong = observed.get(
                "wrong", {"finite": 1.0, "nonzero": 0.0, "norm": 0.0}
            )
            two_sided_pass = float(
                mode == "bidirectional"
                and real["finite"] == 1.0 and real["nonzero"] == 1.0
                and wrong["finite"] == 1.0 and wrong["nonzero"] == 1.0
            )
            one_sided_pass = float(
                mode == "detach_wrong"
                and real["finite"] == 1.0 and real["nonzero"] == 1.0
                and not result.wrong_log_probs.requires_grad
            )
            print(
                "PAC_GRAD_AUDIT=" + json.dumps({
                    "scope": "rank0_first_active_row_per_optimizer_step",
                    "run_name": self.v2_audit_run_name,
                    "attempt_id": self.v2_audit_attempt_id,
                    "global_step_before_update": step,
                    "optimizer_step": step + 1, "id": sample_id,
                    "wrong_gradient_mode": mode,
                    "real_sampled_logprob_gradient_finite": real["finite"],
                    "real_sampled_logprob_gradient_nonzero": real["nonzero"],
                    "real_sampled_logprob_gradient_norm": real["norm"],
                    "wrong_sampled_logprob_requires_grad": float(
                        result.wrong_log_probs.requires_grad
                    ),
                    "wrong_sampled_logprob_gradient_finite": wrong["finite"],
                    "wrong_sampled_logprob_gradient_nonzero": wrong["nonzero"],
                    "wrong_sampled_logprob_gradient_norm": wrong["norm"],
                    "both_branches_gradient_contract_passed": two_sided_pass,
                    "one_sided_gradient_contract_passed": one_sided_pass,
                    "variant_gradient_contract_passed": max(
                        two_sided_pass, one_sided_pass
                    ),
                }, allow_nan=False),
                flush=True,
            )
            self._v2_pac_gradient_pending_steps.discard(step)
            self._v2_pac_gradient_audited_steps.add(step)

        def capture(branch: str):
            def hook(gradient: torch.Tensor) -> torch.Tensor:
                detached = gradient.detach()
                finite = bool(torch.isfinite(detached).all().item())
                norm = float(detached.float().norm().item()) if finite else float("nan")
                observed[branch] = {
                    "finite": float(finite), "nonzero": float(finite and norm > 0.0),
                    "norm": norm,
                }
                emit_if_complete()
                return gradient
            return hook

        result.real_log_probs.register_hook(capture("real"))
        if mode == "bidirectional":
            if not result.wrong_log_probs.requires_grad:
                raise RuntimeError("PAC-2S Wrong sampled log-probability lost its graph")
            result.wrong_log_probs.register_hook(capture("wrong"))
        elif result.wrong_log_probs.requires_grad:
            raise RuntimeError("PAC-1S Wrong sampled log-probability retained a graph")

    def _effective_contrast_beta(self) -> float:
        return self.v2_pac_nce_config.effective_beta(int(self.state.global_step))

    def _generate_rollout(
        self, model: torch.nn.Module, prompt: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        provider = self._v2_calibration_rollout_provider
        sample_id = self._v2_calibration_current_sample_id
        if provider is None:
            return super()._generate_rollout(model, prompt)
        if not self.v2_pac_nce_config.engineering_calibration or not sample_id:
            raise RuntimeError("frozen PAC rollout provider used outside a calibration row")
        token_ids = tuple(provider(sample_id))
        if not token_ids or any(type(value) is not int or value < 0 for value in token_ids):
            raise ValueError(f"invalid frozen PAC rollout for {sample_id}")
        return torch.tensor(
            [token_ids], dtype=torch.long, device=prompt["input_ids"].device
        )

    def _require_teacher_gate_value(self, row: dict[str, Any]) -> bool:
        """Resolve one gate ID without the inherited missing->False fallback."""

        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Round-2 training row requires a non-empty string ID")
        if sample_id not in self.v2_teacher_gate:
            raise KeyError(
                "Round-2 Teacher gate is missing a training-row ID: "
                f"{sample_id}"
            )
        gate_value = self.v2_teacher_gate[sample_id]
        if type(gate_value) is not bool:
            raise ValueError(
                "Round-2 Teacher gate value must be an exact boolean: "
                f"id={sample_id!r}, value={gate_value!r}"
            )
        return gate_value

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: list[dict[str, Any]],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fail closed before I/O, then retain inherited full-row reduction.

        PAC-NCE is row-balanced: each eligible row first averages equally over
        its complete eligible words; the inherited trainer then averages the
        resulting vector over *every* row in the local microbatch.  Empty-gate
        and Teacher-false rows therefore remain explicit zeros in that full
        denominator.  DDP subsequently averages gradients across processes;
        it does not change this per-process row-reduction definition.
        """

        for row in inputs:
            if not isinstance(row, dict):
                raise ValueError("Round-2 training inputs must be row dictionaries")
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

    def calibration_loss_components(
        self,
        model: torch.nn.Module,
        inputs: list[dict[str, Any]],
    ) -> PACCalibrationComponents:
        """Return B and raw P from one shared calibration-only forward graph.

        This entry point is deliberately unavailable to formal training.  It
        retains Teacher-false and empty-gate rows as explicit zeros in the
        *full local-row denominator*, and captures the attached Real/Wrong
        sampled-token tensors needed for branch VJPs.  It never performs an
        optimizer update.
        """

        if not self.v2_pac_nce_config.engineering_calibration:
            raise RuntimeError("PAC component capture is calibration-only")
        if not self.v2_pac_nce_config.enabled:
            raise RuntimeError("PAC component capture requires the enabled objective")
        if int(self.state.global_step) != 1:
            raise RuntimeError("PAC calibration capture freezes logical step=1")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("PAC calibration requires a non-empty local microbatch")
        if self._v2_pac_calibration_rows is not None:
            raise RuntimeError("nested PAC calibration capture is forbidden")
        direct_rows: list[torch.Tensor] = []
        opd_rows: list[torch.Tensor] = []
        pac_rows: list[torch.Tensor] = []
        captures: list[PACCalibrationRowCapture] = []
        self._v2_pac_calibration_rows = captures
        try:
            for row in inputs:
                if not isinstance(row, dict):
                    raise ValueError("PAC calibration rows must be dictionaries")
                teacher_correct = self._require_teacher_gate_value(row)
                real_waveform = load_audio(row["audio_path"], self.v2_sampling_rate)
                direct = self._direct_loss(model, row, real_waveform)
                direct_rows.append(direct)
                if teacher_correct:
                    self._v2_calibration_current_sample_id = str(row["id"])
                    try:
                        opd, pac, _ = self._opd_loss(model, row, real_waveform)
                    finally:
                        self._v2_calibration_current_sample_id = None
                    opd_rows.append(opd)
                    pac_rows.append(pac)
                else:
                    zero = direct.reshape(1) * 0.0
                    opd_rows.append(zero)
                    pac_rows.append(zero)
                    captures.append(
                        PACCalibrationRowCapture(
                            sample_id=str(row["id"]),
                            teacher_correct=False,
                            rollout_token_ids=(), complete_word_mask=None,
                            token_word_ids=(), paired_result=None,
                            wrong_graph_executed=False,
                        )
                    )
        finally:
            self._v2_pac_calibration_rows = None
        if len(captures) != len(inputs):
            raise AssertionError("PAC calibration capture lost a full-denominator row")
        direct_tensor = torch.cat([value.reshape(1) for value in direct_rows])
        opd_tensor = torch.cat([value.reshape(1) for value in opd_rows])
        pac_tensor = torch.cat([value.reshape(1) for value in pac_rows])
        base = combined_v2_loss(
            direct_tensor, opd_tensor, lambda_opd=self.v2_contract.lambda_opd
        )
        pac = pac_tensor.mean()
        if not bool(torch.isfinite(base).item()) or not bool(torch.isfinite(pac).item()):
            raise FloatingPointError("non-finite PAC calibration component")
        return PACCalibrationComponents(
            base_loss=base, pac_loss=pac, rows=tuple(captures),
            full_denominator_rows=len(inputs),
            wrong_graph_rows=sum(row.wrong_graph_executed for row in captures),
        )

    def _opd_loss(
        self,
        model: torch.nn.Module,
        row: dict[str, Any],
        real_waveform: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Compute frozen AA-RKLD plus the isolated paired NCE row loss."""

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
            raise FloatingPointError("non-finite PAC-NCE Teacher-Real logits")
        if not bool(torch.isfinite(teacher_wrong).all().item()):
            raise FloatingPointError("non-finite PAC-NCE Teacher-Wrong logits")

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
        raw_weights = joint_top_fraction_word_weights(
            log_ratios[0].tolist(),
            support[0].tolist(),
            word_ids,
            high_weight=self.v2_contract.aa_high_weight,
            top_fraction=self.v2_contract.aa_top_fraction,
        )
        advantages = torch.where(support, log_ratios, torch.zeros_like(log_ratios))
        weights = torch.tensor(
            raw_weights, dtype=torch.float32, device=log_ratios.device
        ).unsqueeze(0)
        selected_mask = weights.gt(1)
        eligible_mask, _, gate_audits = strict_aa_word_contrast_targets(
            log_ratios,
            selected_mask,
            support,
            [word_ids],
            margin_cap=0.10,
        )
        decision = decide_pac_nce_execution(
            self.v2_pac_nce_config,
            global_step=int(self.state.global_step),
            eligible_mask=eligible_mask,
        )
        # The Wrong Teacher logits have served both selector and strict gate;
        # release them before retaining two trainable Student graphs.
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
                self.v2_pac_nce_config.wrong_gradient_mode == "bidirectional"
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
            raise FloatingPointError("non-finite Round-2 AA-RKLD")
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
                tau=self.v2_pac_nce_config.tau,
                detach_wrong=(
                    self.v2_pac_nce_config.wrong_gradient_mode == "detach_wrong"
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
                raise AssertionError("PAC-NCE objective and Teacher gate populations differ")
            paired_stats.update(nce_audit)
        if not bool(torch.isfinite(contrast).all().item()):
            raise FloatingPointError("non-finite paired audio-condition NCE")

        calibration_rows = getattr(self, "_v2_pac_calibration_rows", None)
        if calibration_rows is not None:
            calibration_rows.append(
                PACCalibrationRowCapture(
                    sample_id=str(row["id"]),
                    teacher_correct=True,
                    rollout_token_ids=tuple(int(value) for value in rollout[0].tolist()),
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
            max(1, math.ceil(lexical_words * self.v2_contract.aa_top_fraction))
            if lexical_words
            else 0
        )
        rollout_text = self.processor.tokenizer.decode(
            rollout[0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        contrast_value = float(contrast.detach().item())
        metrics: dict[str, Any] = {
            **gate_audits[0],
            **decision.audit(),
            **paired.audit,
            **paired_stats,
            "contrast_objective_version": PAC_NCE_OBJECTIVE_VERSION,
            "paired_nce_gate_version": PAC_NCE_GATE_VERSION,
            "paired_nce_enabled": float(self.v2_pac_nce_config.enabled),
            "paired_nce_tau": float(self.v2_pac_nce_config.tau),
            "paired_nce_wrong_gradient_mode": (
                self.v2_pac_nce_config.wrong_gradient_mode
            ),
            "paired_nce_beta_target": float(self.v2_pac_nce_config.beta),
            "paired_nce_engineering_calibration": float(
                self.v2_pac_nce_config.engineering_calibration
            ),
            "paired_nce_computed": float(decision.execute_wrong),
            "paired_nce_statistics_defined": float(decision.execute_wrong),
            "paired_nce_reduction_version": PAC_NCE_REDUCTION_VERSION,
            "paired_nce_reduction_semantics": (
                "equal_complete_words_within_row_then_all_local_microbatch_rows;"
                "empty_gate_and_teacher_false_rows_are_zero"
            ),
            "paired_nce_objective_row_computed": float(decision.execute_wrong),
            "contrast_eligible_row": float(eligible_mask.any().item()),
            # Compatibility only for inherited audit consumers.  PAC's smooth
            # logistic loss has no hinge-like active/inactive state.
            "contrast_active_row": float(
                decision.execute_wrong
            ),
            "contrast_active_row_deprecated": 1.0,
            "contrast_active_row_semantics": (
                "deprecated_alias_of_paired_nce_objective_row_computed_not_hinge_activity"
            ),
            "contrast_loss": contrast_value,
            "contrast_mode": PAC_NCE_OBJECTIVE_VERSION,
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
            "aa_top_fraction": float(self.v2_contract.aa_top_fraction),
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
            "aa_ratio_threshold": 2.0,
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
            raise FloatingPointError("non-finite Round-2 weighted AA-RKLD")
        return opd, contrast, metrics


__all__ = [
    "PAC_NCE_AUDIO_INPUT_KEYS",
    "PAC_NCE_GATE_VERSION",
    "PAC_NCE_OBJECTIVE_VERSION",
    "PAC_NCE_REDUCTION_VERSION",
    "KeOPDRound2Trainer",
    "PACNCEExecutionDecision",
    "PACCalibrationComponents",
    "PACCalibrationRowCapture",
    "PairedAudioConditionNCEConfig",
    "PairedNCEResult",
    "PairedStudentForward",
    "decide_pac_nce_execution",
    "pac_nce_from_paired_forward",
    "paired_student_condition_forward",
    "validate_round2_aa_contract",
    "validate_round2_teacher_contract",
]
