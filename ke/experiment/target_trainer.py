"""Frozen-source OPD with a teacher-only, full-vocabulary audio target change.

The worker must put the immutable portable snapshot on sys.path before import.
This module does not modify CE, teacher correctness gating, rollouts, row means,
the optimizer, or data ordering. No student negative branch is constructed.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ke_opd_v2.losses import weighted_position_mean
from ke_opd_v2.modeling import (
    append_completion, load_audio, match_waveform_length, prepare_prompt_inputs,
)
from ke_opd_v2.rounded_x15_ablation_trainers import KeOPDV10UniformOPDTrainerV1


TARGET_ARMS = ("uniform", "linear_noaudio", "linear_donor", "sigmoid_donor")
TARGET_VERSION = "audio_target_full_valid_vocab_v1"


@torch.no_grad()
def build_teacher_target(
    real_logits: torch.Tensor,
    negative_logits: torch.Tensor | None,
    valid_token_ids: torch.Tensor,
    arm: str,
    *,
    alpha: float | None = None,
    tau: float = 1.0,
    position_chunk_size: int = 4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return detached log q on valid IDs in their declared order and audits.

    Both input branches are independently normalized on the same valid
    vocabulary BEFORE differencing. Quantiles are explicitly a deterministic
    subsample, not full-vocabulary quantiles. Target KL/entropy and ratio means
    are exact across all positions and all valid tokens. No random draws occur.
    """
    if arm not in TARGET_ARMS:
        raise ValueError(f"unknown target arm: {arm}")
    if real_logits.ndim < 2:
        raise ValueError("teacher logits need position and vocabulary dimensions")
    if tau <= 0 or position_chunk_size <= 0:
        raise ValueError("temperature and chunk size must be positive")
    if valid_token_ids.ndim != 1 or valid_token_ids.dtype != torch.long:
        raise ValueError("valid IDs must be one-dimensional int64")
    if valid_token_ids.numel() == 0:
        raise ValueError("valid vocabulary must not be empty")
    if arm == "uniform":
        strength = 0.0
    else:
        strength = (1.0 if arm == "sigmoid_donor" else 0.5) if alpha is None else float(alpha)
        if strength < 0:
            raise ValueError("enhancement strength must be nonnegative")
        if negative_logits is None or negative_logits.shape != real_logits.shape:
            raise ValueError("negative logits must align to the same completion positions")
    ids = valid_token_ids.to(real_logits.device)
    real_flat = real_logits.detach().reshape(-1, real_logits.shape[-1])
    negative_flat = None if negative_logits is None else negative_logits.detach().reshape_as(real_flat)
    if not real_flat.shape[0]:
        raise ValueError("completion must contain at least one position")
    targets: list[torch.Tensor] = []
    summaries: list[torch.Tensor] = []
    ratio_samples: list[torch.Tensor] = []
    # A bounded, deterministic vocabulary grid per position. It is diagnostic
    # only: the training target always includes the entire valid vocabulary.
    sample_stride = max(1, (int(ids.numel()) + 255) // 256)
    for begin in range(0, real_flat.shape[0], position_chunk_size):
        end = begin + position_chunk_size
        real = real_flat[begin:end].index_select(-1, ids).float()
        if not bool(torch.isfinite(real).all()):
            raise FloatingPointError("nonfinite matched teacher logits")
        log_real = F.log_softmax(real, dim=-1)
        if arm == "uniform":
            ratio = torch.zeros_like(log_real)
            log_target = log_real
        else:
            assert negative_flat is not None
            negative = negative_flat[begin:end].index_select(-1, ids).float()
            if not bool(torch.isfinite(negative).all()):
                raise FloatingPointError("nonfinite negative teacher logits")
            ratio = log_real - F.log_softmax(negative, dim=-1)
            if strength == 0.0:
                # Exact alpha-zero recovery, avoiding a second normalization.
                log_target = log_real
            else:
                correction = (F.logsigmoid(ratio / tau) if arm == "sigmoid_donor" else ratio)
                log_target = F.log_softmax(log_real + strength * correction, dim=-1)
        if not bool(torch.isfinite(log_target).all()):
            raise FloatingPointError("nonfinite enhanced teacher target")
        probability = log_target.exp()
        summaries.append(torch.stack([
            (probability * (log_target - log_real)).sum(-1),
            -(probability * log_target).sum(-1),
            -(log_real.exp() * log_real).sum(-1),
            ratio.mean(-1),
            (log_real.exp() * ratio).sum(-1),
            ratio.gt(0).float().mean(-1),
        ], dim=-1))
        ratio_samples.append(ratio[:, ::sample_stride].reshape(-1))
        targets.append(log_target)
    means = torch.cat(summaries).mean(0).cpu().tolist()
    quantiles = torch.quantile(
        torch.cat(ratio_samples),
        torch.tensor([0.05, 0.50, 0.95], device=real_logits.device),
    ).cpu().tolist()
    statistics = dict(zip((
        "target_kl_q_to_teacher_real", "target_entropy", "teacher_real_entropy",
        "teacher_log_ratio_vocab_mean", "teacher_log_ratio_real_weighted_mean",
        "teacher_log_ratio_positive_vocab_fraction",
    ), means))
    statistics.update(
        target_alpha=strength,
        target_tau=float(tau),
        target_valid_vocab_size=float(ids.numel()),
        target_completion_positions=float(real_flat.shape[0]),
        teacher_log_ratio_sample_p05=quantiles[0],
        teacher_log_ratio_sample_p50=quantiles[1],
        teacher_log_ratio_sample_p95=quantiles[2],
        teacher_log_ratio_quantile_vocab_stride=float(sample_stride),
    )
    target = torch.cat(targets).reshape(*real_logits.shape[:-1], ids.numel())
    assert not target.requires_grad
    return target, statistics


def reverse_kl_to_target(
    student_logits: torch.Tensor,
    log_target: torch.Tensor,
    valid_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Exact reverse KL at frozen KL temperature 1; teacher is always detached."""
    student = student_logits.float().index_select(-1, valid_token_ids.to(student_logits.device))
    if student.shape != log_target.shape:
        raise ValueError("student positions/vocabulary do not align with target")
    if not bool(torch.isfinite(student).all()):
        raise FloatingPointError("nonfinite student logits")
    log_student = F.log_softmax(student, dim=-1)
    result = (log_student.exp() * (log_student - log_target.detach())).sum(-1)
    if bool(torch.any(result < -2e-5)):
        raise FloatingPointError("negative reverse KL beyond rounding tolerance")
    return result.clamp_min(0.0)


def prepare_noaudio_prompt_inputs(processor: Any, user_text: str, device: torch.device) -> dict:
    """Actually omit the audio content item, audio argument, and feature tensors."""
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processed = processor(text=text, return_tensors="pt", padding=True)
    result = {key: value.to(device) for key, value in processed.items() if torch.is_tensor(value)}
    for key in ("input_features", "feature_attention_mask", "audio_feature_lengths", "audio_lengths"):
        if key in result:
            raise AssertionError(f"text-only negative unexpectedly contains {key}")
    if "input_ids" not in result or "attention_mask" not in result:
        raise AssertionError("text-only prompt lacks token IDs or attention mask")
    return result


class AudioTargetOPDTrainer(KeOPDV10UniformOPDTrainerV1):
    """Uniform CE/OPD training, changing only the frozen teacher target."""

    def __init__(self, *args: Any, target_arm: str = "uniform", target_alpha: float | None = None, **kwargs: Any) -> None:
        if target_arm not in TARGET_ARMS:
            raise ValueError(f"unknown target arm: {target_arm}")
        self.target_arm = target_arm
        self.target_alpha = target_alpha
        super().__init__(*args, **kwargs)

    def _opd_loss(self, model: torch.nn.Module, row: dict[str, Any], real_waveform: Any):
        arm = self.target_arm
        if arm not in TARGET_ARMS:
            raise ValueError(f"unknown target arm: {arm}")
        if self.v2_contract.kl_temperature != 1.0:
            raise ValueError("Phase 1 freezes KL temperature at 1")
        assert self.v2_arm == "U" and self.v2_contrast_mode == "none"
        real_prompt = prepare_prompt_inputs(
            self.processor, row["canonical_user_text"], real_waveform,
            self.v2_sampling_rate, self.args.device,
        )
        rollout = self._generate_rollout(model, real_prompt)
        real_sequence = append_completion(real_prompt, rollout)
        start = int(real_prompt["input_ids"].shape[1]) - 1
        stop = start + int(rollout.shape[1])
        # Preserve the original Uniform order: rollout, student-real, teacher-real.
        student_logits = model(**real_sequence, use_cache=False, return_dict=True).logits[:, start:stop]
        assert self.v2_teacher is not None and not self.v2_teacher.training
        with torch.inference_mode():
            teacher_real = self.v2_teacher(
                **real_sequence, use_cache=False, return_dict=True,
            ).logits[:, start:stop].clone()
        teacher_negative = None
        negative_prompt_length = 0
        if arm != "uniform":
            if arm == "linear_noaudio":
                negative_prompt = prepare_noaudio_prompt_inputs(
                    self.processor, row["canonical_user_text"], self.args.device,
                )
            else:
                negative_waveform = match_waveform_length(
                    load_audio(row["wrong_audio_path"], self.v2_sampling_rate), len(real_waveform),
                )
                negative_prompt = prepare_prompt_inputs(
                    self.processor, row["canonical_user_text"], negative_waveform,
                    self.v2_sampling_rate, self.args.device,
                )
                if not torch.equal(real_prompt["input_ids"], negative_prompt["input_ids"]):
                    raise AssertionError("matched/donor prompt IDs are not aligned")
            negative_prompt_length = int(negative_prompt["input_ids"].shape[1])
            negative_sequence = append_completion(negative_prompt, rollout)
            if not torch.equal(negative_sequence["input_ids"][:, negative_prompt_length:], rollout):
                raise AssertionError("negative branch changed the student completion")
            negative_start = negative_prompt_length - 1
            with torch.inference_mode():
                teacher_negative = self.v2_teacher(
                    **negative_sequence, use_cache=False, return_dict=True,
                ).logits[:, negative_start:negative_start + int(rollout.shape[1])].clone()
        log_target, target_statistics = build_teacher_target(
            teacher_real, teacher_negative, self.v2_valid_ids, arm,
            alpha=self.target_alpha,
        )
        del teacher_real, teacher_negative
        rkl = reverse_kl_to_target(student_logits, log_target, self.v2_valid_ids)
        mask = torch.ones_like(rkl, dtype=torch.bool)
        weights = torch.ones_like(rkl, dtype=torch.float32)
        # As in frozen Uniform, the inactive contrast is a zero student graph.
        contrast = student_logits.sum().reshape(1) * 0.0
        rollout_text = self.processor.tokenizer.decode(
            rollout[0], skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )
        metrics = dict(
            phaseb_arm_id="uniform_opd", target_arm=arm, target_version=TARGET_VERSION,
            rollout_tokens=float(rollout.numel()), aa_high_tokens=0.0,
            rollout_complete_think=float(
                "<think>" in rollout_text.lower() and "</think>" in rollout_text.lower()
            ),
            rollout_has_answer=float("<answer" in rollout_text.lower()),
            rollout_preview=rollout_text[:240],
            aa_enabled=0.0, aa_top_fraction_used=0.0, aa_ratio_gate_used=0.0,
            opd_token_weighting_uniform=1.0,
            teacher_forward_calls_observed=1 + int(arm != "uniform"),
            teacher_negative_forward_count=int(arm != "uniform"),
            teacher_wrong_forward_executed=float(arm.endswith("donor")),
            teacher_noaudio_forward_executed=float(arm == "linear_noaudio"),
            student_wrong_forward_executed=0.0, additional_student_wrong_forward=0.0,
            paired_nce_enabled=0.0, paired_nce_computed=0.0,
            negative_prompt_tokens=negative_prompt_length,
            matched_prompt_tokens=start + 1,
            negative_same_completion=float(arm != "uniform"),
            **target_statistics,
        )
        return weighted_position_mean(rkl, weights, mask), contrast, metrics
