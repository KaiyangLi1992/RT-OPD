"""On-policy Trainer implementing AO, Uniform RKLD, and AA-weighted RKLD."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Collection
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import StoppingCriteria, StoppingCriteriaList, Trainer

from .contract import V2Contract
from .losses import (
    aa_selected_word_contrastive_rows,
    answer_span_cross_entropy,
    combined_v2_loss,
    joint_top_fraction_word_weights,
    joint_word_ratio_weights,
    reverse_kl_per_position,
    sampled_token_audio_advantage,
    sampled_token_audio_log_ratio,
    sampled_token_log_prob,
    stopword_prefilter_top_fraction_word_weights,
    strict_aa_word_contrast_targets,
    teacher_distribution_contrastive_rows,
    token_top_fraction_weights,
    top_fraction_word_weights,
    weighted_position_mean,
)
from .modeling import (
    append_completion,
    completion_word_ids,
    completion_word_texts,
    first_complete_answer_end,
    load_audio,
    match_waveform_length,
    prepare_prompt_inputs,
)


class JsonlRows(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.rows = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def raw_row_collator(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return features


class CompleteAnswerStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: Any, prompt_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        text = self.tokenizer.decode(
            input_ids[0, self.prompt_tokens :],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return "</answer>" in text


class KeOPDV2Trainer(Trainer):
    def __init__(
        self,
        *args: Any,
        processor: Any,
        teacher: torch.nn.Module | None,
        arm: str,
        valid_token_ids: list[int],
        teacher_gate: dict[str, bool],
        contract: V2Contract,
        aa_selector: str = "top_fraction",
        aa_ratio_threshold: float = 2.0,
        aa_stopwords: Collection[str] | None = None,
        scheduler_horizon: int | None = None,
        contrast_mode: str = "none",
        contrast_beta: float = 0.0,
        contrast_margin: float = 0.10,
        contrast_teacher_ratio_threshold: float = 2.0,
        contrast_temperature: float = 1.0,
        contrast_ramp_start: int = 32,
        contrast_ramp_end: int = 80,
        audit_run_name: str | None = None,
        audit_attempt_id: str | None = None,
        audit_every_microbatch: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if arm not in {"AO", "U", "AA"}:
            raise ValueError(f"unknown V2 arm {arm}")
        if arm != "AO" and teacher is None:
            raise ValueError(f"{arm} requires a Teacher")
        if aa_selector not in {
            "top_fraction",
            "joint_top_fraction",
            "token_top_fraction",
            "stopword_prefilter",
            "word_joint_ratio",
        }:
            raise ValueError(f"unknown AA selector {aa_selector}")
        if aa_ratio_threshold <= 1:
            raise ValueError("AA ratio threshold must be greater than 1")
        if aa_selector == "stopword_prefilter" and not aa_stopwords:
            raise ValueError("stopword_prefilter requires a non-empty stopword set")
        if contrast_mode not in {"none", "aa_word", "teacher_distribution"}:
            raise ValueError(f"unknown contrast mode {contrast_mode}")
        if contrast_mode == "none" and contrast_beta != 0:
            raise ValueError("contrast mode none requires beta=0")
        if contrast_mode != "none" and (arm != "AA" or contrast_beta <= 0):
            raise ValueError("active contrast requires AA and beta>0")
        if contrast_mode != "none" and aa_selector != "joint_top_fraction":
            raise ValueError("Round 1 contrast requires the word-wise joint selector")
        if contrast_mode == "aa_word" and not 0 < contrast_margin <= 0.10:
            raise ValueError("AA-word contrast margin cap must be in (0, 0.10]")
        if contrast_mode == "teacher_distribution" and contrast_margin < 0:
            raise ValueError("distribution contrast margin must be non-negative")
        if contrast_teacher_ratio_threshold != 2.0:
            raise ValueError("Round 1 freezes the strict Teacher word ratio at 2.0")
        if contrast_temperature <= 0:
            raise ValueError("contrast temperature must be positive")
        if not 0 <= contrast_ramp_start < contrast_ramp_end:
            raise ValueError("invalid contrast beta ramp")
        self.processor = processor
        self.v2_teacher = teacher
        self.v2_arm = arm
        self.v2_contract = contract
        self.v2_valid_ids = torch.tensor(valid_token_ids, dtype=torch.long, device=self.args.device)
        self.v2_forbidden_ids = sorted(set(range(151_936)) - set(valid_token_ids))
        self.v2_teacher_gate = teacher_gate
        self.v2_aa_selector = aa_selector
        self.v2_aa_ratio_threshold = float(aa_ratio_threshold)
        self.v2_aa_stopwords = frozenset(
            str(word).casefold() for word in (aa_stopwords or ())
        )
        self.v2_calls = 0
        self.v2_sampling_rate = int(processor.feature_extractor.sampling_rate)
        self.v2_scheduler_horizon = int(
            scheduler_horizon if scheduler_horizon is not None else self.args.max_steps
        )
        if self.v2_scheduler_horizon < int(self.args.max_steps):
            raise ValueError("scheduler horizon cannot precede the execution stop")
        self.v2_contrast_mode = contrast_mode
        self.v2_contrast_beta = float(contrast_beta)
        self.v2_contrast_margin = float(contrast_margin)
        self.v2_contrast_temperature = float(contrast_temperature)
        self.v2_contrast_ramp_start = int(contrast_ramp_start)
        self.v2_contrast_ramp_end = int(contrast_ramp_end)
        self.v2_audit_run_name = str(audit_run_name or "unregistered")
        self.v2_audit_attempt_id = str(audit_attempt_id or f"pid-{os.getpid()}")
        self.v2_audit_interval_calls = 1 if audit_every_microbatch else 10
        if not self.v2_audit_run_name or not self.v2_audit_attempt_id:
            raise ValueError("training audit run/attempt identity must be non-empty")

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        """Decouple the registered cosine horizon from the execution stop."""
        if num_training_steps != int(self.args.max_steps):
            raise ValueError(
                "Trainer requested an unexpected scheduler length: "
                f"{num_training_steps} != {self.args.max_steps}"
            )
        return super().create_scheduler(
            self.v2_scheduler_horizon,
            optimizer=optimizer,
        )

    def _effective_contrast_beta(self) -> float:
        if self.v2_contrast_mode == "none":
            return 0.0
        step = int(self.state.global_step)
        if step < self.v2_contrast_ramp_start:
            return 0.0
        if step >= self.v2_contrast_ramp_end:
            return self.v2_contrast_beta
        progress = (step - self.v2_contrast_ramp_start) / (
            self.v2_contrast_ramp_end - self.v2_contrast_ramp_start
        )
        return self.v2_contrast_beta * progress

    def _rng_state(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        cpu = torch.get_rng_state()
        cuda = (
            torch.cuda.get_rng_state(self.args.device)
            if self.args.device.type == "cuda"
            else None
        )
        return cpu, cuda

    def _set_rng_state(
        self, state: tuple[torch.Tensor, torch.Tensor | None]
    ) -> None:
        cpu, cuda = state
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, self.args.device)

    def _set_signature_columns_if_needed(self) -> None:
        if self._signature_columns is None:
            self._signature_columns = [
                "id", "audio_path", "wrong_audio_path", "canonical_user_text",
                "direct_user_text", "direct_completion"
            ]

    def _prepare_inputs(self, inputs: Any) -> Any:
        return inputs

    def _direct_loss(self, model: torch.nn.Module, row: dict[str, Any], waveform: Any) -> torch.Tensor:
        prompt = prepare_prompt_inputs(
            self.processor,
            row["direct_user_text"],
            waveform,
            self.v2_sampling_rate,
            self.args.device,
        )
        completion_text = str(row["direct_completion"])
        encoded = self.processor.tokenizer(
            completion_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        completion = list(encoded["input_ids"])
        completion_ids = torch.tensor(completion, dtype=torch.long, device=self.args.device)
        model_inputs = append_completion(prompt, completion_ids)
        labels = torch.full_like(model_inputs["input_ids"], -100)
        answer_start = completion_text.index("<answer>")
        answer_end = len(completion_text)
        supervised_offsets = [
            index
            for index, pair in enumerate(encoded["offset_mapping"])
            if max(int(pair[0]), answer_start) < min(int(pair[1]), answer_end)
        ]
        if not supervised_offsets:
            raise ValueError(f"direct answer span has no tokens for {row['id']}")
        completion_start = int(prompt["input_ids"].shape[1])
        for index in supervised_offsets:
            labels[:, completion_start + index] = model_inputs["input_ids"][:, completion_start + index]
        outputs = model(**model_inputs, use_cache=False, return_dict=True)
        return answer_span_cross_entropy(outputs.logits, labels)

    def _generate_rollout(
        self, model: torch.nn.Module, prompt: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        unwrapped = self.accelerator.unwrap_model(model)
        prompt_length = int(prompt["input_ids"].shape[1])
        # Trainer keeps the Student in train mode. Qwen disables KV caching when
        # gradient checkpointing and train mode are both active, which would
        # recompute the entire audio prefix at every rollout token. Rollout has
        # no gradient, so freeze dropout and enable the cache in eval mode.
        was_training = unwrapped.training
        unwrapped.eval()
        try:
            with torch.inference_mode():
                sequences = unwrapped.generate(
                    **prompt,
                    do_sample=True,
                    temperature=self.v2_contract.rollout_temperature,
                    top_p=self.v2_contract.rollout_top_p,
                    top_k=self.v2_contract.rollout_top_k,
                    max_new_tokens=self.v2_contract.max_completion_tokens,
                    use_cache=True,
                    suppress_tokens=self.v2_forbidden_ids,
                    stopping_criteria=StoppingCriteriaList(
                        [CompleteAnswerStoppingCriteria(self.processor.tokenizer, prompt_length)]
                    ),
                )
        finally:
            unwrapped.train(was_training)
        generated = sequences[:, prompt_length:]
        end = first_complete_answer_end(self.processor.tokenizer, generated[0])
        if end <= 0:
            raise ValueError("Student rollout produced no completion tokens")
        return generated[:, :end]

    def _opd_loss(
        self,
        model: torch.nn.Module,
        row: dict[str, Any],
        real_waveform: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        real_prompt = prepare_prompt_inputs(
            self.processor,
            row["canonical_user_text"],
            real_waveform,
            self.v2_sampling_rate,
            self.args.device,
        )
        rollout = self._generate_rollout(model, real_prompt)
        real_sequence = append_completion(real_prompt, rollout)
        paired_rng_before = (
            self._rng_state() if self.v2_contrast_mode == "aa_word" else None
        )
        student_outputs = model(**real_sequence, use_cache=False, return_dict=True)
        paired_rng_after = (
            self._rng_state() if paired_rng_before is not None else None
        )
        prompt_length = int(real_prompt["input_ids"].shape[1])
        # The logit immediately before each sampled token predicts that token.
        start = prompt_length - 1
        stop = start + rollout.shape[1]
        student_logits = student_outputs.logits[:, start:stop]

        assert self.v2_teacher is not None
        with torch.inference_mode():
            teacher_real = self.v2_teacher(**real_sequence, use_cache=False, return_dict=True).logits[
                :, start:stop
            ]
        rkl = reverse_kl_per_position(
            student_logits,
            teacher_real,
            self.v2_valid_ids,
            temperature=self.v2_contract.kl_temperature,
        )
        mask = torch.ones_like(rkl, dtype=torch.bool)
        weights = torch.ones_like(rkl, dtype=torch.float32)
        contrast = student_logits.sum().reshape(1) * 0.0
        metrics = {"rollout_tokens": float(rollout.numel()), "aa_high_tokens": 0.0}

        if self.v2_arm == "AA":
            wrong_waveform = load_audio(row["wrong_audio_path"], self.v2_sampling_rate)
            wrong_waveform = match_waveform_length(wrong_waveform, len(real_waveform))
            wrong_prompt = prepare_prompt_inputs(
                self.processor,
                row["canonical_user_text"],
                wrong_waveform,
                self.v2_sampling_rate,
                self.args.device,
            )
            if not torch.equal(real_prompt["input_ids"], wrong_prompt["input_ids"]):
                raise ValueError(f"Real/Wrong serialized prefixes differ for {row['id']}")
            wrong_sequence = append_completion(wrong_prompt, rollout)
            with torch.inference_mode():
                teacher_wrong = self.v2_teacher(
                    **wrong_sequence, use_cache=False, return_dict=True
                ).logits[:, start:stop]
            word_ids = completion_word_ids(self.processor.tokenizer, rollout[0].tolist())
            selector_metrics: dict[str, Any] = {}
            rollout_text = self.processor.tokenizer.decode(
                rollout[0], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if self.v2_aa_selector in {"joint_top_fraction", "word_joint_ratio"}:
                log_ratios, support = sampled_token_audio_log_ratio(
                    teacher_real,
                    teacher_wrong,
                    rollout,
                    self.v2_valid_ids,
                    support_top_k=self.v2_contract.teacher_support_top_k,
                )
                if self.v2_aa_selector == "word_joint_ratio":
                    raw_weights = joint_word_ratio_weights(
                        log_ratios[0].tolist(),
                        support[0].tolist(),
                        word_ids,
                        high_weight=self.v2_contract.aa_high_weight,
                        ratio_threshold=self.v2_aa_ratio_threshold,
                    )
                else:
                    raw_weights = joint_top_fraction_word_weights(
                        log_ratios[0].tolist(),
                        support[0].tolist(),
                        word_ids,
                        high_weight=self.v2_contract.aa_high_weight,
                        top_fraction=self.v2_contract.aa_top_fraction,
                    )
                advantages = torch.where(support, log_ratios, torch.zeros_like(log_ratios))
            else:
                advantages, support = sampled_token_audio_advantage(
                    teacher_real,
                    teacher_wrong,
                    rollout,
                    self.v2_valid_ids,
                    support_top_k=self.v2_contract.teacher_support_top_k,
                )
                if self.v2_aa_selector == "token_top_fraction":
                    raw_weights = token_top_fraction_weights(
                        advantages[0].tolist(),
                        word_ids,
                        high_weight=self.v2_contract.aa_high_weight,
                        top_fraction=self.v2_contract.aa_top_fraction,
                    )
                elif self.v2_aa_selector == "stopword_prefilter":
                    word_texts = completion_word_texts(
                        self.processor.tokenizer, rollout[0].tolist()
                    )
                    raw_weights, selector_metrics = (
                        stopword_prefilter_top_fraction_word_weights(
                            advantages[0].tolist(),
                            word_ids,
                            word_texts,
                            self.v2_aa_stopwords,
                            high_weight=self.v2_contract.aa_high_weight,
                            top_fraction=self.v2_contract.aa_top_fraction,
                        )
                    )
                else:
                    raw_weights = top_fraction_word_weights(
                        advantages[0].tolist(),
                        word_ids,
                        high_weight=self.v2_contract.aa_high_weight,
                        top_fraction=self.v2_contract.aa_top_fraction,
                    )
            weights = torch.tensor(raw_weights, dtype=torch.float32, device=rkl.device).unsqueeze(0)
            selected_mask = weights.gt(1)
            eligible_mask: torch.Tensor | None = None
            metrics.update(
                {
                    "contrast_active_row": 0.0,
                    "contrast_loss": 0.0,
                    "additional_student_wrong_forward": 0.0,
                    "paired_student_wrong_rng_restored": 0.0,
                }
            )
            if self.v2_aa_selector in {"joint_top_fraction", "word_joint_ratio"}:
                eligible_mask, _, contrast_audits = strict_aa_word_contrast_targets(
                    log_ratios,
                    selected_mask,
                    support,
                    [word_ids],
                    margin_cap=(
                        self.v2_contrast_margin
                        if self.v2_contrast_mode == "aa_word"
                        else 0.10
                    ),
                )
                metrics.update(contrast_audits[0])
                metrics["contrast_eligible_row"] = float(eligible_mask.any())
            if self.v2_contrast_mode != "none":
                if eligible_mask is None:
                    raise AssertionError("active contrast lacks strict Teacher targets")
                if self.v2_contrast_mode == "aa_word":
                    if paired_rng_before is None or paired_rng_after is None:
                        raise AssertionError("paired dropout RNG was not captured")
                    self._set_rng_state(paired_rng_before)
                    try:
                        with torch.no_grad():
                            student_wrong = model(
                                **wrong_sequence,
                                use_cache=False,
                                return_dict=True,
                            ).logits[:, start:stop]
                    finally:
                        # The additional Wrong forward must not perturb the
                        # registered global RNG/data trajectory.
                        self._set_rng_state(paired_rng_after)
                    restored_rng = self._rng_state()
                    if not torch.equal(restored_rng[0], paired_rng_after[0]) or (
                        restored_rng[1] is not None
                        and paired_rng_after[1] is not None
                        and not torch.equal(restored_rng[1], paired_rng_after[1])
                    ):
                        raise RuntimeError(
                            "paired Student-Wrong forward did not restore RNG"
                        )
                    student_real_logp = sampled_token_log_prob(
                        student_logits,
                        rollout,
                        self.v2_valid_ids,
                    )
                    student_wrong_logp = sampled_token_log_prob(
                        student_wrong,
                        rollout,
                        self.v2_valid_ids,
                    )
                    contrast = aa_selected_word_contrastive_rows(
                        student_real_logp,
                        student_wrong_logp,
                        log_ratios,
                        selected_mask,
                        support,
                        [word_ids],
                        margin_cap=self.v2_contrast_margin,
                    )
                    student_gap = student_real_logp - student_wrong_logp.detach()
                    metrics.update(
                        {
                            "contrast_student_gap_nat_per_token": float(
                                student_gap[eligible_mask].mean().detach().item()
                                if eligible_mask.any()
                                else 0.0
                            ),
                            "contrast_active_row": float(
                                contrast.detach().item() > 0
                            ),
                            "additional_student_wrong_forward": 1.0,
                            "paired_student_wrong_rng_restored": 1.0,
                        }
                    )
                elif self.v2_contrast_mode == "teacher_distribution":
                    wrong_rkl = reverse_kl_per_position(
                        student_logits,
                        teacher_wrong,
                        self.v2_valid_ids,
                        temperature=self.v2_contract.kl_temperature,
                    )
                    contrast = teacher_distribution_contrastive_rows(
                        rkl,
                        wrong_rkl,
                        log_ratios,
                        selected_mask,
                        support,
                        [word_ids],
                        margin=self.v2_contrast_margin,
                        temperature=self.v2_contrast_temperature,
                    )
                    metrics.update(
                        {
                            "contrast_real_teacher_kl": float(
                                rkl[eligible_mask].mean().detach().item()
                                if eligible_mask.any()
                                else 0.0
                            ),
                            "contrast_wrong_teacher_kl": float(
                                wrong_rkl[eligible_mask].mean().detach().item()
                                if eligible_mask.any()
                                else 0.0
                            ),
                            "contrast_active_row": float(
                                contrast.detach().item() > 0
                            ),
                        }
                    )
                metrics["contrast_loss"] = float(contrast.detach().item())
                metrics["contrast_mode"] = self.v2_contrast_mode
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
            lexical_words = len(word_scores)
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
            token_selector = self.v2_aa_selector == "token_top_fraction"
            top_denominator = lexical_tokens if token_selector else lexical_words
            top_limit = (
                max(
                    1,
                    math.ceil(
                        top_denominator * self.v2_contract.aa_top_fraction
                    ),
                )
                if top_denominator
                else 0
            )
            metrics.update(
                {
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
                    "partial_word_weight_mismatch": float(
                        partial_word_weight_mismatch
                    ),
                    "aa_selector": self.v2_aa_selector,
                    "aa_selection_unit": "token" if token_selector else "word",
                    "aa_top_denominator_units": float(top_denominator),
                    "aa_ratio_threshold": self.v2_aa_ratio_threshold,
                    "aa_lexical_tokens": float(lexical_tokens),
                    "rollout_complete_think": float(
                        "<think>" in rollout_text.lower() and "</think>" in rollout_text.lower()
                    ),
                    "rollout_has_answer": float("<answer" in rollout_text.lower()),
                    "rollout_preview": rollout_text[:240],
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
                }
            )
            metrics.update(selector_metrics)
        return weighted_position_mean(rkl, weights, mask), contrast, metrics

    def _reduce_contrast_rows(
        self,
        contrast: torch.Tensor,
        objective_row_computed: list[bool],
    ) -> tuple[torch.Tensor, dict[str, float | str]]:
        """Reduce per-row contrast losses without changing the legacy mean.

        Variant trainers may override this hook, but the base implementation
        intentionally remains the historical all-row microbatch mean.  The
        boolean population is passed separately so a variant can distinguish
        a legitimate zero-valued objective from a row on which PAC was not
        computed.
        """

        if contrast.ndim != 1:
            raise ValueError("contrast rows must be a one-dimensional tensor")
        if len(objective_row_computed) != int(contrast.numel()):
            raise ValueError("contrast activity mask must match contrast rows")
        reduced = contrast.mean()
        return reduced, {
            "contrast_reduction_version": "all_local_microbatch_rows_mean_v1",
            "contrast_reduction_scale": 1.0,
            "contrast_objective_rows_local": float(sum(objective_row_computed)),
            "contrast_total_rows_local": float(len(objective_row_computed)),
        }

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: list[dict[str, Any]],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if return_outputs:
            raise ValueError("KeOPDV2Trainer does not return model outputs")
        direct_rows: list[torch.Tensor] = []
        opd_rows: list[torch.Tensor] = []
        contrast_rows: list[torch.Tensor] = []
        contrast_objective_rows: list[bool] = []
        global_step_before_update = int(self.state.global_step)
        accumulation_slot = int(
            self.v2_calls % int(self.args.gradient_accumulation_steps)
        )
        logical_microbatch_index = (
            global_step_before_update * int(self.args.gradient_accumulation_steps)
            + accumulation_slot
        )
        audit_now = self.v2_calls % self.v2_audit_interval_calls == 0
        for row in inputs:
            real_waveform = load_audio(row["audio_path"], self.v2_sampling_rate)
            direct_rows.append(self._direct_loss(model, row, real_waveform))
            teacher_correct = bool(self.v2_teacher_gate.get(row["id"], False))
            metrics: dict[str, Any] = {
                "teacher_correct": float(teacher_correct),
                "opd_executed": 0.0,
                "rollout_tokens": None,
                "rollout_has_answer": None,
                "rollout_complete_think": None,
                "contrast_eligible_row": None,
                "contrast_active_row": None,
                "aa_high_token_fraction": None,
                "support_fraction": None,
                "additional_student_wrong_forward": None,
                "paired_student_wrong_rng_restored": None,
            }
            if self.v2_arm != "AO" and teacher_correct:
                opd, contrast, metrics = self._opd_loss(model, row, real_waveform)
                metrics.update(
                    {
                        "teacher_correct": 1.0,
                        "opd_executed": 1.0,
                    }
                )
                opd_rows.append(opd)
                contrast_rows.append(contrast)
                contrast_objective_rows.append(
                    bool(metrics.get("paired_nce_objective_row_computed", False))
                )
            else:
                opd_rows.append(torch.zeros(1, device=self.args.device))
                contrast_rows.append(torch.zeros(1, device=self.args.device))
                contrast_objective_rows.append(False)
            if self.is_world_process_zero() and audit_now:
                print(
                    "V2_STEP_AUDIT="
                    + json.dumps(
                        {
                            "run_name": self.v2_audit_run_name,
                            "attempt_id": self.v2_audit_attempt_id,
                            "scope": "rank0_microbatch",
                            "audit_interval_calls": self.v2_audit_interval_calls,
                            "global_step_before_update": global_step_before_update,
                            "optimizer_step": global_step_before_update + 1,
                            "gradient_accumulation_slot": accumulation_slot,
                            "logical_microbatch_index": logical_microbatch_index,
                            "compute_loss_call": int(self.v2_calls),
                            "id": row["id"],
                            "contrast_effective_beta": (
                                self._effective_contrast_beta()
                            ),
                            **metrics,
                        }
                    ),
                    flush=True,
                )
        direct = torch.cat(direct_rows)
        opd = None if self.v2_arm == "AO" else torch.cat(opd_rows)
        loss = combined_v2_loss(direct, opd, lambda_opd=self.v2_contract.lambda_opd)
        contrast = torch.cat(contrast_rows)
        reduced_contrast, contrast_reduction_audit = self._reduce_contrast_rows(
            contrast, contrast_objective_rows
        )
        effective_beta = self._effective_contrast_beta()
        if effective_beta:
            loss = loss + effective_beta * reduced_contrast
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite V2 loss at call {self.v2_calls}")
        if self.is_world_process_zero() and audit_now:
            direct_mean = direct.detach().mean()
            opd_mean = (
                torch.zeros((), device=direct_mean.device)
                if opd is None
                else opd.detach().mean()
            )
            contrast_mean = contrast.detach().mean()
            reduced_contrast_detached = reduced_contrast.detach()
            weighted_opd = float(self.v2_contract.lambda_opd) * opd_mean
            weighted_contrast = float(effective_beta) * reduced_contrast_detached
            total_detached = loss.detach()
            denominator = max(abs(float(total_detached.item())), 1e-12)
            print(
                "V2_LOSS_AUDIT="
                + json.dumps(
                    {
                        "scope": "rank0_microbatch",
                        "run_name": self.v2_audit_run_name,
                        "attempt_id": self.v2_audit_attempt_id,
                        "audit_interval_calls": self.v2_audit_interval_calls,
                        "global_step_before_update": global_step_before_update,
                        "optimizer_step": global_step_before_update + 1,
                        "gradient_accumulation_slot": accumulation_slot,
                        "logical_microbatch_index": logical_microbatch_index,
                        "compute_loss_call": int(self.v2_calls),
                        "id": (
                            str(inputs[0]["id"]) if len(inputs) == 1 else None
                        ),
                        "batch_rows": len(inputs),
                        "direct_ce_mean": float(direct_mean.item()),
                        "opd_rkl_mean": float(opd_mean.item()),
                        "weighted_opd": float(weighted_opd.item()),
                        "contrast_loss_mean": float(contrast_mean.item()),
                        "contrast_loss_reduced": float(
                            reduced_contrast_detached.item()
                        ),
                        "contrast_effective_beta": float(effective_beta),
                        "weighted_contrast": float(weighted_contrast.item()),
                        "weighted_contrast_abs_fraction_of_total": abs(
                            float(weighted_contrast.item())
                        )
                        / denominator,
                        "total_loss": float(total_detached.item()),
                        **contrast_reduction_audit,
                    }
                ),
                flush=True,
            )
        self.v2_calls += 1
        return loss
