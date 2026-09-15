#!/usr/bin/env python3
"""Train one generic V1.0 Joint-LR AA-only hyperparameter run on one GPU.

This entry point is deliberately separate from ``train_ke_opd_v2.py`` so the
frozen eight-run V1.0 authority keeps its original CLI and runtime semantics.
Only the registered hyperparameter axes are configurable here.  The selector
is frozen to the support-masked signed whole-word Joint-LR Top-q rule that won
the K/Q follow-up experiment.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Final

import torch

from ke_opd_v2.contract import V2Contract, V2RunSpec, sha256_file
from ke_opd_v2.lora import audit_trainable_parameters, discover_v2_lora_targets
from ke_opd_v2.modeling import load_processor, load_thinker
from ke_opd_v2.trainer import JsonlRows, KeOPDV2Trainer, raw_row_collator
from scripts.ke_opd_contrast_round1_provenance import (
    AUDIO_RECEIPT_CONTRACT,
    RUNTIME_RECEIPT_CONTRACT,
    load_bound_receipt,
    provenance_code_identity,
    stable_file_identity,
    validate_local_rank_runtime_binding,
)


PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
MODEL_PATHS: Final = {
    "Q": PROJECT_ROOT / "models/Qwen--Qwen2.5-Omni-3B",
    "K": PROJECT_ROOT / "models/KE-Team--Ke-Omni-R-3B",
}
TEACHER_PATH: Final = PROJECT_ROOT / "models/Ke-Team--Ke-Omni-R"
RUN_NAME_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
CHECKPOINT_NAME_RE: Final = re.compile(r"^checkpoint-([0-9]+)$")
RECOVERY_SESSION_RE: Final = re.compile(r"^[0-9a-f]{64}$")
EFFECTIVE_BATCH: Final = 32
BASE01_LONG_RUN_NAME: Final = (
    "Q-AA-M3F-V1-r64-lr1em4-q30-lam250-selwordwise-s42-long2p0"
)
CONTRAST_ROUND1_RUNS: Final = {
    "Q-AA-CTR-R1-control-s42-h626cut470": ("none", 0.0),
    "Q-AA-CTR-R1-aaword-b025-s42-h626cut470": ("aa_word", 0.025),
    "Q-AA-CTR-R1-aaword-b050-s42-h626cut470": ("aa_word", 0.05),
    "Q-AA-CTR-R1-tdist-b050-s42-h626cut470": (
        "teacher_distribution",
        0.05,
    ),
}
EPOCH_SCHEDULES: Final = {
    "0.75": (235, (59, 118, 177, 235)),
    "1.0": (313, (79, 157, 235, 313)),
    "1.25": (391, (98, 196, 294, 391)),
}
SCHEDULE_PROFILES: Final = {
    "legacy_epoch_grid": None,
    "macro3full_q_fixed_1p25": (391, (235, 313, 391)),
    # Exploratory long-horizon follow-up of the seed-42 Base01 word-wise
    # winner.  This is a fresh cosine trajectory, not a continuation of the
    # original 391-step scheduler.
    "macro3full_q_word_base01_2p0": (626, (391, 470, 548, 626)),
    # Round 1 stops at the observed Base01 peak while preserving the exact
    # 626-step cosine trajectory that produced that checkpoint.  Execution
    # stop and scheduler horizon are intentionally separate contracts.
    "contrast_round1_h626_cut470": (470, (313, 391, 430, 470)),
}
MODEL_AUXILIARY_FILES: Final = (
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
# ``spk_dict.pt`` belongs to speech output.  Training loads thinker-only
# students, while the full-Omni teacher's TextOnlyOmni.load_speakers is an
# intentional no-op, so that file cannot affect this text-output experiment.


def canonical_learning_rate(value: str | float) -> str:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("learning rate must be finite and positive")
    mantissa, exponent = f"{parsed:.11e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def learning_rate_path_tag(value: str | float) -> str:
    return (
        "lr"
        + canonical_learning_rate(value)
        .replace(".", "p")
        .replace("-", "m")
        .replace("+", "p")
    )


def aa_selector_for_variant(selection_unit: str) -> str:
    mapping = {
        "wordwise": "joint_top_fraction",
        "tokenwise": "token_top_fraction",
    }
    try:
        return mapping[selection_unit]
    except KeyError as error:
        raise ValueError(f"unknown AA selection unit: {selection_unit}") from error


def selector_contract_for_variant(
    selection_unit: str, top_fraction: float, high_weight: float
) -> dict[str, object]:
    common: dict[str, object] = {
        "variant": selection_unit,
        "name": aa_selector_for_variant(selection_unit),
        "top_fraction": top_fraction,
        "high_low_weights": [high_weight, 1.0],
        "stopwords": False,
        "smoothing": False,
        "per_row_mean_normalization": False,
    }
    if selection_unit == "wordwise":
        return {
            **common,
            "selection_unit": "complete lexical reasoning word",
            "piece_score": "support ? signed(logP_real-logP_wrong) : 0",
            "aggregate_score": "sum of support-masked signed piece scores",
            "candidate_pool": "positive complete lexical reasoning words",
            "denominator": "all lexical reasoning words",
            "weight_spread": "all sampled BPE pieces of a selected word",
            "tie_break": "(-word_score, word_id)",
        }
    return {
        **common,
        "selection_unit": "sampled lexical reasoning BPE position",
        "piece_score": "support ? max(0, logP_real-logP_wrong) : 0",
        "aggregate_score": "none",
        "candidate_pool": "positive lexical reasoning sampled-token positions",
        "denominator": "all lexical reasoning sampled-token positions",
        "weight_spread": "selected token position only",
        "tie_break": "(-token_score, generation_position)",
    }


@dataclass(frozen=True)
class ResolvedRun:
    run_name: str
    initialization: str
    learning_rate: float
    epoch: str
    max_steps: int
    checkpoint_steps: tuple[int, ...]
    schedule_profile: str
    scheduler_horizon: int
    seed: int
    aa_top_fraction: float
    aa_selection_unit: str
    lambda_opd: float
    parameterization: str
    lora_rank: int
    lora_alpha: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    engineering_calibration: bool
    engineering_save_load_checkpoint: bool
    performance_warmup_steps: int
    contrast_mode: str
    contrast_beta: float
    contrast_margin: float
    contrast_teacher_ratio_threshold: float
    contrast_temperature: float
    contrast_ramp_start: int
    contrast_ramp_end: int
    contrast_smoke: bool

    @property
    def effective_batch(self) -> int:
        return (
            self.per_device_train_batch_size
            * self.gradient_accumulation_steps
            * self.world_size
        )


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_checkpoint_steps(value: str, *, expected_count: int) -> tuple[int, ...]:
    try:
        steps = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("--checkpoint-steps must contain integers") from error
    if (
        len(steps) != expected_count
        or tuple(sorted(set(steps))) != steps
        or steps[0] <= 0
    ):
        raise ValueError(
            "--checkpoint-steps must contain exactly "
            f"{expected_count} unique increasing positive steps"
        )
    return steps


def resolve_schedule(
    *,
    epoch: str | None,
    max_steps: int | None,
    checkpoint_steps: str | None,
    engineering_calibration: bool = False,
    schedule_profile: str = "legacy_epoch_grid",
) -> tuple[str, int, tuple[int, ...]]:
    if schedule_profile not in SCHEDULE_PROFILES:
        raise ValueError(f"unsupported schedule profile: {schedule_profile}")
    if engineering_calibration:
        if schedule_profile != "legacy_epoch_grid":
            raise ValueError("engineering calibration cannot use a formal schedule profile")
        if epoch is not None or checkpoint_steps is not None:
            raise ValueError(
                "engineering calibration accepts --max-steps only, without epoch/checkpoints"
            )
        if max_steps is None or not 1 <= max_steps <= 50:
            raise ValueError("engineering calibration requires --max-steps in [1, 50]")
        return "engineering_calibration", max_steps, ()
    if schedule_profile in {
        "macro3full_q_fixed_1p25",
        "macro3full_q_word_base01_2p0",
        "contrast_round1_h626_cut470",
    }:
        registered_max, registered_checkpoints = SCHEDULE_PROFILES[schedule_profile]
        allowed_epochs = (
            {None, "1.25"}
            if schedule_profile == "macro3full_q_fixed_1p25"
            else {None}
        )
        if epoch not in allowed_epochs:
            raise ValueError(
                f"{schedule_profile} fixes one scheduler; do not override --epoch"
            )
        if (max_steps is None) != (checkpoint_steps is None):
            raise ValueError("--max-steps and --checkpoint-steps must be supplied together")
        if max_steps is None:
            return schedule_profile, registered_max, registered_checkpoints
        supplied = parse_checkpoint_steps(
            str(checkpoint_steps), expected_count=len(registered_checkpoints)
        )
        if max_steps != registered_max or supplied != registered_checkpoints:
            raise ValueError(
                f"{schedule_profile} requires max_steps={registered_max} and "
                "checkpoints=" + ",".join(map(str, registered_checkpoints))
            )
        return schedule_profile, registered_max, registered_checkpoints
    if (max_steps is None) != (checkpoint_steps is None):
        raise ValueError("--max-steps and --checkpoint-steps must be supplied together")
    selected_epoch = epoch or "1.0"
    if selected_epoch not in EPOCH_SCHEDULES:
        raise ValueError(f"unsupported epoch schedule: {selected_epoch}")
    registered_max, registered_checkpoints = EPOCH_SCHEDULES[selected_epoch]
    if max_steps is None:
        return selected_epoch, registered_max, registered_checkpoints

    supplied = parse_checkpoint_steps(str(checkpoint_steps), expected_count=4)
    if supplied[-1] != max_steps:
        raise ValueError("the terminal checkpoint must equal --max-steps")
    matches = [
        name
        for name, (candidate_max, candidate_steps) in EPOCH_SCHEDULES.items()
        if candidate_max == max_steps and candidate_steps == supplied
    ]
    if not matches:
        raise ValueError(
            "explicit max/checkpoint steps must match a frozen 0.75/1.0/1.25 epoch schedule"
        )
    inferred = matches[0]
    if epoch is not None and inferred != selected_epoch:
        raise ValueError("--epoch disagrees with --max-steps/--checkpoint-steps")
    return inferred, max_steps, supplied


def validate_and_resolve(args: argparse.Namespace) -> ResolvedRun:
    if not RUN_NAME_RE.fullmatch(str(args.run_name)):
        raise ValueError(
            "--run-name must be 1-128 safe characters: letters, digits, dot, underscore, dash"
        )
    learning_rate = float(args.learning_rate)
    top_fraction = float(args.aa_top_fraction)
    lambda_opd = float(args.lambda_opd)
    contrast_mode = str(getattr(args, "contrast_mode", "none"))
    contrast_beta = float(getattr(args, "contrast_beta", 0.0))
    contrast_margin = float(getattr(args, "contrast_margin", 0.10))
    contrast_teacher_ratio_threshold = float(
        getattr(args, "contrast_teacher_ratio_threshold", 2.0)
    )
    contrast_temperature = float(getattr(args, "contrast_temperature", 1.0))
    contrast_ramp_start = int(getattr(args, "contrast_ramp_start", 32))
    contrast_ramp_end = int(getattr(args, "contrast_ramp_end", 80))
    contrast_smoke = bool(getattr(args, "thor2_contrast_round1_smoke", False))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive")
    if not math.isfinite(top_fraction) or not 0 < top_fraction <= 1:
        raise ValueError("--aa-top-fraction must be finite and in (0, 1]")
    if not math.isfinite(lambda_opd) or lambda_opd <= 0:
        raise ValueError("--lambda-opd must be finite and positive")
    if contrast_mode not in {"none", "aa_word", "teacher_distribution"}:
        raise ValueError("unsupported --contrast-mode")
    if not math.isfinite(contrast_beta) or contrast_beta < 0:
        raise ValueError("--contrast-beta must be finite and non-negative")
    if contrast_mode == "none" and contrast_beta != 0:
        raise ValueError("contrast mode none requires beta=0")
    if contrast_mode != "none" and contrast_beta <= 0:
        raise ValueError("an active contrast mode requires beta>0")
    if not math.isfinite(contrast_margin) or contrast_margin < 0:
        raise ValueError("--contrast-margin must be finite and non-negative")
    if (
        not math.isfinite(contrast_teacher_ratio_threshold)
        or contrast_teacher_ratio_threshold <= 1
    ):
        raise ValueError("--contrast-teacher-ratio-threshold must be greater than 1")
    if not math.isfinite(contrast_temperature) or contrast_temperature <= 0:
        raise ValueError("--contrast-temperature must be finite and positive")
    selection_unit = str(args.aa_selection_unit)
    aa_selector_for_variant(selection_unit)
    schedule_profile = str(getattr(args, "schedule_profile", "legacy_epoch_grid"))
    seed = int(getattr(args, "seed", 42))
    if seed not in {42, 43, 44}:
        raise ValueError("formal AA tuning seed must be one of 42, 43, 44")

    epoch, max_steps, checkpoint_steps = resolve_schedule(
        epoch=args.epoch,
        max_steps=args.max_steps,
        checkpoint_steps=args.checkpoint_steps,
        engineering_calibration=bool(args.engineering_calibration),
        schedule_profile=schedule_profile,
    )
    scheduler_horizon = (
        626
        if schedule_profile == "contrast_round1_h626_cut470" or contrast_smoke
        else max_steps
    )
    if schedule_profile == "contrast_round1_h626_cut470":
        if not (0 <= contrast_ramp_start < contrast_ramp_end <= max_steps):
            raise ValueError(
                "contrast ramp must satisfy 0 <= start < end <= execution max steps"
            )
        if str(args.initialization) != "Q" or selection_unit != "wordwise":
            raise ValueError("contrast Round 1 is frozen to Q word-wise AA")
    elif contrast_smoke:
        if not (
            bool(args.engineering_calibration)
            and max_steps == 10
            and str(args.initialization) == "Q"
            and selection_unit == "wordwise"
            and contrast_mode == "aa_word"
            and math.isclose(contrast_beta, 0.05, rel_tol=0, abs_tol=1e-15)
            and contrast_ramp_start == 0
            and contrast_ramp_end == 1
        ):
            raise ValueError("contrast Round 1 smoke contract mismatch")
    elif contrast_mode != "none" or contrast_beta != 0:
        raise ValueError(
            "contrast objectives are restricted to the frozen Round 1 schedule"
        )
    world_size = int(args.world_size)
    environment_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in {1, 4, 8} or environment_world_size != world_size:
        raise ValueError(
            "--world-size must be 1, 4, or 8 and must match the torchrun WORLD_SIZE"
        )
    if args.engineering_calibration and world_size != 1 and not contrast_smoke:
        raise ValueError("engineering calibration remains single-GPU only")
    microbatch = int(args.per_device_train_batch_size)
    accumulation = int(args.gradient_accumulation_steps)
    if microbatch <= 0 or accumulation <= 0:
        raise ValueError("microbatch and gradient accumulation must be positive")
    if microbatch * accumulation * world_size != EFFECTIVE_BATCH:
        raise ValueError(
            "canonical V1.0 effective batch must remain 32: microbatch * accumulation * world_size"
        )

    if args.full_ft:
        if args.lora_rank is not None or args.lora_alpha is not None:
            raise ValueError("--full-ft cannot be combined with LoRA rank/alpha")
        parameterization = "full_ft"
        lora_rank = 0
        lora_alpha = 0
    else:
        parameterization = "lora"
        lora_rank = 16 if args.lora_rank is None else int(args.lora_rank)
        lora_alpha = 2 * lora_rank if args.lora_alpha is None else int(args.lora_alpha)
        if lora_rank <= 0 or lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if lora_alpha != 2 * lora_rank:
            raise ValueError("canonical rank sweep requires lora_alpha == 2 * lora_rank")

    save_load_checkpoint = bool(
        getattr(args, "engineering_save_load_checkpoint", False)
    )
    if save_load_checkpoint and not (
        args.engineering_calibration
        and parameterization == "full_ft"
        and max_steps == 10
    ):
        raise ValueError(
            "--engineering-save-load-checkpoint requires a 10-step Full-FT "
            "--engineering-calibration run"
        )
    performance_warmup_steps = int(
        getattr(args, "performance_warmup_steps", 0)
    )
    if performance_warmup_steps < 0 or performance_warmup_steps >= max_steps:
        raise ValueError(
            "--performance-warmup-steps must be non-negative and smaller than max steps"
        )
    if performance_warmup_steps and not args.engineering_calibration:
        raise ValueError(
            "--performance-warmup-steps is reserved for engineering calibration"
        )

    return ResolvedRun(
        run_name=str(args.run_name),
        initialization=str(args.initialization),
        learning_rate=learning_rate,
        epoch=epoch,
        max_steps=max_steps,
        checkpoint_steps=checkpoint_steps,
        schedule_profile=schedule_profile,
        scheduler_horizon=scheduler_horizon,
        seed=seed,
        aa_top_fraction=top_fraction,
        aa_selection_unit=selection_unit,
        lambda_opd=lambda_opd,
        parameterization=parameterization,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        per_device_train_batch_size=microbatch,
        gradient_accumulation_steps=accumulation,
        world_size=world_size,
        engineering_calibration=bool(args.engineering_calibration),
        engineering_save_load_checkpoint=save_load_checkpoint,
        performance_warmup_steps=performance_warmup_steps,
        contrast_mode=contrast_mode,
        contrast_beta=contrast_beta,
        contrast_margin=contrast_margin,
        contrast_teacher_ratio_threshold=contrast_teacher_ratio_threshold,
        contrast_temperature=contrast_temperature,
        contrast_ramp_start=contrast_ramp_start,
        contrast_ramp_end=contrast_ramp_end,
        contrast_smoke=contrast_smoke,
    )


def scheduled_checkpoint_steps(resolved: ResolvedRun) -> tuple[int, ...]:
    if resolved.engineering_save_load_checkpoint:
        return (resolved.max_steps,)
    return resolved.checkpoint_steps


def validate_output_scope(
    resolved: ResolvedRun, output: Path, resume_from_checkpoint: str | None
) -> None:
    if not resolved.engineering_calibration:
        return
    if "calibration" not in {part.casefold() for part in output.parts}:
        raise ValueError(
            "engineering calibration output must be inside a directory component named calibration"
        )
    if resume_from_checkpoint is not None:
        raise ValueError("engineering calibration runs are short, non-resumable probes")


def validate_thor2_four_gpu_extension(
    resolved: ResolvedRun,
    *,
    enabled: bool,
    visible_gpu_names: list[str],
) -> dict[str, object] | None:
    """Authorize only the exact user-requested Base01 long-horizon run.

    The historical Mantis topology receipts cover single-GPU A100/L40S jobs.
    Thor2's established local topology is four RTX 6000 Ada GPUs at global
    batch 32, so this explicit receipt keeps that separate execution contract
    narrow and machine-checkable instead of weakening the Mantis gates.
    """

    if not enabled:
        return None
    checks = {
        "host": os.uname().nodename.split(".", 1)[0] == "thor2",
        "run_name": resolved.run_name == BASE01_LONG_RUN_NAME,
        "initialization": resolved.initialization == "Q",
        "schedule": resolved.schedule_profile == "macro3full_q_word_base01_2p0",
        "steps": resolved.max_steps == 626
        and resolved.checkpoint_steps == (391, 470, 548, 626),
        "selector": resolved.aa_selection_unit == "wordwise",
        "parameterization": resolved.parameterization == "lora"
        and resolved.lora_rank == 64
        and resolved.lora_alpha == 128,
        "learning_rate": math.isclose(
            resolved.learning_rate, 1e-4, rel_tol=0, abs_tol=1e-15
        ),
        "top_fraction": math.isclose(
            resolved.aa_top_fraction, 0.30, rel_tol=0, abs_tol=1e-15
        ),
        "lambda_opd": math.isclose(
            resolved.lambda_opd, 0.25, rel_tol=0, abs_tol=1e-15
        ),
        "seed": resolved.seed == 42,
        "batch": resolved.per_device_train_batch_size == 1
        and resolved.gradient_accumulation_steps == 8
        and resolved.world_size == 4
        and resolved.effective_batch == EFFECTIVE_BATCH,
        "visible_gpu_count": len(visible_gpu_names) == 4,
        "gpu_family": bool(visible_gpu_names)
        and all("RTX 6000 Ada" in name for name in visible_gpu_names),
        "not_calibration": not resolved.engineering_calibration,
    }
    if not all(checks.values()):
        raise ValueError(f"thor2 four-GPU Base01 extension mismatch: {checks}")
    return {
        "schema_version": 1,
        "user_authorized": True,
        "scope": "exact Base01 word-wise 2.0-epoch horizon extension on thor2",
        "host": os.uname().nodename,
        "gpu_names": visible_gpu_names,
        "microbatch": resolved.per_device_train_batch_size,
        "gradient_accumulation_steps": resolved.gradient_accumulation_steps,
        "world_size": resolved.world_size,
        "effective_batch": resolved.effective_batch,
        "checks": checks,
    }


def validate_thor2_contrast_round1(
    resolved: ResolvedRun,
    *,
    enabled: bool,
    visible_gpu_names: list[str],
) -> dict[str, object] | None:
    """Fail closed unless this is one exact preregistered Round 1 arm."""

    if not enabled:
        return None
    expected = CONTRAST_ROUND1_RUNS.get(resolved.run_name)
    checks = {
        "host": os.uname().nodename.split(".", 1)[0] == "thor2",
        "registered_run": expected is not None,
        "contrast": expected
        == (resolved.contrast_mode, resolved.contrast_beta),
        "initialization": resolved.initialization == "Q",
        "schedule": resolved.schedule_profile == "contrast_round1_h626_cut470",
        "execution_and_scheduler": resolved.max_steps == 470
        and resolved.scheduler_horizon == 626
        and resolved.checkpoint_steps == (313, 391, 430, 470),
        "selector": resolved.aa_selection_unit == "wordwise",
        "parameterization": resolved.parameterization == "lora"
        and resolved.lora_rank == 64
        and resolved.lora_alpha == 128,
        "learning_rate": math.isclose(
            resolved.learning_rate, 1e-4, rel_tol=0, abs_tol=1e-15
        ),
        "top_fraction": math.isclose(
            resolved.aa_top_fraction, 0.30, rel_tol=0, abs_tol=1e-15
        ),
        "lambda_opd": math.isclose(
            resolved.lambda_opd, 0.25, rel_tol=0, abs_tol=1e-15
        ),
        "contrast_contract": math.isclose(
            resolved.contrast_margin,
            0.02 if resolved.contrast_mode == "teacher_distribution" else 0.10,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and math.isclose(
            resolved.contrast_teacher_ratio_threshold,
            2.0,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and math.isclose(
            resolved.contrast_temperature,
            1.0,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and resolved.contrast_ramp_start == 32
        and resolved.contrast_ramp_end == 80,
        "seed": resolved.seed == 42,
        "batch": resolved.per_device_train_batch_size == 1
        and resolved.gradient_accumulation_steps == 4
        and resolved.world_size == 8
        and resolved.effective_batch == EFFECTIVE_BATCH,
        "visible_gpu_count": len(visible_gpu_names) == 8,
        "gpu_family": bool(visible_gpu_names)
        and all("RTX 6000 Ada" in name for name in visible_gpu_names),
        "not_calibration": not resolved.engineering_calibration,
    }
    if not all(checks.values()):
        raise ValueError(f"thor2 contrast Round 1 mismatch: {checks}")
    return {
        "schema_version": 1,
        "user_authorized": True,
        "scope": "preregistered Q Base01 contrast Round 1 on all eight thor2 GPUs",
        "host": os.uname().nodename,
        "gpu_names": visible_gpu_names,
        "microbatch": resolved.per_device_train_batch_size,
        "gradient_accumulation_steps": resolved.gradient_accumulation_steps,
        "world_size": resolved.world_size,
        "effective_batch": resolved.effective_batch,
        "execution_stop": resolved.max_steps,
        "scheduler_horizon": resolved.scheduler_horizon,
        "checks": checks,
    }


def validate_thor2_contrast_round1_smoke(
    resolved: ResolvedRun,
    *,
    enabled: bool,
    visible_gpu_names: list[str],
) -> dict[str, object] | None:
    if not enabled:
        return None
    checks = {
        "host": os.uname().nodename.split(".", 1)[0] == "thor2",
        "run_name": resolved.run_name
        == "Q-AA-CTR-R1-aaword-b050-smoke10-s42-h626",
        "initialization": resolved.initialization == "Q",
        "nonformal": resolved.engineering_calibration and resolved.contrast_smoke,
        "steps": resolved.max_steps == 10
        and not resolved.checkpoint_steps
        and resolved.scheduler_horizon == 626,
        "recipe": resolved.contrast_mode == "aa_word"
        and math.isclose(resolved.contrast_beta, 0.05, rel_tol=0, abs_tol=1e-15)
        and math.isclose(resolved.contrast_margin, 0.10, rel_tol=0, abs_tol=1e-15)
        and math.isclose(
            resolved.contrast_teacher_ratio_threshold,
            2.0,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and resolved.contrast_ramp_start == 0
        and resolved.contrast_ramp_end == 1,
        "base": resolved.aa_selection_unit == "wordwise"
        and resolved.parameterization == "lora"
        and resolved.lora_rank == 64
        and resolved.lora_alpha == 128
        and math.isclose(resolved.learning_rate, 1e-4, rel_tol=0, abs_tol=1e-15)
        and math.isclose(resolved.aa_top_fraction, 0.30, rel_tol=0, abs_tol=1e-15)
        and math.isclose(resolved.lambda_opd, 0.25, rel_tol=0, abs_tol=1e-15)
        and resolved.seed == 42,
        "batch": resolved.per_device_train_batch_size == 1
        and resolved.gradient_accumulation_steps == 4
        and resolved.world_size == 8
        and resolved.effective_batch == EFFECTIVE_BATCH,
        "visible_gpu_count": len(visible_gpu_names) == 8,
        "gpu_family": bool(visible_gpu_names)
        and all("RTX 6000 Ada" in name for name in visible_gpu_names),
    }
    if not all(checks.values()):
        raise ValueError(f"thor2 contrast Round 1 smoke mismatch: {checks}")
    return {
        "schema_version": 1,
        "user_authorized": True,
        "scope": "nonformal eight-GPU 10-step AA-word contrast calibration on thor2",
        "host": os.uname().nodename,
        "gpu_names": visible_gpu_names,
        "execution_stop": resolved.max_steps,
        "scheduler_horizon": resolved.scheduler_horizon,
        "microbatch": resolved.per_device_train_batch_size,
        "gradient_accumulation_steps": resolved.gradient_accumulation_steps,
        "world_size": resolved.world_size,
        "effective_batch": resolved.effective_batch,
        "checks": checks,
    }


def validate_training_calibration_binding(
    report_path: Path,
    *,
    resolved: ResolvedRun,
    actual_gpu_name: str,
) -> dict[str, object]:
    """Fail closed when a formal run is not the exact calibrated topology."""

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    recommendation = payload.get("recommendation")
    measurement_contract = payload.get("measurement_contract")
    if not isinstance(recommendation, dict):
        raise ValueError("training calibration recommendation is missing")
    expected_parameterization = (
        "fullft" if resolved.parameterization == "full_ft" else f"r{resolved.lora_rank}"
    )
    gpu_kind = str(payload.get("gpu_kind", ""))
    results = payload.get("results")
    selected_rows = [
        row
        for row in results
        if isinstance(row, dict)
        and int(row.get("microbatch", 0)) == resolved.per_device_train_batch_size
    ] if isinstance(results, list) else []
    def joint_lr_probe_bound(probe: object) -> bool:
        if not isinstance(probe, dict):
            return False
        launch_path = Path(str(probe.get("launch_contract_json", "")))
        if (
            probe.get("experiment_axis") != "joint_lr_top_fraction"
            or probe.get("experiment_contract")
            != "joint_lr_topq_v1_0_aa_only_hparam_v2"
            or probe.get("aa_selector") != "joint_top_fraction"
            or probe.get("first_v2_step_audit", {}).get("aa_selector")
            != "joint_top_fraction"
            or not launch_path.is_file()
            or probe.get("launch_contract_sha256") != sha256_file(launch_path)
        ):
            return False
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        immutable = launch.get("immutable", {})
        selector = immutable.get("selector", {})
        return (
            immutable.get("experiment_contract")
            == "joint_lr_topq_v1_0_aa_only_hparam_v2"
            and isinstance(selector, dict)
            and selector.get("name") == "joint_top_fraction"
            and selector.get("stopwords") is False
        )

    selected_probes = (
        list(selected_rows[0].get("repetitions", []))
        if selected_rows
        and isinstance(selected_rows[0].get("repetitions"), list)
        else []
    )
    if selected_rows and isinstance(selected_rows[0].get("steady_confirmation"), dict):
        selected_probes.append(selected_rows[0]["steady_confirmation"])
    valid = {
        "schema": payload.get("schema_version") == 2,
        "complete": payload.get("status") == "complete",
        "measurement_contract": isinstance(measurement_contract, dict)
        and measurement_contract.get("experiment_axis") == "joint_lr_top_fraction"
        and measurement_contract.get("experiment_contract")
        == "joint_lr_topq_v1_0_aa_only_hparam_v2"
        and measurement_contract.get("aa_selector") == "joint_top_fraction",
        "initialization": payload.get("initialization") == resolved.initialization,
        "parameterization": payload.get("parameterization") == expected_parameterization,
        "candidate_set": payload.get("microbatches") == [1, 2, 4, 8, 16, 32],
        "global_batch": int(payload.get("global_batch", 0)) == EFFECTIVE_BATCH,
        "gpu_kind": gpu_kind in {"A100", "L40S"} and gpu_kind in actual_gpu_name,
        "report_gpu": gpu_kind in str(payload.get("actual_gpu_name", "")),
        "selected_row": len(selected_rows) == 1,
        "selected_row_complete": bool(
            selected_rows
            and selected_rows[0].get("complete") is True
            and selected_rows[0].get("stable") is True
            and selected_rows[0].get("confirmation_stable") is True
        ),
        "selected_joint_lr_probes": bool(selected_probes)
        and all(joint_lr_probe_bound(probe) for probe in selected_probes),
        "recommendation_parameterization": (
            recommendation.get("parameterization") == expected_parameterization
        ),
        "recommendation_gpu": recommendation.get("gpu_kind") == gpu_kind,
        "microbatch": int(recommendation.get("microbatch", 0))
        == resolved.per_device_train_batch_size,
        "accumulation": int(recommendation.get("gradient_accumulation_steps", 0))
        == resolved.gradient_accumulation_steps,
        "world_size": int(recommendation.get("world_size", 0)) == resolved.world_size,
        "effective_batch": int(recommendation.get("effective_batch", 0))
        == resolved.effective_batch == EFFECTIVE_BATCH,
        "steady_confirmation": (
            recommendation.get("bounded_steady_state_confirmation_complete") is True
            and recommendation.get("long_run_stability_proven") is False
        ),
    }
    if resolved.parameterization == "full_ft":
        valid["fullft_learning_rate"] = math.isclose(
            float(payload.get("learning_rate", 0)),
            resolved.learning_rate,
            rel_tol=0,
            abs_tol=1e-15,
        )
    if not all(valid.values()):
        raise ValueError(f"formal training/calibration binding mismatch: {valid}")
    return {
        "schema_version": 1,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "initialization": resolved.initialization,
        "parameterization": expected_parameterization,
        "gpu_kind": gpu_kind,
        "actual_gpu_name": actual_gpu_name,
        "microbatch": resolved.per_device_train_batch_size,
        "gradient_accumulation_steps": resolved.gradient_accumulation_steps,
        "world_size": resolved.world_size,
        "effective_batch": resolved.effective_batch,
    }


def validate_training_topology_binding(
    selection_path: Path,
    *,
    calibration_path: Path,
    calibration_binding: dict[str, object],
    resolved: ResolvedRun,
) -> dict[str, object]:
    """Bind a formal run to its frozen topology decision."""

    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    parameterization = (
        "fullft" if resolved.parameterization == "full_ft" else f"r{resolved.lora_rank}"
    )
    if payload.get("selection_scope") == "Q_only":
        if (
            resolved.initialization != "Q"
            or resolved.schedule_profile != "macro3full_q_fixed_1p25"
            or payload.get("schema_version") != 3
            or payload.get("complete") is not True
            or payload.get("parameterization") != parameterization
        ):
            raise ValueError("Q-only topology receipt identity mismatch")
        cells = payload.get("cells")
        scores = payload.get("aggregate_steps_per_second")
        selected = payload.get("selected")
        if not all(isinstance(value, dict) for value in (cells, scores, selected)):
            raise ValueError("Q-only topology receipt structure is incomplete")
        if set(cells) != {"A100", "L40S"} or set(selected) != {"Q"}:
            raise ValueError("Q-only topology receipt lacks the exact hardware/Q grid")
        derived_scores: dict[str, float] = {}
        for gpu_kind in ("A100", "L40S"):
            gpu_cells = cells[gpu_kind]
            if not isinstance(gpu_cells, dict) or set(gpu_cells) != {"Q"}:
                raise ValueError("Q-only topology cell is malformed")
            cell = gpu_cells["Q"]
            if not isinstance(cell, dict):
                raise ValueError("Q-only topology cell is not an object")
            source = Path(str(cell.get("source", "")))
            if not source.is_file() or cell.get("source_sha256") != sha256_file(source):
                raise ValueError("Q-only topology source/hash binding mismatch")
            if cell.get("status") == "infeasible_oom":
                if parameterization != "fullft":
                    raise ValueError("LoRA Q-only topology cannot be infeasible")
                continue
            speed = float(cell.get("steady_optimizer_steps_per_second", 0))
            microbatch = int(cell.get("microbatch", 0))
            if (
                not math.isfinite(speed)
                or speed <= 0
                or microbatch <= 0
                or int(cell.get("gradient_accumulation_steps", 0))
                != EFFECTIVE_BATCH // microbatch
                or int(cell.get("effective_batch", 0)) != EFFECTIVE_BATCH
                or int(cell.get("peak_allocated_bytes", 0)) <= 0
                or int(cell.get("peak_reserved_bytes", 0))
                < int(cell.get("peak_allocated_bytes", 0))
            ):
                raise ValueError("Q-only topology completed-cell evidence is invalid")
            derived_scores[gpu_kind] = speed
        if not derived_scores or set(scores) != set(derived_scores):
            raise ValueError("Q-only topology has no exact feasible hardware score set")
        if any(
            not math.isclose(
                float(scores[gpu]), speed, rel_tol=0, abs_tol=1e-12
            )
            for gpu, speed in derived_scores.items()
        ):
            raise ValueError("Q-only topology throughput score drift")
        policy_selected_gpu = max(
            (gpu for gpu in ("A100", "L40S") if gpu in derived_scores),
            key=lambda gpu: (derived_scores[gpu], gpu == "A100"),
        )
        selected_cell = selected["Q"]
        execution_gpu = str(calibration_binding["gpu_kind"])
        execution_cell = (
            cells.get(execution_gpu, {}).get("Q")
            if execution_gpu in derived_scores
            and isinstance(cells.get(execution_gpu), dict)
            else None
        )
        if (
            payload.get("selected_gpu") != policy_selected_gpu
            or selected_cell != cells[policy_selected_gpu]["Q"]
            or not isinstance(execution_cell, dict)
            or execution_cell.get("source") != str(calibration_path)
            or execution_cell.get("source_sha256") != sha256_file(calibration_path)
            or int(execution_cell.get("microbatch", 0))
            != resolved.per_device_train_batch_size
            or int(execution_cell.get("gradient_accumulation_steps", 0))
            != resolved.gradient_accumulation_steps
            or int(execution_cell.get("effective_batch", 0))
            != resolved.effective_batch
        ):
            raise ValueError("formal Q training/topology selection binding mismatch")
        if parameterization == "fullft" and (
            payload.get("learning_rate_canonical")
            != canonical_learning_rate(resolved.learning_rate)
            or payload.get("learning_rate_path_tag")
            != learning_rate_path_tag(resolved.learning_rate)
        ):
            raise ValueError("Q-only Full-FT topology LR binding mismatch")
        return {
            "schema_version": 1,
            "selection_path": str(selection_path),
            "selection_sha256": sha256_file(selection_path),
            "selection_scope": "Q_only",
            "parameterization": parameterization,
            "selected_gpu": policy_selected_gpu,
            "policy_selected_gpu": policy_selected_gpu,
            "execution_gpu": execution_gpu,
            "initialization": "Q",
            "calibration_report": str(calibration_path),
            "calibration_report_sha256": sha256_file(calibration_path),
            "microbatch": resolved.per_device_train_batch_size,
            "gradient_accumulation_steps": resolved.gradient_accumulation_steps,
            "aggregate_steps_per_second": derived_scores,
            "selected_cell_evidence": dict(selected_cell),
            "execution_cell_evidence": dict(execution_cell),
        }
    cells = payload.get("cells")
    scores = payload.get("aggregate_steps_per_second")
    selected = payload.get("selected")
    if not all(isinstance(value, dict) for value in (cells, scores, selected)):
        raise ValueError("topology selection receipt structure is incomplete")

    def validate_fullft_probe_evidence(
        probe: object, *, gpu_kind: str
    ) -> tuple[bool, bool]:
        if not isinstance(probe, dict):
            raise ValueError("Full-FT OOM probe evidence is malformed")
        complete = probe.get("complete") is True
        oom = probe.get("oom") is True
        if complete == oom:
            raise ValueError("Full-FT OOM probe has an invalid terminal state")
        log_path = Path(str(probe.get("log", "")))
        if not log_path.is_file() or probe.get("log_sha256") != sha256_file(log_path):
            raise ValueError("Full-FT OOM probe log evidence is invalid")
        if oom:
            oom_log = log_path.read_text(encoding="utf-8", errors="replace").casefold()
            if (
                set(probe) != {"complete", "oom", "log", "log_sha256"}
                or not ("out of memory" in oom_log or "cuda oom" in oom_log)
            ):
                raise ValueError("Full-FT OOM probe contains unrecognized evidence fields")
            return complete, oom
        required = {
            "complete",
            "oom",
            "steady_optimizer_steps_per_second",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "actual_gpu_name",
            "log",
            "log_sha256",
            "performance_sha256",
            "train_complete_sha256",
            "launch_contract_sha256",
            "experiment_axis",
            "experiment_contract",
            "aa_selector",
            "first_v2_step_audit",
        }
        if (
            set(probe) != required
            or gpu_kind not in str(probe.get("actual_gpu_name", ""))
            or not math.isfinite(float(probe.get("steady_optimizer_steps_per_second", 0)))
            or float(probe.get("steady_optimizer_steps_per_second", 0)) <= 0
            or int(probe.get("peak_allocated_bytes", 0)) <= 0
            or int(probe.get("peak_reserved_bytes", 0))
            < int(probe.get("peak_allocated_bytes", 0))
            or probe.get("experiment_axis") != "joint_lr_top_fraction"
            or probe.get("experiment_contract")
            != "joint_lr_topq_v1_0_aa_only_hparam_v2"
            or probe.get("aa_selector") != "joint_top_fraction"
            or not isinstance(probe.get("first_v2_step_audit"), dict)
            or probe["first_v2_step_audit"].get("aa_selector")
            != "joint_top_fraction"
        ):
            raise ValueError("Full-FT completed probe evidence is invalid")
        return complete, oom

    derived_scores: dict[str, float] = {}
    failed_cells: list[dict[str, object]] = []
    for gpu_kind in ("A100", "L40S"):
        gpu_cells = cells.get(gpu_kind)
        if not isinstance(gpu_cells, dict) or set(gpu_cells) != {"K", "Q"}:
            raise ValueError("topology selection lacks an exact GPU x initialization grid")
        speeds = []
        for initialization in ("K", "Q"):
            cell = gpu_cells[initialization]
            if not isinstance(cell, dict):
                raise ValueError("topology selection cell is malformed")
            source = Path(str(cell.get("source", "")))
            if not source.is_file() or cell.get("source_sha256") != sha256_file(source):
                raise ValueError("topology selection cell source evidence is invalid")
            if cell.get("status") == "infeasible_oom":
                if parameterization != "fullft":
                    raise ValueError("LoRA topology cannot contain an infeasible Full-FT cell")
                expected_keys = {
                    "status",
                    "gpu_kind",
                    "initialization",
                    "parameterization",
                    "source",
                    "source_sha256",
                    "microbatch",
                    "oom_stage",
                    "oom_evidence",
                }
                oom_evidence = cell.get("oom_evidence")
                if (
                    set(cell) != expected_keys
                    or cell.get("gpu_kind") != gpu_kind
                    or cell.get("initialization") != initialization
                    or cell.get("parameterization") != "fullft"
                    or int(cell.get("microbatch", 0)) != 1
                    or cell.get("oom_stage")
                    not in {"one_step_gate", "ten_step_screen"}
                    or not isinstance(oom_evidence, dict)
                    or set(oom_evidence) != {"one_step_gate", "repetitions"}
                    or not isinstance(oom_evidence.get("repetitions"), list)
                ):
                    raise ValueError("Full-FT infeasible-cell identity/evidence is invalid")
                gate_complete, gate_oom = validate_fullft_probe_evidence(
                    oom_evidence["one_step_gate"], gpu_kind=gpu_kind
                )
                repetitions = [
                    validate_fullft_probe_evidence(probe, gpu_kind=gpu_kind)
                    for probe in oom_evidence["repetitions"]
                ]
                if cell.get("oom_stage") == "one_step_gate":
                    evidence_valid = gate_oom and not repetitions
                else:
                    evidence_valid = (
                        gate_complete
                        and bool(repetitions)
                        and repetitions[-1][1]
                        and all(complete for complete, _ in repetitions[:-1])
                    )
                if not evidence_valid:
                    raise ValueError("Full-FT infeasible-cell OOM stage is not proven")
                failed_cells.append(
                    {
                        "gpu_kind": gpu_kind,
                        "initialization": initialization,
                        **cell,
                    }
                )
                continue
            speed = float(cell.get("steady_optimizer_steps_per_second", 0))
            if (
                cell.get("status") not in (None, "complete")
                or not math.isfinite(speed)
                or speed <= 0
                or int(cell.get("microbatch", 0)) <= 0
                or int(cell.get("gradient_accumulation_steps", 0))
                != EFFECTIVE_BATCH // int(cell.get("microbatch", 1))
                or int(cell.get("effective_batch", 0)) != EFFECTIVE_BATCH
                or int(cell.get("peak_allocated_bytes", 0)) <= 0
                or int(cell.get("peak_reserved_bytes", 0))
                < int(cell.get("peak_allocated_bytes", 0))
            ):
                raise ValueError("topology selection cell evidence is invalid")
            speeds.append(speed)
        if len(speeds) == 2:
            derived_scores[gpu_kind] = len(speeds) / sum(1.0 / speed for speed in speeds)
        elif speeds and parameterization != "fullft":
            raise ValueError("topology hardware is only partially calibrated")
    if not derived_scores:
        raise ValueError("topology selection has no feasible dual-initialization hardware")
    selected_gpu = max(
        (gpu for gpu in ("A100", "L40S") if gpu in derived_scores),
        key=lambda gpu: derived_scores[gpu],
    )
    selected_cell = selected.get(resolved.initialization)
    calibration_gpu = str(calibration_binding["gpu_kind"])
    if (
        payload.get("schema_version") != 2
        or payload.get("complete") is not True
        or payload.get("parameterization") != parameterization
        or payload.get("selected_gpu") != selected_gpu
        or calibration_gpu != selected_gpu
        or set(scores) != set(derived_scores)
        or any(
            not math.isclose(
                float(scores.get(gpu, 0)),
                speed,
                rel_tol=0,
                abs_tol=1e-12,
            )
            for gpu, speed in derived_scores.items()
        )
        or set(selected) != {"K", "Q"}
        or not isinstance(selected_cell, dict)
        or selected_cell != cells[selected_gpu][resolved.initialization]
        or selected_cell.get("source") != str(calibration_path)
        or selected_cell.get("source_sha256") != sha256_file(calibration_path)
        or int(selected_cell.get("microbatch", 0))
        != resolved.per_device_train_batch_size
        or int(selected_cell.get("gradient_accumulation_steps", 0))
        != resolved.gradient_accumulation_steps
        or int(selected_cell.get("effective_batch", 0)) != resolved.effective_batch
    ):
        raise ValueError("formal training/topology selection binding mismatch")
    if parameterization == "fullft":
        receipt_failures = payload.get("failed_cells")
        if (
            not isinstance(receipt_failures, list)
            or payload.get("learning_rate_canonical")
            != canonical_learning_rate(resolved.learning_rate)
            or payload.get("learning_rate_path_tag")
            != learning_rate_path_tag(resolved.learning_rate)
        ):
            raise ValueError("Full-FT topology receipt must enumerate failed cells")
        expected_failures = sorted(
            failed_cells,
            key=lambda cell: (str(cell["gpu_kind"]), str(cell["initialization"])),
        )
        actual_failures = sorted(
            receipt_failures,
            key=lambda cell: (
                str(cell.get("gpu_kind", "")) if isinstance(cell, dict) else "",
                str(cell.get("initialization", "")) if isinstance(cell, dict) else "",
            ),
        )
        if actual_failures != expected_failures:
            raise ValueError("Full-FT topology failed-cell evidence is inconsistent")
    return {
        "schema_version": 1,
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "parameterization": parameterization,
        "selected_gpu": selected_gpu,
        "initialization": resolved.initialization,
        "calibration_report": str(calibration_path),
        "calibration_report_sha256": sha256_file(calibration_path),
        "microbatch": resolved.per_device_train_batch_size,
        "gradient_accumulation_steps": resolved.gradient_accumulation_steps,
        "aggregate_steps_per_second": derived_scores,
        "selected_cell_evidence": dict(selected_cell),
        "failed_cells": failed_cells,
    }


def full_ft_parameter_group(name: str) -> str | None:
    if name.startswith("audio_tower.proj."):
        return "audio_projector"
    if name.startswith("model."):
        return "text_model"
    if name.startswith("lm_head."):
        return "lm_head"
    return None


def configure_full_ft_parameters(model: torch.nn.Module) -> dict[str, object]:
    """Freeze encoders/vision and train only projector plus the full text model."""
    groups: dict[str, dict[str, int]] = {
        "audio_projector": {"tensors": 0, "parameters": 0},
        "text_model": {"tensors": 0, "parameters": 0},
        "lm_head": {"tensors": 0, "parameters": 0},
    }
    for name, parameter in model.named_parameters():
        group = full_ft_parameter_group(name)
        parameter.requires_grad_(group is not None)
        if group is not None:
            groups[group]["tensors"] += 1
            groups[group]["parameters"] += int(parameter.numel())
    missing = [name for name, row in groups.items() if row["tensors"] == 0]
    if missing:
        raise ValueError(f"Full-FT topology is missing required trainable groups: {missing}")
    base_audit = audit_trainable_parameters(model)
    names = list(base_audit["names"])
    unexpected = [name for name in names if full_ft_parameter_group(name) is None]
    if unexpected:
        raise ValueError(f"Full-FT enabled forbidden parameters: {unexpected[:20]}")
    return {
        **base_audit,
        "policy": "audio_tower.proj + model + lm_head; audio encoder and visual frozen",
        "groups": groups,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--initialization", choices=("K", "Q"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--epoch", choices=tuple(EPOCH_SCHEDULES))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--checkpoint-steps")
    parser.add_argument(
        "--schedule-profile",
        choices=tuple(SCHEDULE_PROFILES),
        default="legacy_epoch_grid",
    )
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--aa-top-fraction", type=float, default=0.20)
    parser.add_argument(
        "--aa-selection-unit",
        choices=("wordwise", "tokenwise"),
        default="wordwise",
    )
    parser.add_argument("--lambda-opd", type=float, default=0.25)
    parser.add_argument(
        "--contrast-mode",
        choices=("none", "aa_word", "teacher_distribution"),
        default="none",
    )
    parser.add_argument("--contrast-beta", type=float, default=0.0)
    parser.add_argument("--contrast-margin", type=float, default=0.10)
    parser.add_argument(
        "--contrast-teacher-ratio-threshold", type=float, default=2.0
    )
    parser.add_argument("--contrast-temperature", type=float, default=1.0)
    parser.add_argument("--contrast-ramp-start", type=int, default=32)
    parser.add_argument("--contrast-ramp-end", type=int, default=80)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--full-ft", action="store_true")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--engineering-calibration", action="store_true")
    parser.add_argument("--engineering-save-load-checkpoint", action="store_true")
    parser.add_argument("--performance-warmup-steps", type=int, default=0)
    parser.add_argument("--training-calibration-report")
    parser.add_argument("--training-topology-selection-report")
    parser.add_argument("--user-speed-override-uncalibrated-fullft", action="store_true")
    parser.add_argument(
        "--thor2-four-gpu-extension",
        action="store_true",
        help="authorize only the exact Base01 word-wise 626-step thor2 run",
    )
    parser.add_argument(
        "--thor2-contrast-round1",
        action="store_true",
        help="authorize only one preregistered eight-GPU contrast Round 1 arm",
    )
    parser.add_argument(
        "--thor2-contrast-round1-smoke",
        action="store_true",
        help="authorize the exact nonformal eight-GPU 10-step AA-word probe",
    )
    parser.add_argument(
        "--train-jsonl",
        default=PROJECT_ROOT / "data/runtime/v1_0/assets/audiomcq_aa_10k.jsonl",
    )
    parser.add_argument(
        "--teacher-gate",
        default=PROJECT_ROOT / "data/runtime/v1_0/assets/teacher_gate.jsonl",
    )
    parser.add_argument(
        "--vocab-json",
        default=PROJECT_ROOT / "data/runtime/v1_0/assets/canonical_valid_vocab_v2.json",
    )
    parser.add_argument(
        "--asset-bundle",
        default=PROJECT_ROOT / "data/runtime/v1_0/assets/asset_bundle_contract.local.json",
    )
    parser.add_argument(
        "--contrast-pair-audit",
        default=(
            PROJECT_ROOT
            / "data/frozen/v1_0/contrast_round1_pair_audit.json"
        ),
    )
    parser.add_argument("--campaign-audio-receipt")
    parser.add_argument("--campaign-audio-receipt-sha256")
    parser.add_argument("--campaign-runtime-receipt")
    parser.add_argument("--campaign-runtime-receipt-sha256")
    parser.add_argument("--campaign-preregistration")
    parser.add_argument("--campaign-preregistration-sha256")
    parser.add_argument("--attempt-id")
    parser.add_argument("--student-model-dir")
    parser.add_argument("--teacher-model-dir", default=TEACHER_PATH)
    parser.add_argument("--expected-gate-count", type=int, default=10_000)
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


def _model_path_guard(path: Path, *, directory: bool) -> tuple[int, ...]:
    """Reject links/non-regular entries and return a stable stat signature."""

    supplied = path.absolute()
    try:
        value = supplied.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(supplied) from None
    expected_kind = stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(
        value.st_mode
    )
    if stat.S_ISLNK(value.st_mode) or not expected_kind:
        kind = "directory" if directory else "file"
        raise ValueError(f"model {kind} is missing, linked, or unsafe: {supplied}")
    if supplied.resolve(strict=True) != supplied:
        raise ValueError(f"model path traverses a symlink: {supplied}")
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _json_without_duplicate_keys(raw: bytes, *, source: Path) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in model index: {source}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except UnicodeDecodeError as error:
        raise ValueError(f"model index is not UTF-8: {source}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"model index is malformed: {source}") from error


def _read_stable_model_index(path: Path) -> tuple[object, tuple[int, ...]]:
    before = _model_path_guard(path, directory=False)
    raw = path.read_bytes()
    after = _model_path_guard(path, directory=False)
    if before != after or len(raw) != after[4]:
        raise RuntimeError(f"model index changed while reading: {path}")
    return _json_without_duplicate_keys(raw, source=path), after


def model_identity(path: Path) -> dict[str, object]:
    """Bind every local file that can change model or processor semantics."""

    directory = path.absolute()
    directory_before = _model_path_guard(directory, directory=True)
    config = directory / "config.json"
    index = directory / "model.safetensors.index.json"
    _model_path_guard(config, directory=False)
    index_payload, parsed_index_guard = _read_stable_model_index(index)
    weight_map = (
        index_payload.get("weight_map")
        if isinstance(index_payload, dict)
        else None
    )
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"model safetensor index has no weight map: {index}")

    referenced: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError(f"model index has an invalid tensor key: {index}")
        if (
            not isinstance(shard_name, str)
            or not shard_name
            or shard_name in {".", ".."}
            or "/" in shard_name
            or "\\" in shard_name
            or Path(shard_name).name != shard_name
            or not shard_name.endswith(".safetensors")
        ):
            raise ValueError(
                f"model index shard path is non-canonical or escapes its root: "
                f"{shard_name!r}"
            )
        referenced.add(shard_name)

    actual_shards: dict[str, Path] = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.name.endswith(".safetensors"):
                continue
            shard = directory / entry.name
            _model_path_guard(shard, directory=False)
            actual_shards[entry.name] = shard
    if referenced != set(actual_shards):
        raise ValueError(
            f"model index/shard set mismatch: referenced={sorted(referenced)} "
            f"actual={sorted(actual_shards)}"
        )

    relevant: dict[str, Path] = {
        "config.json": config,
        "model.safetensors.index.json": index,
        **{name: actual_shards[name] for name in sorted(actual_shards)},
    }
    for name in MODEL_AUXILIARY_FILES:
        auxiliary = directory / name
        if os.path.lexists(auxiliary):
            _model_path_guard(auxiliary, directory=False)
            relevant[name] = auxiliary

    ordered_names = sorted(relevant)
    guards_before = {
        name: _model_path_guard(relevant[name], directory=False)
        for name in ordered_names
    }
    if guards_before["model.safetensors.index.json"] != parsed_index_guard:
        raise RuntimeError(f"model index changed during identity capture: {index}")
    records: dict[str, dict[str, object]] = {}
    for name in ordered_names:
        records[name] = {
            "size": guards_before[name][4],
            "sha256": sha256_file(relevant[name]),
        }
    guards_after = {
        name: _model_path_guard(relevant[name], directory=False)
        for name in ordered_names
    }
    if guards_after != guards_before:
        changed = [
            name
            for name in ordered_names
            if guards_after[name] != guards_before[name]
        ]
        raise RuntimeError(
            f"model files changed during identity capture: {changed}"
        )
    if _model_path_guard(directory, directory=True) != directory_before:
        raise RuntimeError(
            f"model directory changed during identity capture: {directory}"
        )

    canonical = [
        {
            "name": name,
            "size": records[name]["size"],
            "sha256": records[name]["sha256"],
        }
        for name in ordered_names
    ]
    metadata_names = ("config.json", "model.safetensors.index.json")
    return {
        "path": str(directory),
        # Compatibility key used by already-written diagnostic readers.
        "metadata_files": {name: dict(records[name]) for name in metadata_names},
        "files": records,
        "count": len(records),
        "materialization_file_count": len(records),
        "materialization_sha256": sha256_json(canonical),
    }


def repository_provenance(project: Path) -> dict[str, object]:
    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments], cwd=project, text=True, capture_output=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256((status or "").encode()).hexdigest(),
    }


def code_identity() -> dict[str, str]:
    portable_root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "ke_opd_v2/contract.py",
        "ke_opd_v2/lora.py",
        "ke_opd_v2/losses.py",
        "ke_opd_v2/modeling.py",
        "ke_opd_v2/trainer.py",
        "scripts/train_ke_opd_v1_hparam.py",
        "scripts/ke_opd_contrast_round1_provenance.py",
    )
    return {
        relative: sha256_file(portable_root / relative) for relative in relative_paths
    }


def validate_asset_bundle(
    bundle_path: Path,
    *,
    train_path: Path,
    teacher_gate_path: Path,
    vocab_path: Path,
    prompt_sha256: str,
) -> dict[str, object]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("status") != "frozen_complete":
        raise ValueError("V1.0 asset bundle is not frozen_complete")
    if bundle.get("prompt_sha256") != prompt_sha256:
        raise ValueError("V1.0 asset bundle prompt hash does not match the runtime")
    expected = {
        "train": train_path,
        "teacher_gate": teacher_gate_path,
        "valid_vocab": vocab_path,
    }
    for key, path in expected.items():
        frozen = bundle.get("assets", {}).get(key, {})
        if frozen.get("path") != str(path) or frozen.get("sha256") != sha256_file(path):
            raise ValueError(f"V1.0 frozen asset mismatch: {key}")
    return bundle


def validate_contrast_pair_audit(
    path: Path,
    *,
    train_path: Path,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    hard_gates = payload.get("hard_gates")
    train = payload.get("inputs", {}).get("train", {})
    if not (
        payload.get("complete") is True
        and payload.get("pass") is True
        and isinstance(hard_gates, dict)
        and hard_gates
        and all(value is True for value in hard_gates.values())
        and train.get("path") == str(train_path)
        and train.get("sha256") == sha256_file(train_path)
    ):
        raise ValueError("contrast Round 1 pair audit is not a closed passing receipt")
    return payload


def build_immutable_manifest(
    *,
    resolved: ResolvedRun,
    contract: V2Contract,
    student_path: Path,
    teacher_path: Path,
    train_path: Path,
    teacher_gate_path: Path,
    vocab_path: Path,
    bundle_path: Path,
    contrast_pair_audit_path: Path | None,
    training_calibration_report: Path | None,
    training_calibration_binding: dict[str, object] | None = None,
    training_topology_selection_report: Path | None = None,
    training_topology_selection_binding: dict[str, object] | None = None,
    campaign_audio_binding: dict[str, object] | None = None,
    campaign_runtime_binding: dict[str, object] | None = None,
    campaign_preregistration_binding: dict[str, object] | None = None,
) -> dict[str, object]:
    experiment_contract = (
        "q_base01_contrast_round1_aaword_smoke10_h626_v1"
        if resolved.contrast_smoke
        else (
        "joint_lr_topq_v1_0_aa_only_hparam_v2"
        if resolved.engineering_calibration
        else (
            "q_macro3full_word_token_balanced_design_v1"
            if resolved.schedule_profile == "macro3full_q_fixed_1p25"
            else (
                "q_macro3full_base01_word_long_horizon_2p0_v1"
                if resolved.schedule_profile == "macro3full_q_word_base01_2p0"
                else (
                    "q_base01_contrast_round1_h626_cut470_v1"
                    if resolved.schedule_profile
                    == "contrast_round1_h626_cut470"
                    else "word_token_topq_v1_0_aa_only_hparam_v3"
                )
            )
        ))
    )
    selector = selector_contract_for_variant(
        resolved.aa_selection_unit,
        contract.aa_top_fraction,
        contract.aa_high_weight,
    )
    return {
        "experiment_contract": experiment_contract,
        "run": asdict(
            V2RunSpec(
                run_name=resolved.run_name,
                initialization=resolved.initialization,  # type: ignore[arg-type]
                arm="AA",
                learning_rate=resolved.learning_rate,
            )
        ),
        "resolved": {
            **asdict(resolved),
            "checkpoint_steps": list(resolved.checkpoint_steps),
        },
        "contract": asdict(contract),
        "selector": {
            **selector,
            "teacher_real_support_top_k": contract.teacher_support_top_k,
        },
        "prompt_sha256": contract.prompt_sha256,
        "assets": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "teacher_gate": {
                "path": str(teacher_gate_path),
                "sha256": sha256_file(teacher_gate_path),
            },
            "vocab": {"path": str(vocab_path), "sha256": sha256_file(vocab_path)},
            "bundle": {"path": str(bundle_path), "sha256": sha256_file(bundle_path)},
            "contrast_pair_audit": (
                {
                    "path": str(contrast_pair_audit_path),
                    "sha256": sha256_file(contrast_pair_audit_path),
                }
                if contrast_pair_audit_path is not None
                else None
            ),
        },
        "training_calibration": (
            {
                "path": str(training_calibration_report),
                "sha256": sha256_file(training_calibration_report),
                "binding": training_calibration_binding,
            }
            if training_calibration_report is not None
            else (
                {"path": None, "sha256": None, "binding": training_calibration_binding}
                if training_calibration_binding is not None
                else None
            )
        ),
        "training_topology_selection": (
            {
                "path": str(training_topology_selection_report),
                "sha256": sha256_file(training_topology_selection_report),
                "binding": training_topology_selection_binding,
            }
            if training_topology_selection_report is not None
            else (
                {
                    "path": None,
                    "sha256": None,
                    "binding": training_topology_selection_binding,
                }
                if training_topology_selection_binding is not None
                else None
            )
        ),
        "models": {
            "student": model_identity(student_path),
            "teacher": model_identity(teacher_path),
        },
        "campaign_provenance": {
            "current_audio_bytes": campaign_audio_binding,
            "training_runtime": campaign_runtime_binding,
            "preregistration": campaign_preregistration_binding,
        },
        "code_sha256": code_identity(),
    }


def write_or_validate_launch_contract(
    output: Path, immutable: dict[str, object]
) -> tuple[dict[str, object], str, bool]:
    launch_path = output / "launch_contract.json"
    if launch_path.is_file():
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        if launch.get("schema_version") != 1 or launch.get("immutable") != immutable:
            raise ValueError("existing launch_contract.json does not match this exact run")
        return launch, sha256_file(launch_path), True
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to use a non-empty output without launch_contract.json: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    launch = {
        "schema_version": 1,
        "immutable": immutable,
        "first_launch": {
            "time": time.time(),
            "host": os.uname().nodename,
            "argv": sys.argv,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node": os.environ.get("SLURMD_NODENAME"),
            "repository": repository_provenance(PROJECT_ROOT),
        },
    }
    atomic_json(launch_path, launch)
    return launch, sha256_file(launch_path), False


def campaign_receipt_launch_view(
    binding: dict[str, object], *, runtime: bool
) -> dict[str, object]:
    payload = binding.get("payload")
    identity = binding.get("identity")
    if not isinstance(payload, dict) or not isinstance(identity, dict):
        raise ValueError("campaign provenance binding is malformed")
    common = {
        "identity": identity,
        "contract": payload.get("contract"),
        "code_sha256": payload.get("code_sha256"),
    }
    if runtime:
        return {**common, "runtime": payload.get("runtime")}
    return {
        **common,
        "inputs": payload.get("inputs"),
        "counts": payload.get("counts"),
        "ordered_row_pair_sha256": payload.get("ordered_row_pair_sha256"),
        "files_sha256": payload.get("files_sha256"),
    }


def checkpoint_weight_files(checkpoint: Path, parameterization: str) -> list[Path]:
    if parameterization == "lora":
        return [
            checkpoint / "adapter_config.json",
            checkpoint / "adapter_model.safetensors",
        ]
    direct = checkpoint / "model.safetensors"
    if direct.is_file():
        return [checkpoint / "config.json", direct]
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Full-FT checkpoint has no safetensors model or index: {checkpoint}"
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    names = sorted(set(map(str, index.get("weight_map", {}).values())))
    if not names:
        raise ValueError(f"Full-FT checkpoint index has no weight map: {index_path}")
    return [checkpoint / "config.json", index_path, *(checkpoint / name for name in names)]


def checkpoint_required_files(
    checkpoint: Path, parameterization: str, world_size: int
) -> list[Path]:
    rng = (
        [checkpoint / "rng_state.pth"]
        if world_size == 1
        else [checkpoint / f"rng_state_{index}.pth" for index in range(world_size)]
    )
    return [
        *checkpoint_weight_files(checkpoint, parameterization),
        checkpoint / "trainer_state.json",
        checkpoint / "optimizer.pt",
        checkpoint / "scheduler.pt",
        *rng,
    ]


def gpu_performance_snapshot(
    started_at: float, step: int, starting_step: int = 0
) -> dict[str, object]:
    progressed = max(0, step - starting_step)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        device = torch.cuda.current_device()
        elapsed = time.monotonic() - started_at
        return {
            "elapsed_seconds": elapsed,
            "global_optimizer_step": step,
            "optimizer_steps_this_attempt": progressed,
            "optimizer_steps_per_second": progressed / max(elapsed, 1e-9),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    return {
        "elapsed_seconds": time.monotonic() - started_at,
        "global_optimizer_step": step,
        "optimizer_steps_this_attempt": progressed,
        "optimizer_steps_per_second": 0.0,
        "cuda_device_name": None,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
    }


def write_checkpoint_manifest(
    checkpoint: Path,
    *,
    parameterization: str,
    world_size: int,
    launch_contract_sha256: str,
    performance_snapshot: dict[str, object],
) -> dict[str, object]:
    step_match = re.fullmatch(r"checkpoint-([0-9]+)", checkpoint.name)
    if step_match is None:
        raise ValueError(f"invalid checkpoint directory: {checkpoint}")
    step = int(step_match.group(1))
    files = checkpoint_required_files(checkpoint, parameterization, world_size)
    missing = [str(path) for path in files if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"checkpoint is incomplete: {missing}")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if state.get("global_step") != step:
        raise ValueError(f"trainer_state.global_step does not match {checkpoint.name}")
    payload = {
        "schema_version": 1,
        "complete": True,
        "step": step,
        "parameterization": parameterization,
        "world_size": world_size,
        "launch_contract_sha256": launch_contract_sha256,
        "files": {
            path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(files)
        },
        "performance_snapshot": performance_snapshot,
        "finished_at": time.time(),
    }
    atomic_json(checkpoint / "checkpoint_manifest.json", payload)
    return payload


def validate_checkpoint_manifest(
    checkpoint: Path,
    *,
    parameterization: str,
    world_size: int,
    launch_contract_sha256: str,
) -> dict[str, object]:
    step_match = CHECKPOINT_NAME_RE.fullmatch(checkpoint.name)
    if step_match is None:
        raise ValueError(f"invalid checkpoint directory name: {checkpoint}")
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError(f"checkpoint must be a real directory, not a symlink: {checkpoint}")
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is missing or unsafe: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint manifest must be a JSON object: {manifest_path}")
    expected_step = int(step_match.group(1))
    if not (
        payload.get("complete") is True
        and payload.get("step") == expected_step
        and payload.get("parameterization") == parameterization
        and payload.get("world_size") == world_size
        and payload.get("launch_contract_sha256") == launch_contract_sha256
    ):
        raise ValueError(f"checkpoint manifest contract mismatch: {manifest_path}")
    expected_files = checkpoint_required_files(checkpoint, parameterization, world_size)
    records = payload.get("files")
    if not isinstance(records, dict) or set(records) != {path.name for path in expected_files}:
        raise ValueError(f"checkpoint manifest file set mismatch: {manifest_path}")
    for path in expected_files:
        record = records[path.name]
        if (
            not isinstance(record, dict)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(record["size"])
            or sha256_file(path) != str(record["sha256"])
        ):
            raise ValueError(f"checkpoint file hash mismatch: {path}")
    return payload


def write_engineering_checkpoint_closed(
    checkpoint: Path,
    *,
    run_name: str,
    parameterization: str,
    launch_contract_sha256: str,
    checkpoint_manifest: dict[str, object],
) -> dict[str, object]:
    step = int(checkpoint.name.split("-")[-1])
    if not (
        checkpoint_manifest.get("complete") is True
        and checkpoint_manifest.get("step") == step
        and checkpoint_manifest.get("parameterization") == parameterization
        and checkpoint_manifest.get("launch_contract_sha256")
        == launch_contract_sha256
    ):
        raise ValueError("cannot close a checkpoint with a mismatched manifest")
    manifest_path = checkpoint / "checkpoint_manifest.json"
    payload = {
        "schema_version": 1,
        "complete": True,
        "nonformal": True,
        "eligible_for_formal_schedule": False,
        "purpose": "engineering_save_load_preflight",
        "run_name": run_name,
        "step": step,
        "parameterization": parameterization,
        "directory": str(checkpoint),
        "launch_contract_sha256": launch_contract_sha256,
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "files": checkpoint_manifest.get("files"),
        "closed_at": time.time(),
    }
    atomic_json(checkpoint / "CHECKPOINT_CLOSED.json", payload)
    return payload


def validate_engineering_checkpoint_closed(
    checkpoint: Path,
    *,
    run_name: str,
    parameterization: str,
    launch_contract_sha256: str,
) -> dict[str, object]:
    path = checkpoint / "CHECKPOINT_CLOSED.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    step = int(checkpoint.name.split("-")[-1])
    if not (
        payload.get("schema_version") == 1
        and payload.get("complete") is True
        and payload.get("nonformal") is True
        and payload.get("eligible_for_formal_schedule") is False
        and payload.get("purpose") == "engineering_save_load_preflight"
        and payload.get("run_name") == run_name
        and payload.get("step") == step
        and payload.get("parameterization") == parameterization
        and payload.get("launch_contract_sha256") == launch_contract_sha256
        and payload.get("checkpoint_manifest_sha256")
        == sha256_file(manifest_path)
        and payload.get("files") == manifest.get("files")
    ):
        raise ValueError(f"engineering checkpoint closure mismatch: {path}")
    return payload


def verify_engineering_checkpoint_save_load(
    checkpoint: Path,
    *,
    receipt_path: Path,
    run_name: str,
    parameterization: str,
    world_size: int,
    launch_contract_sha256: str,
    expected_parameter_count: int,
    expected_parameter_tensors: int,
    model_loader: Callable[..., torch.nn.Module] = load_thinker,
) -> tuple[dict[str, object], dict[str, object]]:
    started_at = time.time()
    started_monotonic = time.monotonic()
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "complete": False,
        "nonformal": True,
        "eligible_for_formal_schedule": False,
        "purpose": "engineering_save_load_preflight",
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "parameterization": parameterization,
        "launch_contract_sha256": launch_contract_sha256,
        "started_at": started_at,
    }
    reloaded: torch.nn.Module | None = None
    try:
        if parameterization != "full_ft":
            raise ValueError("save/load preflight accepts Full-FT checkpoints only")
        manifest = validate_checkpoint_manifest(
            checkpoint,
            parameterization=parameterization,
            world_size=world_size,
            launch_contract_sha256=launch_contract_sha256,
        )
        closed = validate_engineering_checkpoint_closed(
            checkpoint,
            run_name=run_name,
            parameterization=parameterization,
            launch_contract_sha256=launch_contract_sha256,
        )
        load_started = time.monotonic()
        reloaded = model_loader(checkpoint, dtype=torch.bfloat16)
        load_seconds = time.monotonic() - load_started
        parameters = list(reloaded.parameters())
        parameter_count = sum(int(parameter.numel()) for parameter in parameters)
        parameter_tensors = len(parameters)
        devices = sorted({str(parameter.device) for parameter in parameters})
        dtypes = sorted({str(parameter.dtype) for parameter in parameters})
        if parameter_count != expected_parameter_count:
            raise ValueError(
                "reloaded parameter count mismatch: "
                f"expected={expected_parameter_count} actual={parameter_count}"
            )
        if parameter_tensors != expected_parameter_tensors:
            raise ValueError(
                "reloaded parameter tensor count mismatch: "
                f"expected={expected_parameter_tensors} actual={parameter_tensors}"
            )
        if devices != ["cpu"]:
            raise ValueError(f"checkpoint reload was not CPU-only: {devices}")
        receipt.update(
            {
                "status": "complete",
                "complete": True,
                "checkpoint_manifest_sha256": sha256_file(
                    checkpoint / "checkpoint_manifest.json"
                ),
                "checkpoint_closed_sha256": sha256_file(
                    checkpoint / "CHECKPOINT_CLOSED.json"
                ),
                "checkpoint_step": manifest["step"],
                "closed_contract": closed["purpose"],
                "reloaded_parameter_count": parameter_count,
                "reloaded_parameter_tensors": parameter_tensors,
                "reloaded_devices": devices,
                "reloaded_dtypes": dtypes,
                "reload_seconds": load_seconds,
                "elapsed_seconds": time.monotonic() - started_monotonic,
                "finished_at": time.time(),
            }
        )
        atomic_json(receipt_path, receipt)
        return receipt, manifest
    except Exception as error:
        receipt.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.monotonic() - started_monotonic,
                "finished_at": time.time(),
            }
        )
        atomic_json(receipt_path, receipt)
        raise
    finally:
        if reloaded is not None:
            del reloaded
        gc.collect()


def _exact_checkpoint_directories(output: Path) -> list[Path]:
    """Return real direct-child checkpoint directories and reject lookalikes.

    Non-matching files and directories are unrelated output artifacts and are
    deliberately ignored.  An exact ``checkpoint-N`` name, however, is part of
    the resume namespace: if it is a symlink or not a directory, continuing
    would be ambiguous and therefore fails closed without touching the entry.
    """
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"checkpoint output must be a real directory: {output}")
    checkpoints: list[Path] = []
    unsafe: list[str] = []
    for path in output.iterdir():
        match = CHECKPOINT_NAME_RE.fullmatch(path.name)
        if match is None:
            continue
        entry_stat = path.lstat()
        if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISDIR(entry_stat.st_mode):
            unsafe.append(str(path))
        else:
            checkpoints.append(path)
    if unsafe:
        raise ValueError(
            "unsafe non-directory or symlink occupies checkpoint namespace: "
            + ", ".join(sorted(unsafe))
        )
    return sorted(checkpoints, key=lambda path: int(path.name.split("-")[-1]))


def resolve_resume_checkpoint(
    output: Path,
    *,
    explicit: str | None,
    parameterization: str,
    world_size: int,
    launch_contract_sha256: str,
) -> tuple[Path | None, dict[str, object] | None]:
    checkpoints = _exact_checkpoint_directories(output)
    manifests: dict[Path, dict[str, object]] = {}
    for checkpoint in checkpoints:
        manifests[checkpoint] = validate_checkpoint_manifest(
            checkpoint,
            parameterization=parameterization,
            world_size=world_size,
            launch_contract_sha256=launch_contract_sha256,
        )
    if explicit is None:
        selected = checkpoints[-1] if checkpoints else None
    else:
        selected = Path(explicit).resolve()
        if not checkpoints or selected != checkpoints[-1].resolve():
            raise ValueError("explicit resume checkpoint must be the latest checkpoint in output")
    return selected, manifests.get(selected) if selected is not None else None


def checkpoint_recovery_session_id(
    *, launch_contract_sha256: str, world_size: int
) -> str:
    """Derive one shared, restart-specific ID for all local torchrun ranks."""
    elastic_run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    if world_size > 1 and not elastic_run_id:
        raise RuntimeError(
            "multi-rank checkpoint recovery requires TORCHELASTIC_RUN_ID"
        )
    if elastic_run_id is None:
        elastic_run_id = f"single-process-{os.getpid()}-{time.time_ns()}"
    return sha256_json(
        {
            "purpose": "checkpoint_save_recovery",
            "launch_contract_sha256": launch_contract_sha256,
            "world_size": world_size,
            "torchelastic_run_id": elastic_run_id,
            "torchelastic_restart_count": os.environ.get(
                "TORCHELASTIC_RESTART_COUNT", "0"
            ),
            "master_addr": os.environ.get("MASTER_ADDR"),
            "master_port": os.environ.get("MASTER_PORT"),
        }
    )


def _checkpoint_recovery_receipt_path(output: Path, recovery_session_id: str) -> Path:
    if RECOVERY_SESSION_RE.fullmatch(recovery_session_id) is None:
        raise ValueError("checkpoint recovery session ID must be 64 lowercase hex digits")
    return output / f"CHECKPOINT_RECOVERY_{recovery_session_id}.json"


def _ensure_checkpoint_quarantine_root(
    output: Path, *, launch_contract_sha256: str
) -> Path:
    quarantine = output / ".checkpoint-save-quarantine"
    marker = quarantine / "QUARANTINE_ROOT.json"
    expected = {
        "schema_version": 1,
        "purpose": "interrupted_checkpoint_save_quarantine",
        "output": str(output.resolve()),
        "launch_contract_sha256": launch_contract_sha256,
    }
    if os.path.lexists(quarantine):
        entry_stat = quarantine.lstat()
        if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISDIR(entry_stat.st_mode):
            raise ValueError(f"unsafe checkpoint quarantine path: {quarantine}")
        if marker.is_symlink() or not marker.is_file():
            raise ValueError(f"unowned checkpoint quarantine directory: {quarantine}")
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"checkpoint quarantine contract mismatch: {marker}")
    else:
        quarantine.mkdir(mode=0o700)
        atomic_json(marker, expected)
    return quarantine


def _recovery_receipt_contract_valid(
    payload: object,
    *,
    output: Path,
    recovery_session_id: str,
    parameterization: str,
    world_size: int,
    launch_contract_sha256: str,
) -> bool:
    return bool(
        isinstance(payload, dict)
        and payload.get("schema_version") == 1
        and payload.get("purpose") == "interrupted_checkpoint_save_recovery"
        and payload.get("output") == str(output.resolve())
        and payload.get("recovery_session_id") == recovery_session_id
        and payload.get("parameterization") == parameterization
        and payload.get("world_size") == world_size
        and payload.get("launch_contract_sha256") == launch_contract_sha256
    )


def recover_interrupted_checkpoints_and_resolve(
    output: Path,
    *,
    explicit: str | None,
    parameterization: str,
    world_size: int,
    launch_contract_sha256: str,
    global_rank: int,
    recovery_session_id: str,
    wait_timeout_seconds: float = 600.0,
) -> tuple[Path | None, dict[str, object] | None, dict[str, object]]:
    """Quarantine interrupted saves on rank zero, then resume the last valid one.

    The caller/controller guarantees that no training writer is live.  Rank
    zero alone validates and atomically renames invalid exact ``checkpoint-N``
    directories into a contract-owned quarantine.  Other ranks wait for the
    same atomic receipt and independently revalidate the surviving checkpoint
    set before accepting the selected resume point.
    """
    output = output.resolve()
    receipt_path = _checkpoint_recovery_receipt_path(output, recovery_session_id)
    if receipt_path.is_symlink():
        raise ValueError(f"checkpoint recovery receipt must not be a symlink: {receipt_path}")
    if not math.isfinite(wait_timeout_seconds) or wait_timeout_seconds <= 0:
        raise ValueError("checkpoint recovery wait timeout must be finite and positive")
    if not 0 <= global_rank < world_size:
        raise ValueError("global rank is outside the declared world size")

    if global_rank == 0:
        lock_path = output / ".checkpoint-recovery.lock"
        if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
            raise ValueError(f"unsafe checkpoint recovery lock path: {lock_path}")
        lock_fd = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(lock_fd, "r+") as recovery_lock:
            try:
                fcntl.flock(recovery_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another checkpoint recovery is active") from error
            if os.path.lexists(receipt_path):
                raise FileExistsError(
                    f"checkpoint recovery session receipt already exists: {receipt_path}"
                )
            receipt: dict[str, object] = {
                "schema_version": 1,
                "purpose": "interrupted_checkpoint_save_recovery",
                "status": "running",
                "complete": False,
                "output": str(output),
                "recovery_session_id": recovery_session_id,
                "parameterization": parameterization,
                "world_size": world_size,
                "launch_contract_sha256": launch_contract_sha256,
                "rank_zero_pid": os.getpid(),
                "started_at": time.time(),
                "valid_checkpoints": [],
                "planned_quarantines": [],
                "quarantined": [],
            }
            atomic_json(receipt_path, receipt)
            try:
                checkpoints = _exact_checkpoint_directories(output)
                valid: list[dict[str, object]] = []
                invalid: list[dict[str, object]] = []
                source_identity: dict[Path, tuple[int, int]] = {}
                for checkpoint in checkpoints:
                    entry_stat = checkpoint.lstat()
                    source_identity[checkpoint] = (entry_stat.st_dev, entry_stat.st_ino)
                    symlink_children = sorted(
                        str(path) for path in checkpoint.iterdir() if path.is_symlink()
                    )
                    if symlink_children:
                        raise ValueError(
                            "checkpoint contains a symlink and will not be touched: "
                            + ", ".join(symlink_children)
                        )
                    try:
                        manifest = validate_checkpoint_manifest(
                            checkpoint,
                            parameterization=parameterization,
                            world_size=world_size,
                            launch_contract_sha256=launch_contract_sha256,
                        )
                    except (
                        OSError,
                        ValueError,
                        TypeError,
                        KeyError,
                        AttributeError,
                        OverflowError,
                    ) as error:
                        invalid.append(
                            {
                                "name": checkpoint.name,
                                "step": int(checkpoint.name.split("-")[-1]),
                                "validation_error": f"{type(error).__name__}: {error}",
                            }
                        )
                    else:
                        valid.append(
                            {
                                "name": checkpoint.name,
                                "step": int(manifest["step"]),
                                "checkpoint_manifest_sha256": sha256_file(
                                    checkpoint / "checkpoint_manifest.json"
                                ),
                            }
                        )

                quarantine = None
                plans: list[dict[str, object]] = []
                if invalid:
                    quarantine = _ensure_checkpoint_quarantine_root(
                        output,
                        launch_contract_sha256=launch_contract_sha256,
                    )
                    for record in invalid:
                        source = output / str(record["name"])
                        destination = quarantine / (
                            f"{source.name}.invalid-{recovery_session_id[:16]}"
                        )
                        if os.path.lexists(destination):
                            raise FileExistsError(
                                f"checkpoint quarantine destination exists: {destination}"
                            )
                        plans.append(
                            {
                                **record,
                                "source": str(source),
                                "destination": str(destination),
                            }
                        )
                receipt["valid_checkpoints"] = valid
                receipt["planned_quarantines"] = plans
                atomic_json(receipt_path, receipt)

                quarantined: list[dict[str, object]] = []
                for plan in plans:
                    source = Path(str(plan["source"]))
                    destination = Path(str(plan["destination"]))
                    current = source.lstat()
                    if (
                        stat.S_ISLNK(current.st_mode)
                        or not stat.S_ISDIR(current.st_mode)
                        or (current.st_dev, current.st_ino) != source_identity[source]
                    ):
                        raise RuntimeError(
                            f"checkpoint changed during recovery validation: {source}"
                        )
                    os.rename(source, destination)
                    quarantined.append(
                        {
                            **plan,
                            "renamed_atomically": True,
                            "quarantined_at": time.time(),
                        }
                    )
                    receipt["quarantined"] = quarantined
                    atomic_json(receipt_path, receipt)

                selected, manifest = resolve_resume_checkpoint(
                    output,
                    explicit=explicit,
                    parameterization=parameterization,
                    world_size=world_size,
                    launch_contract_sha256=launch_contract_sha256,
                )
                receipt.update(
                    {
                        "status": "complete",
                        "complete": True,
                        "resume_checkpoint": (
                            str(selected.resolve()) if selected is not None else None
                        ),
                        "resume_checkpoint_manifest_sha256": (
                            sha256_file(selected / "checkpoint_manifest.json")
                            if selected is not None
                            else None
                        ),
                        "finished_at": time.time(),
                    }
                )
                atomic_json(receipt_path, receipt)
                return selected, manifest, receipt
            except Exception as error:
                receipt.update(
                    {
                        "status": "failed",
                        "complete": False,
                        "error": f"{type(error).__name__}: {error}",
                        "finished_at": time.time(),
                    }
                )
                atomic_json(receipt_path, receipt)
                raise

    deadline = time.monotonic() + wait_timeout_seconds
    while True:
        if receipt_path.is_symlink():
            raise ValueError(
                f"checkpoint recovery receipt became a symlink: {receipt_path}"
            )
        if receipt_path.is_file():
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not _recovery_receipt_contract_valid(
                payload,
                output=output,
                recovery_session_id=recovery_session_id,
                parameterization=parameterization,
                world_size=world_size,
                launch_contract_sha256=launch_contract_sha256,
            ):
                raise ValueError(f"checkpoint recovery receipt contract mismatch: {receipt_path}")
            if payload.get("status") == "failed":
                raise RuntimeError(
                    "rank-zero checkpoint recovery failed: " + str(payload.get("error"))
                )
            if payload.get("status") == "complete" and payload.get("complete") is True:
                selected, manifest = resolve_resume_checkpoint(
                    output,
                    explicit=explicit,
                    parameterization=parameterization,
                    world_size=world_size,
                    launch_contract_sha256=launch_contract_sha256,
                )
                selected_path = str(selected.resolve()) if selected is not None else None
                selected_hash = (
                    sha256_file(selected / "checkpoint_manifest.json")
                    if selected is not None
                    else None
                )
                if (
                    payload.get("resume_checkpoint") != selected_path
                    or payload.get("resume_checkpoint_manifest_sha256") != selected_hash
                ):
                    raise ValueError(
                        "checkpoint set changed after rank-zero recovery receipt"
                    )
                return selected, manifest, payload
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for rank-zero checkpoint recovery: {receipt_path}"
            )
        time.sleep(0.05)


def load_teacher_gate(path: Path, expected_count: int) -> dict[str, bool]:
    gate: dict[str, bool] = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["id"])
        if sample_id in gate:
            raise ValueError(f"duplicate teacher-gate ID: {sample_id}")
        gate[sample_id] = bool(row["correct"] and row["parseable"])
    if len(gate) != expected_count:
        raise ValueError(
            f"teacher gate contains {len(gate)} rows, expected {expected_count}"
        )
    return gate


def complete_payload_valid(
    path: Path,
    *,
    resolved: ResolvedRun,
    launch_contract_sha256: str,
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return (
        payload.get("run_name") == resolved.run_name
        and payload.get("global_step") == resolved.max_steps
        and payload.get("launch_contract_sha256") == launch_contract_sha256
        and payload.get("complete") is True
    )


def write_or_validate_exact_json(path: Path, payload: dict[str, object]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"existing immutable receipt does not match: {path}")
        return
    atomic_json(path, payload)


def main() -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import TrainerCallback, TrainingArguments, set_seed

    args = parse_args()
    resolved = validate_and_resolve(args)
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_primary = global_rank == 0
    audit_attempt_id = str(args.attempt_id or f"standalone-{os.getpid()}")
    if not RUN_NAME_RE.fullmatch(audit_attempt_id):
        raise ValueError("--attempt-id must be a safe non-empty identifier")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != resolved.world_size
        or not 0 <= local_rank < resolved.world_size
    ):
        raise RuntimeError(
            "visible CUDA devices and LOCAL_RANK must exactly match --world-size"
        )
    torch.cuda.set_device(local_rank)
    visible_gpu_names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    local_authorizations = sum(
        bool(getattr(args, name, False))
        for name in (
            "thor2_four_gpu_extension",
            "thor2_contrast_round1",
            "thor2_contrast_round1_smoke",
        )
    )
    if local_authorizations > 1:
        raise ValueError("choose exactly one explicit thor2 topology authorization")
    thor2_extension_binding = validate_thor2_four_gpu_extension(
        resolved,
        enabled=bool(getattr(args, "thor2_four_gpu_extension", False)),
        visible_gpu_names=visible_gpu_names,
    )
    contrast_round1_binding = validate_thor2_contrast_round1(
        resolved,
        enabled=bool(getattr(args, "thor2_contrast_round1", False)),
        visible_gpu_names=visible_gpu_names,
    )
    contrast_smoke_binding = validate_thor2_contrast_round1_smoke(
        resolved,
        enabled=bool(getattr(args, "thor2_contrast_round1_smoke", False)),
        visible_gpu_names=visible_gpu_names,
    )
    local_thor2_binding = (
        thor2_extension_binding or contrast_round1_binding or contrast_smoke_binding
    )
    output = Path(args.output_dir).resolve()
    validate_output_scope(resolved, output, args.resume_from_checkpoint)
    campaign_audio_binding: dict[str, object] | None = None
    campaign_runtime_binding: dict[str, object] | None = None
    campaign_preregistration_binding: dict[str, object] | None = None
    supplied_campaign_provenance = (
        args.campaign_audio_receipt,
        args.campaign_audio_receipt_sha256,
        args.campaign_runtime_receipt,
        args.campaign_runtime_receipt_sha256,
        args.campaign_preregistration,
        args.campaign_preregistration_sha256,
    )
    if contrast_round1_binding is not None:
        if not args.attempt_id:
            raise ValueError("formal contrast Round 1 requires --attempt-id")
        if not all(supplied_campaign_provenance):
            raise ValueError(
                "formal contrast Round 1 requires bound audio/runtime receipts"
            )
        run_root = output.parents[1]
        audio_receipt_path = Path(args.campaign_audio_receipt).absolute()
        runtime_receipt_path = Path(args.campaign_runtime_receipt).absolute()
        expected_audio_path = run_root / "controller/CURRENT_AUDIO_BYTES.json"
        expected_runtime_path = run_root / "controller/TRAINING_RUNTIME.json"
        preregistration_path = Path(args.campaign_preregistration).absolute()
        expected_preregistration_path = (
            PROJECT_ROOT
            / "docs/Ke_Omni_R_AA_V1_0_CONTRASTIVE_MULTIROUND_LEDGER_20260729.md"
        )
        if (
            audio_receipt_path != expected_audio_path
            or runtime_receipt_path != expected_runtime_path
            or preregistration_path != expected_preregistration_path
        ):
            raise ValueError("campaign provenance receipts are outside the run root")
        full_audio_binding = load_bound_receipt(
            audio_receipt_path,
            expected_sha256=str(args.campaign_audio_receipt_sha256),
            expected_contract=AUDIO_RECEIPT_CONTRACT,
        )
        full_runtime_binding = load_bound_receipt(
            runtime_receipt_path,
            expected_sha256=str(args.campaign_runtime_receipt_sha256),
            expected_contract=RUNTIME_RECEIPT_CONTRACT,
        )
        expected_provenance_code = provenance_code_identity(PROJECT_ROOT)
        for label, binding in (
            ("audio", full_audio_binding),
            ("runtime", full_runtime_binding),
        ):
            payload = binding.get("payload")
            if (
                not isinstance(payload, dict)
                or payload.get("code_sha256") != expected_provenance_code
            ):
                raise ValueError(f"campaign {label} receipt code identity is stale")
        validate_local_rank_runtime_binding(
            full_runtime_binding,
            local_rank=local_rank,
            world_size=resolved.world_size,
        )
        campaign_audio_binding = campaign_receipt_launch_view(
            full_audio_binding, runtime=False
        )
        campaign_runtime_binding = campaign_receipt_launch_view(
            full_runtime_binding, runtime=True
        )
        campaign_preregistration_binding = stable_file_identity(
            preregistration_path
        )
        if (
            campaign_preregistration_binding.get("sha256")
            != str(args.campaign_preregistration_sha256)
        ):
            raise ValueError("campaign preregistration identity is stale")
    elif any(supplied_campaign_provenance):
        raise ValueError(
            "campaign audio/runtime receipts are restricted to formal contrast Round 1"
        )
    train_path = Path(args.train_jsonl).resolve()
    teacher_gate_path = Path(args.teacher_gate).resolve()
    vocab_path = Path(args.vocab_json).resolve()
    bundle_path = Path(args.asset_bundle).resolve()
    contrast_pair_audit_path = (
        Path(args.contrast_pair_audit).resolve()
        if resolved.schedule_profile == "contrast_round1_h626_cut470"
        or resolved.contrast_smoke
        else None
    )
    training_calibration_report = (
        Path(args.training_calibration_report).resolve()
        if args.training_calibration_report
        else None
    )
    training_topology_selection_report = (
        Path(args.training_topology_selection_report).resolve()
        if args.training_topology_selection_report
        else None
    )
    if local_thor2_binding is not None and (
        (resolved.engineering_calibration and contrast_smoke_binding is None)
        or training_calibration_report is not None
        or training_topology_selection_report is not None
        or bool(args.user_speed_override_uncalibrated_fullft)
    ):
        raise ValueError(
            "explicit thor2 topology cannot consume Mantis calibration/override flags"
        )
    if resolved.engineering_calibration:
        if (
            training_calibration_report is not None
            or training_topology_selection_report is not None
        ):
            raise ValueError("engineering calibration cannot consume calibration selection reports")
    elif local_thor2_binding is not None:
        pass
    elif training_calibration_report is None or not training_calibration_report.is_file():
        raise ValueError("formal training requires a closed topology-specific calibration report")
    elif (
        training_topology_selection_report is None
        or not training_topology_selection_report.is_file()
    ):
        raise ValueError("formal training requires a frozen dual-hardware topology receipt")
    direct_fullft = bool(args.user_speed_override_uncalibrated_fullft)
    if local_thor2_binding is not None:
        training_calibration_binding = {
            **local_thor2_binding,
            "binding_role": "local_thor2_execution",
        }
        training_topology_selection_binding = {
            **local_thor2_binding,
            "binding_role": "user_requested_thor2_topology",
            "execution_gpu": "RTX 6000 Ada Generation",
        }
    elif direct_fullft:
        if resolved.parameterization != "full_ft" or resolved.per_device_train_batch_size != 1:
            raise ValueError("direct Full-FT speed override requires Full-FT microbatch 1")
        actual_gpu_name = torch.cuda.get_device_name(local_rank)
        if "A100" not in actual_gpu_name:
            raise ValueError("direct Full-FT speed override is restricted to A100")
        training_calibration_binding = {
            "schema_version": 1,
            "user_speed_override": True,
            "calibrated": False,
            "gpu_kind": "A100",
            "actual_gpu_name": actual_gpu_name,
            "microbatch": 1,
            "gradient_accumulation_steps": resolved.gradient_accumulation_steps,
            "world_size": resolved.world_size,
            "effective_batch": resolved.effective_batch,
        }
        training_topology_selection_binding = {
            **training_calibration_binding,
            "execution_gpu": "A100",
            "reason": "user_requested_direct_48_without_preflight",
        }
    else:
        training_calibration_binding = (
            None
            if training_calibration_report is None
            else validate_training_calibration_binding(
                training_calibration_report,
                resolved=resolved,
                actual_gpu_name=torch.cuda.get_device_name(local_rank),
            )
        )
        training_topology_selection_binding = (
            None
            if training_topology_selection_report is None
            else validate_training_topology_binding(
                training_topology_selection_report,
                calibration_path=training_calibration_report,
                calibration_binding=training_calibration_binding,
                resolved=resolved,
            )
        )
    student_path = (
        Path(args.student_model_dir).absolute()
        if args.student_model_dir
        else MODEL_PATHS[resolved.initialization].absolute()
    )
    teacher_path = Path(args.teacher_model_dir).absolute()
    contract = V2Contract(
        version=(
            "ke_opd_v1_q_contrast_round1_smoke10_h626_v1_2026-07-29"
            if resolved.contrast_smoke
            else (
            "ke_opd_v1_joint_lr_topq_hparam_v2_2026-07-23"
            if resolved.engineering_calibration
            else (
                "ke_opd_v1_q_macro3full_balanced_design_v1_2026-07-23"
                if resolved.schedule_profile == "macro3full_q_fixed_1p25"
                else (
                    "ke_opd_v1_q_base01_word_long_horizon_2p0_v1_2026-07-29"
                    if resolved.schedule_profile == "macro3full_q_word_base01_2p0"
                    else (
                        "ke_opd_v1_q_contrast_round1_h626_cut470_v1_2026-07-29"
                        if resolved.schedule_profile
                        == "contrast_round1_h626_cut470"
                        else "ke_opd_v1_word_token_topq_hparam_v3_2026-07-23"
                    )
                )
            ))
        ),
        seed=resolved.seed,
        lambda_opd=resolved.lambda_opd,
        aa_top_fraction=resolved.aa_top_fraction,
        lora_rank=resolved.lora_rank,
        lora_alpha=resolved.lora_alpha,
        per_device_train_batch_size=resolved.per_device_train_batch_size,
        gradient_accumulation_steps=resolved.gradient_accumulation_steps,
        world_size=resolved.world_size,
        optimizer_steps=resolved.max_steps,
    )
    validate_asset_bundle(
        bundle_path,
        train_path=train_path,
        teacher_gate_path=teacher_gate_path,
        vocab_path=vocab_path,
        prompt_sha256=contract.prompt_sha256,
    )
    if contrast_pair_audit_path is not None:
        validate_contrast_pair_audit(
            contrast_pair_audit_path,
            train_path=train_path,
        )
    launch_path = output / "launch_contract.json"
    if is_primary:
        immutable = build_immutable_manifest(
            resolved=resolved,
            contract=contract,
            student_path=student_path,
            teacher_path=teacher_path,
            train_path=train_path,
            teacher_gate_path=teacher_gate_path,
            vocab_path=vocab_path,
            bundle_path=bundle_path,
            contrast_pair_audit_path=contrast_pair_audit_path,
            training_calibration_report=training_calibration_report,
            training_calibration_binding=training_calibration_binding,
            training_topology_selection_report=training_topology_selection_report,
            training_topology_selection_binding=training_topology_selection_binding,
            campaign_audio_binding=campaign_audio_binding,
            campaign_runtime_binding=campaign_runtime_binding,
            campaign_preregistration_binding=(
                campaign_preregistration_binding
            ),
        )
        _, launch_hash, resumed_launch = write_or_validate_launch_contract(
            output, immutable
        )
    else:
        deadline = time.monotonic() + 600
        while not launch_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("rank zero did not publish launch_contract.json")
            time.sleep(0.25)
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        expected_resolved = {
            **asdict(resolved),
            "checkpoint_steps": list(resolved.checkpoint_steps),
        }
        if (
            launch.get("schema_version") != 1
            or launch.get("immutable", {}).get("resolved") != expected_resolved
        ):
            raise ValueError("rank-zero launch contract does not match this worker")
        launch_hash = sha256_file(launch_path)
        resumed_launch = True
    complete_path = output / "TRAIN_COMPLETE.json"
    if complete_payload_valid(
        complete_path,
        resolved=resolved,
        launch_contract_sha256=launch_hash,
    ):
        if is_primary:
            print(
                "V2_ATTEMPT_AUDIT="
                + json.dumps(
                    {
                        "run_name": resolved.run_name,
                        "attempt_id": audit_attempt_id,
                        "status": "already_complete",
                        "starting_step": resolved.max_steps,
                        "resume_checkpoint": str(
                            output / f"checkpoint-{resolved.max_steps}"
                        ),
                        "resume_checkpoint_manifest_sha256": sha256_file(
                            output
                            / f"checkpoint-{resolved.max_steps}"
                            / "checkpoint_manifest.json"
                        ),
                        "gradient_accumulation_steps": (
                            resolved.gradient_accumulation_steps
                        ),
                    }
                ),
                flush=True,
            )
        return
    if complete_path.exists():
        raise ValueError("existing TRAIN_COMPLETE.json does not match the exact launch")

    if resolved.engineering_calibration:
        checkpoint_dirs = [path for path in output.glob("checkpoint-*") if path.is_dir()]
        if checkpoint_dirs:
            raise ValueError("engineering calibration output unexpectedly contains checkpoints")
        resume, resume_manifest = None, None
    else:
        recovery_session_id = checkpoint_recovery_session_id(
            launch_contract_sha256=launch_hash,
            world_size=resolved.world_size,
        )
        resume, resume_manifest, _ = recover_interrupted_checkpoints_and_resolve(
            output,
            explicit=args.resume_from_checkpoint,
            parameterization=resolved.parameterization,
            world_size=resolved.world_size,
            launch_contract_sha256=launch_hash,
            global_rank=global_rank,
            recovery_session_id=recovery_session_id,
        )
    if resume is not None and int(resume.name.split("-")[-1]) == resolved.max_steps:
        if is_primary:
            print(
                "V2_ATTEMPT_AUDIT="
                + json.dumps(
                    {
                        "run_name": resolved.run_name,
                        "attempt_id": audit_attempt_id,
                        "status": "recovered_terminal_checkpoint",
                        "starting_step": resolved.max_steps,
                        "resume_checkpoint": str(resume),
                        "resume_checkpoint_manifest_sha256": sha256_file(
                            resume / "checkpoint_manifest.json"
                        ),
                        "gradient_accumulation_steps": (
                            resolved.gradient_accumulation_steps
                        ),
                    }
                ),
                flush=True,
            )
            performance = {
                "schema_version": 1,
                "status": "recovered_from_terminal_checkpoint",
                "run_name": resolved.run_name,
                "checkpoint": str(resume),
                "checkpoint_performance_snapshot": (resume_manifest or {}).get(
                    "performance_snapshot"
                ),
                "finished_at": time.time(),
            }
            atomic_json(output / "performance.json", performance)
            atomic_json(
                output / "TRAIN_COMPLETE.json",
                {
                    "complete": True,
                    "run_name": resolved.run_name,
                    "global_step": resolved.max_steps,
                    "parameterization": resolved.parameterization,
                    "launch_contract_sha256": launch_hash,
                    "final_checkpoint": str(resume),
                    "final_checkpoint_manifest_sha256": sha256_file(
                        resume / "checkpoint_manifest.json"
                    ),
                    "performance": str(output / "performance.json"),
                    "checkpoint_performance_snapshot": (resume_manifest or {}).get(
                        "performance_snapshot"
                    ),
                    "recovered": True,
                    "nonformal": False,
                    "eligible_for_formal_schedule": True,
                    "save_load_checkpoint_receipt_sha256": (
                        sha256_file(output / "SAVE_LOAD_CHECKPOINT_RECEIPT.json")
                        if (output / "SAVE_LOAD_CHECKPOINT_RECEIPT.json").is_file()
                        else None
                    ),
                },
            )
        return

    starting_step = int(resume.name.split("-")[-1]) if resume is not None else 0
    if is_primary:
        print(
            "V2_ATTEMPT_AUDIT="
            + json.dumps(
                {
                    "run_name": resolved.run_name,
                    "attempt_id": audit_attempt_id,
                    "status": "training",
                    "starting_step": starting_step,
                    "resume_checkpoint": str(resume) if resume is not None else None,
                    "resume_checkpoint_manifest_sha256": (
                        sha256_file(resume / "checkpoint_manifest.json")
                        if resume is not None
                        else None
                    ),
                    "gradient_accumulation_steps": (
                        resolved.gradient_accumulation_steps
                    ),
                }
            ),
            flush=True,
        )

    set_seed(contract.seed)
    processor = load_processor(student_path)
    student = load_thinker(student_path)
    if resolved.parameterization == "lora":
        expected_layers = int(student.config.text_config.num_hidden_layers)
        targets = discover_v2_lora_targets(student.named_modules(), expected_layers)
        if len(targets) != expected_layers * 7 + 1:
            raise ValueError(
                f"expected {expected_layers * 7 + 1} LoRA modules, got {len(targets)}"
            )
        student = get_peft_model(
            student,
            LoraConfig(
                r=resolved.lora_rank,
                lora_alpha=resolved.lora_alpha,
                lora_dropout=contract.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=targets,
            ),
        )
        trainable = audit_trainable_parameters(student)
    else:
        targets = []
        trainable = configure_full_ft_parameters(student)
    expected_model_parameter_count = sum(
        int(parameter.numel()) for parameter in student.parameters()
    )
    expected_model_parameter_tensors = sum(1 for _ in student.parameters())

    teacher = load_thinker(teacher_path)
    teacher.to(torch.device("cuda", local_rank))
    teacher.eval()
    teacher.requires_grad_(False)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    gate = load_teacher_gate(teacher_gate_path, int(args.expected_gate_count))
    if is_primary:
        write_or_validate_exact_json(
            output / "trainable_parameters.json",
            {
                "parameterization": resolved.parameterization,
                "lora_targets": targets,
                "audit": trainable,
            },
        )

    training_started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(local_rank)
    attempt_checkpoint_steps = scheduled_checkpoint_steps(resolved)

    class FrozenCheckpointCallback(TrainerCallback):
        def __init__(self) -> None:
            self.step_end_times: list[tuple[int, float]] = []

        def on_step_end(self, training_args, state, control, **kwargs):
            self.step_end_times.append((int(state.global_step), time.monotonic()))
            if state.global_step in attempt_checkpoint_steps:
                control.should_save = True
            return control

        def on_save(self, training_args, state, control, **kwargs):
            distributed = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and resolved.world_size > 1
            )
            if distributed:
                torch.distributed.barrier()
            if state.is_world_process_zero and state.global_step in attempt_checkpoint_steps:
                checkpoint = (
                    Path(training_args.output_dir) / f"checkpoint-{state.global_step}"
                )
                manifest = write_checkpoint_manifest(
                    checkpoint,
                    parameterization=resolved.parameterization,
                    world_size=resolved.world_size,
                    launch_contract_sha256=launch_hash,
                    performance_snapshot=gpu_performance_snapshot(
                        training_started, int(state.global_step), starting_step
                    ),
                )
                if resolved.engineering_save_load_checkpoint:
                    write_engineering_checkpoint_closed(
                        checkpoint,
                        run_name=resolved.run_name,
                        parameterization=resolved.parameterization,
                        launch_contract_sha256=launch_hash,
                        checkpoint_manifest=manifest,
                    )
            if distributed:
                torch.distributed.barrier()
            return control

    training_args = TrainingArguments(
        output_dir=str(output),
        max_steps=resolved.max_steps,
        per_device_train_batch_size=resolved.per_device_train_batch_size,
        gradient_accumulation_steps=resolved.gradient_accumulation_steps,
        learning_rate=resolved.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=contract.warmup_ratio,
        weight_decay=contract.weight_decay,
        max_grad_norm=contract.max_grad_norm,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        save_total_limit=4,
        save_safetensors=True,
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=contract.seed,
        data_seed=contract.seed,
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
        optim="adamw_torch_fused",
    )
    timing_callback = FrozenCheckpointCallback()
    trainer = KeOPDV2Trainer(
        model=student,
        args=training_args,
        train_dataset=JsonlRows(train_path),
        data_collator=raw_row_collator,
        processor=processor,
        teacher=teacher,
        arm="AA",
        valid_token_ids=vocab["valid_ids"],
        teacher_gate=gate,
        contract=contract,
        aa_selector=aa_selector_for_variant(resolved.aa_selection_unit),
        scheduler_horizon=resolved.scheduler_horizon,
        contrast_mode=resolved.contrast_mode,
        contrast_beta=resolved.contrast_beta,
        contrast_margin=resolved.contrast_margin,
        contrast_teacher_ratio_threshold=(
            resolved.contrast_teacher_ratio_threshold
        ),
        contrast_temperature=resolved.contrast_temperature,
        contrast_ramp_start=resolved.contrast_ramp_start,
        contrast_ramp_end=resolved.contrast_ramp_end,
        audit_run_name=resolved.run_name,
        audit_attempt_id=audit_attempt_id,
        audit_every_microbatch=(
            contrast_round1_binding is not None
            or contrast_smoke_binding is not None
        ),
        callbacks=[timing_callback],
    )
    attempt_started = time.time()
    try:
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        if not trainer.is_world_process_zero():
            return
        final_checkpoint = output / f"checkpoint-{resolved.max_steps}"
        save_load_receipt = None
        if resolved.engineering_save_load_checkpoint:
            save_load_receipt, final_manifest = verify_engineering_checkpoint_save_load(
                final_checkpoint,
                receipt_path=output / "SAVE_LOAD_CHECKPOINT_RECEIPT.json",
                run_name=resolved.run_name,
                parameterization=resolved.parameterization,
                world_size=resolved.world_size,
                launch_contract_sha256=launch_hash,
                expected_parameter_count=expected_model_parameter_count,
                expected_parameter_tensors=expected_model_parameter_tensors,
            )
        elif resolved.engineering_calibration:
            final_manifest = None
        else:
            final_manifest = validate_checkpoint_manifest(
                final_checkpoint,
                parameterization=resolved.parameterization,
                world_size=resolved.world_size,
                launch_contract_sha256=launch_hash,
            )
        step_end_times = timing_callback.step_end_times
        if len(step_end_times) != resolved.max_steps - starting_step:
            raise ValueError(
                "optimizer-step timing count mismatch: "
                f"expected={resolved.max_steps - starting_step} "
                f"actual={len(step_end_times)}"
            )
        warmup_steps = resolved.performance_warmup_steps
        if warmup_steps > len(step_end_times) - 1:
            raise ValueError("performance warmup leaves no measured optimizer steps")
        measurement_start = (
            step_end_times[warmup_steps - 1][1]
            if warmup_steps
            else training_started
        )
        measurement_end = step_end_times[-1][1]
        measured_steps = len(step_end_times) - warmup_steps
        steady_seconds = measurement_end - measurement_start
        if steady_seconds <= 0 or measured_steps <= 0:
            raise ValueError("invalid steady-state timing window")
        steady_state = {
            "warmup_optimizer_steps": warmup_steps,
            "measured_optimizer_steps": measured_steps,
            "measurement_seconds": steady_seconds,
            "optimizer_steps_per_second": measured_steps / steady_seconds,
            "first_measured_global_step": step_end_times[warmup_steps][0],
            "last_measured_global_step": step_end_times[-1][0],
            "timing_scope": "optimizer-step tail after explicit warmup exclusion",
        }
        performance = {
            "schema_version": 1,
            "status": "complete",
            "nonformal": resolved.engineering_calibration,
            "run_name": resolved.run_name,
            "parameterization": resolved.parameterization,
            "resumed_launch": resumed_launch,
            "resume_from": str(resume) if resume else None,
            "batch": {
                "per_device": resolved.per_device_train_batch_size,
                "gradient_accumulation": resolved.gradient_accumulation_steps,
                "world_size": resolved.world_size,
                "effective": resolved.effective_batch,
            },
            "trainer_metrics": result.metrics,
            "steady_state": steady_state,
            "attempt_wall_seconds": time.time() - attempt_started,
            "final_gpu_snapshot": gpu_performance_snapshot(
                training_started, int(trainer.state.global_step), starting_step
            ),
            "save_load_checkpoint_receipt": (
                None
                if save_load_receipt is None
                else str(output / "SAVE_LOAD_CHECKPOINT_RECEIPT.json")
            ),
            "save_load_checkpoint_complete": (
                None
                if save_load_receipt is None
                else save_load_receipt.get("complete")
            ),
            "finished_at": time.time(),
        }
        atomic_json(output / "performance.json", performance)
        atomic_json(
            output / "TRAIN_COMPLETE.json",
            {
                "complete": True,
                "run_name": resolved.run_name,
                "global_step": int(trainer.state.global_step),
                "parameterization": resolved.parameterization,
                "nonformal": resolved.engineering_calibration,
                "launch_contract_sha256": launch_hash,
                "final_checkpoint": (
                    str(final_checkpoint) if final_manifest is not None else None
                ),
                "final_checkpoint_manifest_sha256": (
                    None
                    if final_manifest is None
                    else sha256_file(final_checkpoint / "checkpoint_manifest.json")
                ),
                "performance": str(output / "performance.json"),
                "checkpoint_performance_snapshot": (
                    None
                    if final_manifest is None
                    else final_manifest.get("performance_snapshot")
                ),
                "eligible_for_formal_schedule": not resolved.engineering_calibration,
                "save_load_checkpoint_receipt_sha256": (
                    None
                    if save_load_receipt is None
                    else sha256_file(output / "SAVE_LOAD_CHECKPOINT_RECEIPT.json")
                ),
            },
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "run_name": resolved.run_name,
            "parameterization": resolved.parameterization,
            "resume_from": str(resume) if resume else None,
            "error": f"{type(error).__name__}: {error}",
            "gpu_snapshot": gpu_performance_snapshot(
                training_started, int(trainer.state.global_step), starting_step
            ),
            "failed_at": time.time(),
            "global_rank": global_rank,
            "local_rank": local_rank,
        }
        atomic_json(
            output
            / "attempts"
            / f"failure_rank{global_rank}_{int(time.time() * 1000)}.json",
            failure,
        )
        raise


if __name__ == "__main__":
    main()
