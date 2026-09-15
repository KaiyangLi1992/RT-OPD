"""Frozen experiment contract shared by data, training, and evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal


SYSTEM_PROMPT: Final = None
REASONING_INSTRUCTION: Final = """Output the thinking process (less than 50 words) in <think> </think> and final answer in <answer> </answer>.
The thinking process must be non-empty. In <answer>, copy the exact text of one option; do not output an option letter."""
DIRECT_INSTRUCTION: Final = """Output only the final answer in <answer> </answer>.
In <answer>, copy the exact text of one option; do not output an option letter or any other text."""
LETTERS: Final = ("A", "B", "C", "D")
BENCHMARK_LETTERS: Final = tuple("ABCDEFGH")
QUESTION_TYPES: Final = ("speech", "sound", "music", "temporal")
CHECKPOINT_STEPS: Final = (79, 157, 235, 313)


@dataclass(frozen=True)
class V2RunSpec:
    run_name: str
    initialization: Literal["Q", "K"]
    arm: Literal["AO", "U", "AA"]
    learning_rate: float


RUN_SPECS: Final = (
    V2RunSpec("K-AA-lr5e5", "K", "AA", 5e-5),
    V2RunSpec("Q-AA-lr5e5", "Q", "AA", 5e-5),
    V2RunSpec("K-AA-lr1e5", "K", "AA", 1e-5),
    V2RunSpec("Q-AA-lr1e5", "Q", "AA", 1e-5),
    V2RunSpec("K-U-lr5e5", "K", "U", 5e-5),
    V2RunSpec("Q-U-lr5e5", "Q", "U", 5e-5),
    V2RunSpec("K-AO-lr5e5", "K", "AO", 5e-5),
    V2RunSpec("Q-AO-lr5e5", "Q", "AO", 5e-5),
)


@dataclass(frozen=True)
class V2Contract:
    version: str = "ke_opd_v2_native_content_2026-07-21"
    seed: int = 42
    max_completion_tokens: int = 96
    rollout_temperature: float = 1.0
    rollout_top_p: float = 0.95
    rollout_top_k: int = 64
    kl_temperature: float = 1.0
    lambda_opd: float = 0.25
    aa_high_weight: float = 2.0
    aa_top_fraction: float = 0.20
    teacher_support_top_k: int = 128
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    world_size: int = 4
    optimizer_steps: int = 313

    @property
    def global_batch_size(self) -> int:
        return (
            self.per_device_train_batch_size
            * self.gradient_accumulation_steps
            * self.world_size
        )

    @property
    def prompt_sha256(self) -> str:
        payload = canonical_prompt_payload()
        return sha256_json(payload)


def canonical_prompt_payload() -> dict[str, object]:
    return {
        "version": "ke_native_content_prompt_v1",
        "system": "omitted, matching the official Ke-Omni-R MCQ examples",
        "audio_placement": "first item in user content",
        "question_format": "{question}",
        "option_format": "Python-style list of exact option strings",
        "reasoning_instruction": REASONING_INSTRUCTION,
        "direct_instruction": DIRECT_INSTRUCTION,
        "reasoning_view": "OPD training and every validation/test evaluation",
        "direct_view": "shared answer-only CE view for all AO/U/AA arms",
        "answer_representation": "exact option text inside <answer> </answer>",
        "max_completion_tokens": 96,
        "stop": "first complete </answer>",
    }


def _clean_choices(choices: list[str], *, minimum: int, maximum: int) -> list[str]:
    if not minimum <= len(choices) <= maximum:
        raise ValueError(
            f"V2 requires {minimum}--{maximum} choices, got {len(choices)}"
        )
    cleaned = [str(choice).strip() for choice in choices]
    if any(not choice for choice in cleaned):
        raise ValueError("V2 choices must be non-empty")
    if len({choice.casefold() for choice in cleaned}) != len(cleaned):
        raise ValueError("V2 answer-content format requires unique option texts")
    return cleaned


def _clean_benchmark_choices(choices: list[str]) -> list[str]:
    """Keep official content choices while dropping empty padding slots.

    Some Core-3 source rows contain duplicated option text or a trailing empty
    padding slot.  Content-format evaluation can represent duplicated text, but
    an empty string is not an answer the model can copy.  Training assets remain
    strictly four-way, non-empty, and unique through ``_clean_choices``.
    """
    if not 2 <= len(choices) <= len(BENCHMARK_LETTERS):
        raise ValueError(f"V2 requires 2--{len(BENCHMARK_LETTERS)} choices, got {len(choices)}")
    cleaned = [str(choice).strip() for choice in choices]
    cleaned = [choice for choice in cleaned if choice]
    if len(cleaned) < 2:
        raise ValueError("V2 benchmark row has fewer than two non-empty choices")
    return cleaned


def _native_question_text(question: str, choices: list[str]) -> str:
    return f"{question.strip()} {choices!r}"


def canonical_user_text(question: str, choices: list[str]) -> str:
    """Ke-Omni-native mandatory-reasoning view for four-way training rows."""
    cleaned = _clean_choices(choices, minimum=4, maximum=4)
    return f"{_native_question_text(question, cleaned)}\n{REASONING_INSTRUCTION}"


def direct_user_text(question: str, choices: list[str]) -> str:
    """Shared answer-only CE view; never used by OPD or final evaluation."""
    cleaned = _clean_choices(choices, minimum=4, maximum=4)
    return f"{_native_question_text(question, cleaned)}\n{DIRECT_INSTRUCTION}"


def benchmark_user_text(question: str, choices: list[str]) -> str:
    """Render the same frozen prompt template for official 2--8 way MCQs.

    V2 training assets remain strictly four-way.  The official Core-3 files,
    however, contain a small number of 2/3/5/6/8-way questions.  Dropping those
    rows would change benchmark coverage, so evaluation varies only the number
    of rendered option lines while preserving the instruction and output form.
    """
    cleaned = _clean_benchmark_choices(choices)
    return f"{_native_question_text(question, cleaned)}\n{REASONING_INSTRUCTION}"


def sha256_json(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_frozen_contract(output_dir: str | Path) -> dict[str, object]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    contract = V2Contract()
    prompt = canonical_prompt_payload()
    payload = {
        "contract": asdict(contract),
        "prompt": prompt,
        "prompt_sha256": sha256_json(prompt),
        "runs": [asdict(spec) for spec in RUN_SPECS],
        "checkpoint_steps": list(CHECKPOINT_STEPS),
    }
    path = root / "v2_contract.json"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return payload
