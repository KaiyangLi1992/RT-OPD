"""Qwen2.5-Omni loading, audio preparation, generation, and span utilities."""

from __future__ import annotations

import json
import re
import unicodedata
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

LEXICAL_RE = re.compile(r"[^\W_]+(?:[.'’\-/&][^\W_]+)*", re.UNICODE)


def load_thinker(
    model_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> torch.nn.Module:
    """Load a thinker-only or full Omni checkpoint as a Thinker module."""
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    path = Path(model_path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") == "qwen2_5_omni":
        transformers_version = version("transformers")
        if transformers_version != "4.52.4":
            raise RuntimeError(
                "Ke-Omni-R full checkpoints require transformers==4.52.4; "
                f"found {transformers_version}. Other tested 4.57.x stacks produce "
                "vocabulary-offset gibberish."
            )
        class TextOnlyOmni(Qwen2_5OmniForConditionalGeneration):
            """Skip the unused speech-output speaker map during text-only loading."""

            def load_speakers(self, speaker_path: str | Path) -> None:
                return None

        # Ke-Omni-R is a full Omni checkpoint whose safetensor keys carry the
        # ``thinker.`` prefix.  Loading it through the full wrapper preserves
        # the checkpoint's generation config; direct Thinker loading is known
        # to produce vocabulary-offset gibberish in unsupported stacks.
        wrapper = TextOnlyOmni.from_pretrained(
            path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            enable_audio_output=False,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        return wrapper.thinker
    return Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )


def load_processor(model_path: str | Path) -> Any:
    from transformers import Qwen2_5OmniProcessor

    return Qwen2_5OmniProcessor.from_pretrained(
        model_path, use_fast=True, local_files_only=True
    )


def load_audio(path: str | Path, sampling_rate: int) -> Any:
    import librosa

    waveform, _ = librosa.load(str(path), sr=sampling_rate, mono=True)
    return waveform


def match_waveform_length(waveform: Any, target_samples: int) -> Any:
    import numpy as np

    if target_samples <= 0:
        raise ValueError("target waveform length must be positive")
    value = np.asarray(waveform, dtype=np.float32)
    if value.size == target_samples:
        return value
    if value.size > target_samples:
        # Center cropping avoids a systematic bias toward beginnings.
        start = (value.size - target_samples) // 2
        return value[start : start + target_samples]
    repeats = (target_samples + value.size - 1) // value.size
    return np.tile(value, repeats)[:target_samples]


def chat_text(processor: Any, canonical_user_text: str) -> str:
    user_message = {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "unused-by-template"},
            {"type": "text", "text": canonical_user_text},
        ],
    }
    # The official Ke-Omni-R MCQ examples use an audio+text user message with
    # no extra system turn.  Keep this unconditional so an environment variable
    # cannot silently change the frozen serialized prefix.
    messages = [user_message]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prepare_prompt_inputs(
    processor: Any,
    canonical_user_text: str,
    waveform: Any,
    sampling_rate: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    inputs = processor(
        text=chat_text(processor, canonical_user_text),
        audio=[waveform],
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    result: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            continue
        value = value.to(device)
        if torch.is_floating_point(value):
            value = value.to(torch.bfloat16)
        result[key] = value
    return result


def append_completion(
    prompt_inputs: dict[str, torch.Tensor], completion_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    if completion_ids.ndim == 1:
        completion_ids = completion_ids.unsqueeze(0)
    result = dict(prompt_inputs)
    result["input_ids"] = torch.cat([prompt_inputs["input_ids"], completion_ids], dim=1)
    suffix_mask = torch.ones_like(completion_ids, dtype=prompt_inputs["attention_mask"].dtype)
    result["attention_mask"] = torch.cat([prompt_inputs["attention_mask"], suffix_mask], dim=1)
    # Cached position IDs describe the prompt only; let the model rebuild them.
    result.pop("position_ids", None)
    return result


def first_complete_answer_end(tokenizer: Any, token_ids: torch.Tensor) -> int:
    ids = token_ids.tolist()
    for end in range(1, len(ids) + 1):
        text = tokenizer.decode(ids[:end], skip_special_tokens=False)
        if "</answer>" in text:
            return end
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos in ids:
        return ids.index(eos) + 1
    return len(ids)


def completion_word_ids(tokenizer: Any, token_ids: list[int]) -> list[int | None]:
    """Map lexical reasoning words to IDs, excluding tags and the answer region.

    A complete ``<think>`` span is preferred. If a rollout violates the requested
    wrapper, lexical free text before ``<answer>`` remains eligible so AA does not
    silently collapse to Uniform solely because of formatting.
    """
    assignments: list[int | None] = [None] * len(token_ids)
    # Never encode the decoded text again: tags can merge with adjacent content
    # under BPE (for example ``>D``), so a separately encoded ``<answer>`` is
    # not guaranteed to occur as a token-ID subsequence.  Instead, concatenate
    # the pieces from the *original sampled IDs* and project character spans
    # back onto those same token positions.
    pieces = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]
    rendered = "".join(pieces)
    think_open = re.search(r"<think\b[^>]*>", rendered, re.I)
    think_close = (
        re.search(r"</think\s*>", rendered[think_open.end() :], re.I)
        if think_open is not None
        else None
    )
    if think_open is not None and think_close is not None:
        content_start = think_open.end()
        content_end = think_open.end() + think_close.start()
    else:
        answer_open = re.search(r"<answer\b[^>]*>", rendered, re.I)
        content_start = 0
        content_end = answer_open.start() if answer_open is not None else len(rendered)

    current_word = -1
    previous_ended_lexical = False
    cursor = 0
    for token_index, piece in enumerate(pieces):
        token_start = cursor
        token_end = cursor + len(piece)
        cursor = token_end
        overlap_start = max(token_start, content_start)
        overlap_end = min(token_end, content_end)
        if overlap_start >= overlap_end:
            previous_ended_lexical = False
            continue
        eligible_piece = rendered[overlap_start:overlap_end]
        match = LEXICAL_RE.search(eligible_piece)
        if match is None:
            previous_ended_lexical = False
            continue
        prefix = eligible_piece[: match.start()]
        starts_new = current_word < 0 or bool(prefix) or not previous_ended_lexical
        if starts_new:
            current_word += 1
        assignments[token_index] = current_word
        previous_ended_lexical = bool(LEXICAL_RE.search(eligible_piece[match.start() :])) and bool(
            re.search(r"[^\W_]$", eligible_piece, re.UNICODE)
        )
    return assignments


def completion_word_texts(tokenizer: Any, token_ids: list[int]) -> dict[int, str]:
    """Return exact lexical text for every frozen V1 word ID.

    The V1 token-to-word mapper can split a contraction or another connected
    lexeme at a BPE boundary (for example ``It`` + ``'s``).  Preserve those
    historical IDs for matched ablations, but map each split ID back to the
    same complete rendered lexeme so exact whole-word filters remain valid.
    """
    pieces = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]
    rendered = "".join(pieces)
    think_open = re.search(r"<think\b[^>]*>", rendered, re.I)
    think_close = (
        re.search(r"</think\s*>", rendered[think_open.end() :], re.I)
        if think_open is not None
        else None
    )
    if think_open is not None and think_close is not None:
        content_start = think_open.end()
        content_end = think_open.end() + think_close.start()
    else:
        answer_open = re.search(r"<answer\b[^>]*>", rendered, re.I)
        content_start = 0
        content_end = answer_open.start() if answer_open is not None else len(rendered)
    matches = list(LEXICAL_RE.finditer(rendered, content_start, content_end))
    assignments = completion_word_ids(tokenizer, token_ids)
    texts: dict[int, str] = {}
    cursor = 0
    for piece, word_id in zip(pieces, assignments):
        token_start = cursor
        token_end = cursor + len(piece)
        cursor = token_end
        if word_id is None:
            continue
        candidates = [
            match
            for match in matches
            if max(token_start, match.start()) < min(token_end, match.end())
            and re.search(
                r"[^\W_]",
                rendered[
                    max(token_start, match.start()) : min(token_end, match.end())
                ],
                re.UNICODE,
            )
        ]
        if not candidates:
            raise ValueError(
                "sampled-token word ID has no overlapping complete lexeme: "
                f"word_id={word_id} piece={piece!r}"
            )
        # A sampled token can theoretically contain more than one lexical
        # word.  V1 assigns that token to its first lexical match, so retain
        # the same deterministic choice here.
        text = candidates[0].group(0)
        previous = texts.setdefault(word_id, text)
        if previous != text:
            raise ValueError(
                "sampled-token word ID overlaps multiple complete lexemes: "
                f"word_id={word_id} texts={[previous, text]}"
            )
    assigned = {word_id for word_id in assignments if word_id is not None}
    if assigned != set(texts):
        raise ValueError(
            "lexical text projection lost sampled-token word IDs: "
            f"assigned={sorted(assigned)} texts={sorted(texts)}"
        )
    return texts


def strict_answer_letter_from_text(text: str, valid_letters: str = "ABCD") -> str | None:
    letter_class = re.escape(valid_letters.upper())
    match = re.search(
        rf"<answer\b[^>]*>\s*([{letter_class}])\s*</answer\s*>",
        text,
        re.I | re.S,
    )
    return match.group(1).upper() if match else None


def answer_letter_from_text(text: str, valid_letters: str = "ABCD") -> str | None:
    strict = strict_answer_letter_from_text(text, valid_letters)
    if strict:
        return strict
    # Formatting compliance is reported separately from answer correctness.
    # A shared conservative fallback prevents XML obedience from becoming an
    # accidental method advantage in the primary MCQ metric.
    letter_class = re.escape(valid_letters.upper())
    matches = re.findall(
        rf"(?:answer|option|choice)\s*(?:is\s*)?[:=\-]?\s*[\(\[]?([{letter_class}])[\)\]]?\b",
        text,
        re.I,
    )
    if matches:
        return matches[-1].upper()
    final = re.search(rf"(?:^|\s)([{letter_class}])\s*[.。)]?\s*$", text.strip(), re.I)
    return final.group(1).upper() if final else None


def normalize_option_text(text: str) -> str:
    """Normalize formatting noise without changing option semantics."""
    return " ".join(unicodedata.normalize("NFKC", str(text)).split()).casefold()


def _unique_option_index(candidate: str, choices: list[str]) -> int | None:
    normalized = normalize_option_text(candidate)
    if not normalized:
        return None
    matches = [
        index
        for index, choice in enumerate(choices)
        if normalize_option_text(choice) == normalized
    ]
    return matches[0] if len(matches) == 1 else None


def matching_option_indices(candidate: str, choices: list[str]) -> list[int]:
    """Return every option whose normalized content matches ``candidate``."""
    normalized = normalize_option_text(candidate)
    if not normalized:
        return []
    return [
        index
        for index, choice in enumerate(choices)
        if normalize_option_text(choice) == normalized
    ]


def strict_answer_content_from_text(text: str, choices: list[str]) -> str | None:
    """Parse one answer tag and validate its payload against option content."""
    payloads = re.findall(
        r"<answer\b[^>]*>\s*(.*?)\s*</answer\s*>",
        str(text),
        re.I | re.S,
    )
    if len(payloads) != 1 or not matching_option_indices(payloads[0], choices):
        return None
    return payloads[0].strip()


def answer_content_from_text(text: str, choices: list[str]) -> str | None:
    """Content parser that also accepts a conservative unwrapped final line."""
    strict = strict_answer_content_from_text(text, choices)
    if strict is not None:
        return strict
    raw = str(text).strip()
    if matching_option_indices(raw, choices):
        return raw
    nonempty = [line.strip() for line in raw.splitlines() if line.strip()]
    if nonempty:
        final = re.sub(
            r"^(?:final\s+)?(?:answer|option|choice)\s*(?:is\s*)?[:=\-]?\s*",
            "",
            nonempty[-1],
            flags=re.I,
        )
        if matching_option_indices(final, choices):
            return final
    return None


def strict_answer_index_from_text(text: str, choices: list[str]) -> int | None:
    """Parse exactly one answer tag whose payload is one exact option text."""
    content = strict_answer_content_from_text(text, choices)
    return _unique_option_index(content, choices) if content is not None else None


def answer_index_from_text(text: str, choices: list[str]) -> int | None:
    """Content-based MCQ parser with no option-letter fallback.

    Formatting compliance is reported by ``strict_answer_index_from_text``.
    The primary accuracy parser additionally accepts a bare exact option or a
    final ``Answer: <exact option>`` line, but never maps A/B/C/... to indices.
    """
    content = answer_content_from_text(text, choices)
    return _unique_option_index(content, choices) if content is not None else None
