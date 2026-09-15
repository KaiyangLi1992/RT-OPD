"""Exact LoRA target policy: audio projection plus every Thinker decoder layer."""

from __future__ import annotations

import re
from collections.abc import Iterable


DECODER_PROJECTIONS = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)
LAYER_RE = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.(?:self_attn|mlp)\.([^.]+)$")


def is_v2_lora_module(name: str) -> bool:
    """True only for Thinker decoder projections or the audio tower output projection."""
    normalized = name.removeprefix("thinker.")
    if normalized == "audio_tower.proj":
        return True
    match = LAYER_RE.search(normalized)
    return bool(match and match.group(2) in DECODER_PROJECTIONS)


def discover_v2_lora_targets(named_modules: Iterable[tuple[str, object]], expected_layers: int) -> list[str]:
    targets: list[str] = []
    seen_layers: set[int] = set()
    for name, module in named_modules:
        if not is_v2_lora_module(name):
            continue
        # Avoid importing torch in contract-only tools; structural check is enough here.
        if module.__class__.__name__ != "Linear":
            continue
        targets.append(name)
        match = LAYER_RE.search(name.removeprefix("thinker."))
        if match:
            seen_layers.add(int(match.group(1)))
    missing = set(range(expected_layers)) - seen_layers
    if missing:
        raise ValueError(f"LoRA discovery missed Thinker decoder layers: {sorted(missing)}")
    if not any(name.removeprefix("thinker.") == "audio_tower.proj" for name in targets):
        raise ValueError("LoRA discovery missed audio_tower.proj")
    return sorted(targets)


def audit_trainable_parameters(model: object) -> dict[str, object]:
    trainable = [(name, param.numel()) for name, param in model.named_parameters() if param.requires_grad]
    forbidden = [
        name
        for name, _ in trainable
        if ("audio_tower." in name and "audio_tower.proj" not in name)
        or ".visual." in name
        or name.startswith("visual.")
    ]
    if forbidden:
        raise ValueError(f"frozen encoder/vision parameters became trainable: {forbidden[:20]}")
    return {
        "trainable_parameters": sum(count for _, count in trainable),
        "trainable_tensors": len(trainable),
        "names": [name for name, _ in trainable],
    }
