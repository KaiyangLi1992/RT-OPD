#!/usr/bin/env python3
"""One GPU, one checkpoint, all four frozen datasets; raw outputs and exact closure."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment.common import ROOT, check, config, identity, read, rows, sha, verify_prepared, verify_release, verify_runtime, write


def score(raw, row):
    from ke_opd_v2.modeling import answer_content_from_text, matching_option_indices, normalize_option_text
    choices = [str(x) for x in row["choices"]]
    parsed = answer_content_from_text(raw, choices)
    indices = matching_option_indices(parsed, choices) if parsed is not None else []
    correct = parsed is not None and normalize_option_text(parsed) == normalize_option_text(choices[int(row["gold_index"])])
    return dict(parsed_answer=parsed, parsed_indices=indices, parseable=parsed is not None, correct=correct)


def validate_predictions(predictions, gold):
    if [r["id"] for r in predictions] != [r["id"] for r in gold]:
        raise ValueError("Evaluation IDs missing, duplicated, reordered, or foreign")
    for item, row in zip(predictions, gold):
        expected = score(item["raw_output"], row)
        if any(item[k] != v for k, v in expected.items()) or item["gold_index"] != row["gold_index"]:
            raise ValueError("Prediction/parser/gold mismatch")
        if item["generated_tokens"] > 256:
            raise ValueError("Generation limit violated")
    correct = sum(r["correct"] for r in predictions)
    return dict(rows=len(gold), correct=correct, accuracy=correct / len(gold),
                parseable=sum(r["parseable"] for r in predictions))



if __name__ == "__main__":
    raise SystemExit("Use scripts/grid16_eval.py for the frozen main-experiment evaluation.")
