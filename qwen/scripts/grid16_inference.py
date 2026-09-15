#!/usr/bin/env python3
"""Reproduce the frozen Nitro2 256-token adapter around the 96-token source.

The training contract stays at 96. Only small-benchmark evaluation replaces
its generation budget and binds the historical effective-evaluation identity.
This is the same adaptation used by the Q campaign's benchmark_eval.child.
"""
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment.common import ROOT
from scripts import eval_ke_opd_v2_vllm as evaluator

SMALL_PROMPT = "4bbd60bb2bcbc472e46c160bc87db8d2a4d69eb1f6114d4361d37a8d46ae3529"


def evaluation_prompt_sha256(mode):
    if mode == "v2_contract":
        return SMALL_PROMPT
    if mode == "ke_mmau_v051525":
        return evaluator.KE_MMAU_TEST_PROMPT_SHA256
    raise ValueError("Unknown prompt mode")


def configure(module, full):
    baseline = module.V2Contract()
    if baseline.max_completion_tokens != 96:
        raise ValueError("Training generation contract drift")
    if not full:
        module.V2Contract = lambda: replace(baseline, max_completion_tokens=256)
        module.evaluation_prompt_sha256 = evaluation_prompt_sha256


if __name__ == "__main__":
    args = evaluator.parse_args()
    configure(evaluator, args.prompt_mode == "ke_mmau_v051525")
    evaluator.main()
