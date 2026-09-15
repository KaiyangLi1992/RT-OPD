#!/usr/bin/env python3
"""Canonical-prompt vLLM evaluation for a merged V2 thinker checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from ke_opd_v2.contract import BENCHMARK_LETTERS, V2Contract, benchmark_user_text
from ke_opd_v2.modeling import (
    answer_content_from_text,
    chat_text,
    load_audio,
    matching_option_indices,
    normalize_option_text,
    strict_answer_content_from_text,
)


PROMPT_MODES = ("v2_contract", "ke_mmau_v051525")
AUDIO_INTERVENTIONS = ("matched", "silence_same_duration")
KE_MMAU_TEST_PROMPT_PAYLOAD = {
    "version": "ke_mmau_v051525",
    "source": "source/original/Ke-Omni-R/src/test.py::_get_message",
    "think": True,
    "think_max_len_words": 50,
    "system_turn": None,
    "audio_placement": "first user-content item",
    "template": (
        "{question} Please choose the answer from the following options: "
        "{choices_repr}. Output the thinking process(less than 50 words) in "
        "<think> </think> and final answer in <answer> </answer>."
    ),
    "generation": {
        "do_sample": False,
        "max_new_tokens": 256,
        "stop": "model EOS only",
        "vocabulary_suppression": False,
    },
}


def sha256_json(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


KE_MMAU_TEST_PROMPT_SHA256 = sha256_json(KE_MMAU_TEST_PROMPT_PAYLOAD)


def evaluation_prompt_sha256(prompt_mode: str) -> str:
    if prompt_mode == "v2_contract":
        return V2Contract().prompt_sha256
    if prompt_mode == "ke_mmau_v051525":
        return KE_MMAU_TEST_PROMPT_SHA256
    raise ValueError(f"unsupported prompt mode: {prompt_mode}")


def ke_mmau_test_user_text(question: str, choices: list[str]) -> str:
    """Reproduce Ke-Omni-R ``src/test.py --think True --think_max_len 50``."""

    choice_str = f"Please choose the answer from the following options: {choices}."
    return (
        f"{question} {choice_str} Output the thinking process(less than 50 words) "
        "in <think> </think> and final answer in <answer> </answer>."
    )


def stable_shard(sample_id: str, count: int) -> int:
    return int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "big") % count


def select_shard_rows(
    rows: list[dict],
    *,
    count: int,
    index: int,
    batch_size: int,
    mode: str,
) -> list[dict]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid shard count/index")
    if mode == "stable_hash":
        return [
            row for row in rows if stable_shard(str(row["id"]), count) == index
        ]
    if mode == "batch_stride":
        return [
            row
            for row_index, row in enumerate(rows)
            if (row_index // batch_size) % count == index
        ]
    raise ValueError(f"unsupported shard mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-model-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--vocab-json", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument(
        "--shard-mode",
        choices=("stable_hash", "batch_stride"),
        default="stable_hash",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="Inference dtype; freeze one value for every checkpoint in a campaign.",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--warmup-batches", type=int, default=0)
    parser.add_argument("--warmup-jsonl")
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, default="v2_contract")
    parser.add_argument(
        "--audio-intervention",
        choices=AUDIO_INTERVENTIONS,
        default="matched",
        help=(
            "Audio input used for inference. silence_same_duration replaces each "
            "decoded waveform with exact float32 zeros while preserving its sample count."
        ),
    )
    parser.add_argument(
        "--answers-hidden",
        action="store_true",
        help="Do not require or emit gold labels (for the 9,000-sample MMAU test split).",
    )
    return parser.parse_args()


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result = set()
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("id") and row.get("error") is None:
            result.add(str(row["id"]))
    return result


def launch_prompt_sha256(launch: dict[str, object]) -> str | None:
    """Read both legacy V2 and generic hparam launch-contract layouts."""
    prompt_sha256 = launch.get("prompt_sha256")
    if isinstance(prompt_sha256, str):
        return prompt_sha256
    immutable = launch.get("immutable")
    if isinstance(immutable, dict):
        nested = immutable.get("prompt_sha256")
        if isinstance(nested, str):
            return nested
    contract = launch.get("contract")
    if isinstance(contract, dict):
        nested = contract.get("prompt_sha256")
        if isinstance(nested, str):
            return nested
    return None


def launch_parameterization(launch: dict[str, object]) -> str | None:
    direct = launch.get("parameterization")
    if isinstance(direct, str):
        return direct
    immutable = launch.get("immutable")
    if not isinstance(immutable, dict):
        return None
    nested = immutable.get("parameterization")
    if isinstance(nested, str):
        return nested
    resolved = immutable.get("resolved")
    if isinstance(resolved, dict) and isinstance(resolved.get("parameterization"), str):
        return str(resolved["parameterization"])
    return None


def make_request(
    processor,
    row: dict,
    sampling_rate: int,
    *,
    prompt_mode: str = "v2_contract",
    audio_intervention: str = "matched",
) -> dict[str, object]:
    choices = [str(choice) for choice in row["choices"]]
    if prompt_mode == "v2_contract":
        user_text = row.get("canonical_user_text") or benchmark_user_text(
            str(row["question"]), choices
        )
    elif prompt_mode == "ke_mmau_v051525":
        user_text = ke_mmau_test_user_text(str(row["question"]), choices)
    else:
        raise ValueError(f"unsupported prompt mode: {prompt_mode}")
    rendered = chat_text(processor, user_text)
    if isinstance(rendered, list) and len(rendered) == 1:
        prompt = rendered[0]
    elif isinstance(rendered, str):
        prompt = rendered
    else:
        raise ValueError("canonical chat serialization did not return one prompt")
    waveform = load_audio(row["audio_path"], sampling_rate)
    if audio_intervention == "silence_same_duration":
        import numpy as np

        waveform = np.zeros_like(np.asarray(waveform), dtype=np.float32)
    elif audio_intervention != "matched":
        raise ValueError(f"unsupported audio intervention: {audio_intervention}")
    return {
        "prompt": prompt,
        "multi_modal_data": {"audio": (waveform, sampling_rate)},
    }


def _nvidia_smi_identity(cuda_visible_devices: str | None) -> dict[str, object]:
    """Best-effort physical GPU identity when PyTorch omits the UUID."""
    try:
        target = (
            os.environ.get("SLURM_JOB_GPUS") or cuda_visible_devices or ""
        ).split(",")[0].strip()
        command = [
                "nvidia-smi",
                "--query-gpu=index,name,uuid",
                "--format=csv,noheader,nounits",
            ]
        if target:
            command[1:1] = ["--id", target]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return {}
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) == 3:
            rows.append({"physical_index": fields[0], "name": fields[1], "uuid": fields[2]})
    visible = (cuda_visible_devices or "").split(",")[0].strip()
    if visible:
        for row in rows:
            if visible in {str(row["physical_index"]), str(row["uuid"])}:
                return row
    return rows[0] if len(rows) == 1 else {}


def gpu_runtime_receipt(torch_module) -> dict[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    receipt: dict[str, object] = {
        "cuda_available": bool(torch_module.cuda.is_available()),
        "host": os.uname().nodename,
        "cuda_visible_devices": visible,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }
    if not receipt["cuda_available"]:
        return receipt
    index = int(torch_module.cuda.current_device())
    properties = torch_module.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    fallback = _nvidia_smi_identity(visible) if uuid is None else {}
    receipt.update(
        {
            "logical_device_index": index,
            "name": str(getattr(properties, "name", fallback.get("name", "unknown"))),
            "uuid": str(uuid or fallback.get("uuid") or "unknown"),
            "physical_index": fallback.get("physical_index"),
            "total_memory_bytes": int(getattr(properties, "total_memory", 0)),
            "compute_capability": [
                int(getattr(properties, "major", 0)),
                int(getattr(properties, "minor", 0)),
            ],
        }
    )
    return receipt


def peak_memory_receipt(torch_module, device_index: int | None) -> dict[str, object]:
    if device_index is None or not torch_module.cuda.is_available():
        return {
            "measurement_scope": "evaluator_python_process",
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    torch_module.cuda.synchronize(device_index)
    return {
        "measurement_scope": "evaluator_python_process",
        "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated(device_index)),
        "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved(device_index)),
    }


def nvidia_driver_version() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        raise RuntimeError(f"ambiguous NVIDIA driver versions: {sorted(versions)}")
    return versions.pop()


def evaluation_runtime_receipt(
    *,
    torch_module,
    transformers_module,
    vllm_module,
    tokenizers_module,
    numpy_module,
    librosa_module,
    soundfile_module,
    soxr_module,
    safetensors_module,
) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "vllm": str(vllm_module.__version__),
        "vllm_distribution": metadata.version("vllm"),
        "torch": str(torch_module.__version__),
        "transformers": str(transformers_module.__version__),
        "tokenizers": str(tokenizers_module.__version__),
        "numpy": str(numpy_module.__version__),
        "librosa": str(librosa_module.__version__),
        "soundfile": str(soundfile_module.__version__),
        "soxr": str(soxr_module.__version__),
        "safetensors": str(safetensors_module.__version__),
        "torch_cuda_version": str(torch_module.version.cuda),
        "cudnn_version": int(torch_module.backends.cudnn.version()),
        "nvidia_driver_version": nvidia_driver_version(),
        "cuda": bool(torch_module.cuda.is_available()),
    }


def partition_forbidden_token_ids(
    token_ids: list[int], logit_bias_vocab_size: int
) -> tuple[list[int], list[int]]:
    """Separate IDs accepted by vLLM's tokenizer-bounded logit-bias API."""

    if logit_bias_vocab_size <= 0:
        raise ValueError(
            f"invalid vLLM logit-bias vocabulary size: {logit_bias_vocab_size}"
        )
    normalized = [int(token_id) for token_id in token_ids]
    if len(set(normalized)) != len(normalized):
        raise ValueError("frozen forbidden token IDs must be unique")
    effective = [
        token_id for token_id in normalized if 0 <= token_id < logit_bias_vocab_size
    ]
    out_of_vocab = [
        token_id
        for token_id in normalized
        if token_id < 0 or token_id >= logit_bias_vocab_size
    ]
    return effective, out_of_vocab


def main() -> None:
    import librosa
    import numpy
    import safetensors
    import soundfile
    import soxr
    import tokenizers
    import torch
    import transformers
    import vllm
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    runtime = evaluation_runtime_receipt(
        torch_module=torch,
        transformers_module=transformers,
        vllm_module=vllm,
        tokenizers_module=tokenizers,
        numpy_module=numpy,
        librosa_module=librosa,
        soundfile_module=soundfile,
        soxr_module=soxr,
        safetensors_module=safetensors,
    )

    args = parse_args()
    process_started = time.perf_counter()
    runtime_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = runtime_bin + os.pathsep + os.environ.get("PATH", "")
    if args.batch_size < 1 or args.max_samples < 0 or args.warmup_batches < 0:
        raise ValueError("--batch-size must be positive and --max-samples nonnegative")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid --num-shards/--shard-index")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be between zero and one")
    contract = V2Contract()
    eval_prompt_sha256 = evaluation_prompt_sha256(args.prompt_mode)
    merged_model_dir = Path(args.merged_model_dir).resolve()
    materialization = json.loads(
        (merged_model_dir / "vllm_materialization.json").read_text(encoding="utf-8")
    )
    if not materialization.get("complete"):
        raise ValueError("merged vLLM model is not complete")
    launch_path = Path(args.adapter_dir).resolve().parent / "launch_contract.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    prompt_sha256 = launch_prompt_sha256(launch)
    if prompt_sha256 != contract.prompt_sha256:
        raise ValueError(f"checkpoint was not trained under the active prompt: {launch_path}")
    parameterization = launch_parameterization(launch)

    processor = AutoProcessor.from_pretrained(
        merged_model_dir, local_files_only=True, use_fast=True
    )
    sampling_rate = int(processor.feature_extractor.sampling_rate)
    vocab = json.loads(Path(args.vocab_json).read_text(encoding="utf-8"))
    source_forbidden_token_ids = list(map(int, vocab["forbidden_ids"]))
    output = Path(args.output_jsonl)
    audit_path = Path(args.audit_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(output)
    rows = [
        json.loads(line)
        for line in Path(args.input_jsonl).open(encoding="utf-8")
        if line.strip()
    ]
    rows = select_shard_rows(
        rows,
        count=args.num_shards,
        index=args.shard_index,
        batch_size=args.batch_size,
        mode=args.shard_mode,
    )
    if args.max_samples and args.max_samples < len(rows):
        if args.max_samples == 1:
            rows = [rows[0]]
        else:
            last = len(rows) - 1
            rows = [
                rows[round(index * last / (args.max_samples - 1))]
                for index in range(args.max_samples)
            ]
    pending_rows = [row for row in rows if str(row["id"]) not in done]
    warmup_pool = pending_rows
    if args.warmup_jsonl:
        warmup_pool = [
            json.loads(line)
            for line in Path(args.warmup_jsonl).open(encoding="utf-8")
            if line.strip()
        ]
        measured_ids = {str(row["id"]) for row in rows}
        warmup_ids = [str(row["id"]) for row in warmup_pool]
        if (
            not warmup_pool
            or len(set(warmup_ids)) != len(warmup_ids)
            or measured_ids.intersection(warmup_ids)
        ):
            raise ValueError("warmup rows must be nonempty, unique, and disjoint from measured rows")

    gpu = gpu_runtime_receipt(torch)
    gpu_index = (
        int(gpu["logical_device_index"])
        if gpu.get("logical_device_index") is not None
        else None
    )
    if gpu_index is not None:
        torch.cuda.reset_peak_memory_stats(gpu_index)
    engine_started = time.perf_counter()
    llm = LLM(
        model=str(merged_model_dir),
        generation_config="vllm",
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        seed=1234,
    )
    engine_load_seconds = time.perf_counter() - engine_started
    model_vocab_size = int(llm.llm_engine.model_config.get_vocab_size())
    tokenizer_vocab_size = len(llm.get_tokenizer())
    logit_bias_vocab_size = min(model_vocab_size, tokenizer_vocab_size)
    if args.prompt_mode == "v2_contract":
        forbidden_token_ids, out_of_vocab_forbidden_token_ids = (
            partition_forbidden_token_ids(
                source_forbidden_token_ids, logit_bias_vocab_size
            )
        )
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=contract.max_completion_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True,
            # Match Transformers' suppress_tokens exactly.  The vLLM
            # allowed-token list is capped at 1,024 entries, whereas this
            # tokenizer has more than 150k valid IDs.  The complementary deny
            # list has only a few hundred IDs and -inf makes their logits
            # impossible to sample.
            logit_bias={token_id: float("-inf") for token_id in forbidden_token_ids},
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
        )
        generation_contract = {
            "do_sample": False,
            "max_new_tokens": contract.max_completion_tokens,
            "stop": ["</answer>"],
            "include_stop_str_in_output": True,
            "vocabulary_suppression": True,
        }
    else:
        # Match Ke-Omni-R's public MMAU test entry point: greedy generation,
        # 256 new tokens, model-EOS termination, and no vocabulary filtering.
        forbidden_token_ids = []
        out_of_vocab_forbidden_token_ids = []
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
        )
        generation_contract = dict(KE_MMAU_TEST_PROMPT_PAYLOAD["generation"])

    warmup_seconds = 0.0
    warmup_rows = 0
    if args.warmup_batches:
        if len(warmup_pool) < args.batch_size:
            raise ValueError("warmup pool must contain at least one full configured batch")
        for warmup_index in range(args.warmup_batches):
            offset = (warmup_index * args.batch_size) % len(warmup_pool)
            warmup_batch = warmup_pool[offset : offset + args.batch_size]
            if len(warmup_batch) < args.batch_size:
                warmup_batch = (
                    warmup_batch
                    + warmup_pool[: args.batch_size - len(warmup_batch)]
                )
            warmup_started = time.perf_counter()
            warmup_requests = [
                make_request(
                    processor,
                    row,
                    sampling_rate,
                    prompt_mode=args.prompt_mode,
                    audio_intervention=args.audio_intervention,
                )
                for row in warmup_batch
            ]
            warmup_responses = llm.generate(
                warmup_requests, sampling_params=sampling, use_tqdm=False
            )
            if len(warmup_responses) != len(warmup_batch):
                raise RuntimeError("vLLM warmup response count mismatch")
            warmup_seconds += time.perf_counter() - warmup_started
            warmup_rows += len(warmup_batch)

    started = time.perf_counter()
    new = 0
    errors = 0
    generated_tokens = 0
    request_preparation_seconds = 0.0
    generation_seconds = 0.0
    postprocess_seconds = 0.0
    batch_receipts: list[dict[str, object]] = []
    with output.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending_rows), args.batch_size):
            batch_rows = pending_rows[offset : offset + args.batch_size]
            batch_started = time.perf_counter()
            preparation_elapsed = 0.0
            generation_elapsed = 0.0
            postprocess_elapsed = 0.0
            batch_generated_tokens = 0
            batch_errors = 0
            try:
                preparation_started = time.perf_counter()
                requests = [
                    make_request(
                        processor,
                        row,
                        sampling_rate,
                        prompt_mode=args.prompt_mode,
                        audio_intervention=args.audio_intervention,
                    )
                    for row in batch_rows
                ]
                preparation_elapsed = time.perf_counter() - preparation_started
                request_preparation_seconds += preparation_elapsed
                generation_started = time.perf_counter()
                responses = llm.generate(requests, sampling_params=sampling, use_tqdm=False)
                generation_elapsed = time.perf_counter() - generation_started
                generation_seconds += generation_elapsed
                if len(responses) != len(batch_rows):
                    raise RuntimeError("vLLM response count mismatch")
                postprocess_started = time.perf_counter()
                items = []
                for row, response in zip(batch_rows, responses):
                    if len(response.outputs) != 1:
                        raise RuntimeError("vLLM returned a non-single generation")
                    choice = response.outputs[0]
                    raw = choice.text
                    choices = [str(value) for value in row["choices"]]
                    parsed_answer = answer_content_from_text(raw, choices)
                    strict_parsed_answer = strict_answer_content_from_text(raw, choices)
                    parsed_indices = (
                        matching_option_indices(parsed_answer, choices)
                        if parsed_answer is not None
                        else []
                    )
                    strict_parsed_indices = (
                        matching_option_indices(strict_parsed_answer, choices)
                        if strict_parsed_answer is not None
                        else []
                    )
                    gold_index = (
                        None if args.answers_hidden else int(row["gold_index"])
                    )
                    item = {
                        "id": str(row["id"]),
                        "gold_index": gold_index,
                        "gold_letter": (
                            None
                            if gold_index is None
                            else str(
                                row.get("gold_letter")
                                or BENCHMARK_LETTERS[gold_index]
                            ).upper()
                        ),
                        "gold_answer": (
                            None if gold_index is None else choices[gold_index]
                        ),
                        "prompt_sha256": eval_prompt_sha256,
                        "prompt_mode": args.prompt_mode,
                        "audio_intervention": args.audio_intervention,
                        "shard_index": args.shard_index,
                        "raw_output": raw,
                        "parsed_index": parsed_indices[0] if len(parsed_indices) == 1 else None,
                        "parsed_indices": parsed_indices,
                        "parsed_answer": parsed_answer,
                        "parseable": parsed_answer is not None,
                        "strict_parsed_index": (
                            strict_parsed_indices[0]
                            if len(strict_parsed_indices) == 1
                            else None
                        ),
                        "strict_parsed_indices": strict_parsed_indices,
                        "strict_parsed_answer": strict_parsed_answer,
                        "strict_parseable": strict_parsed_answer is not None,
                        "ambiguous_option_text": len(parsed_indices) > 1,
                        "correct": (
                            None
                            if gold_index is None
                            else (
                                parsed_answer is not None
                                and normalize_option_text(parsed_answer)
                                == normalize_option_text(choices[gold_index])
                            )
                        ),
                        "generated_tokens": len(choice.token_ids),
                        "error": None,
                        "backend": "vllm",
                    }
                    generated_tokens += len(choice.token_ids)
                    batch_generated_tokens += len(choice.token_ids)
                    items.append(item)
                postprocess_elapsed = time.perf_counter() - postprocess_started
                postprocess_seconds += postprocess_elapsed
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:
                items = []
                for row in batch_rows:
                    gold_index = (
                        None if args.answers_hidden else int(row["gold_index"])
                    )
                    items.append(
                        {
                            "id": str(row["id"]),
                            "gold_index": gold_index,
                            "gold_letter": (
                                None
                                if gold_index is None
                                else str(
                                    row.get("gold_letter")
                                    or BENCHMARK_LETTERS[gold_index]
                                ).upper()
                            ),
                            "gold_answer": (
                                None
                                if gold_index is None
                                else list(row["choices"])[gold_index]
                            ),
                            "prompt_sha256": eval_prompt_sha256,
                            "prompt_mode": args.prompt_mode,
                            "audio_intervention": args.audio_intervention,
                            "shard_index": args.shard_index,
                            "raw_output": "",
                            "parsed_index": None,
                            "parsed_indices": [],
                            "parsed_answer": None,
                            "parseable": False,
                            "strict_parsed_index": None,
                            "strict_parsed_indices": [],
                            "strict_parsed_answer": None,
                            "strict_parseable": False,
                            "ambiguous_option_text": False,
                            "correct": None if args.answers_hidden else False,
                            "generated_tokens": 0,
                            "error": f"{type(exc).__name__}: {exc}",
                            "backend": "vllm",
                        }
                    )
                errors += len(batch_rows)
                batch_errors = len(batch_rows)
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                new += 1
            handle.flush()
            if new and new % 25 == 0:
                print(
                    f"vllm eval shard={args.shard_index}/{args.num_shards} new={new} "
                    f"batch_size={args.batch_size} "
                    f"rate={new / max(time.perf_counter() - started, 1e-6):.3f}/s "
                    f"errors={errors}",
                    flush=True,
                )
            batch_elapsed = time.perf_counter() - batch_started
            batch_receipts.append(
                {
                    "batch_index": len(batch_receipts),
                    "offset": offset,
                    "requested_rows": len(batch_rows),
                    "successful_rows": len(batch_rows) - batch_errors,
                    "error_rows": batch_errors,
                    "generated_tokens": batch_generated_tokens,
                    "request_preparation_seconds": preparation_elapsed,
                    "generation_seconds": generation_elapsed,
                    "postprocess_seconds": postprocess_elapsed,
                    "elapsed_seconds": batch_elapsed,
                    "samples_per_second": len(batch_rows) / max(batch_elapsed, 1e-9),
                    "generation_samples_per_second": (
                        (len(batch_rows) - batch_errors) / max(generation_elapsed, 1e-9)
                        if generation_elapsed > 0
                        else 0.0
                    ),
                    "generated_tokens_per_second": (
                        batch_generated_tokens / max(generation_elapsed, 1e-9)
                        if generation_elapsed > 0
                        else 0.0
                    ),
                }
            )
    elapsed = time.perf_counter() - started
    seen = completed_ids(output)
    eligible = {str(row["id"]) for row in rows}
    successful_rows = new - errors
    total_elapsed = time.perf_counter() - process_started
    gpu["memory"] = peak_memory_receipt(torch, gpu_index)
    throughput = {
        "eligible_rows": len(eligible),
        "already_completed_rows": len(done & eligible),
        "pending_rows": len(pending_rows),
        "attempted_rows": new,
        "successful_rows": successful_rows,
        "error_rows": errors,
        "batch_count": len(batch_receipts),
        "configured_batch_size": args.batch_size,
        "warmup_batches_excluded": args.warmup_batches,
        "warmup_rows_excluded": warmup_rows,
        "warmup_seconds_excluded": warmup_seconds,
        "request_preparation_seconds": request_preparation_seconds,
        "generation_seconds": generation_seconds,
        "postprocess_seconds": postprocess_seconds,
        "evaluation_seconds": elapsed,
        "engine_load_seconds": engine_load_seconds,
        "process_seconds": total_elapsed,
        "samples_per_second": new / max(elapsed, 1e-9),
        "successful_samples_per_second": successful_rows / max(elapsed, 1e-9),
        "generation_samples_per_second": (
            successful_rows / max(generation_seconds, 1e-9)
            if generation_seconds > 0
            else 0.0
        ),
        "generated_tokens_per_second": (
            generated_tokens / max(generation_seconds, 1e-9)
            if generation_seconds > 0
            else 0.0
        ),
        "end_to_end_samples_per_second": new / max(total_elapsed, 1e-9),
        "batch_receipts": batch_receipts,
    }
    audit = {
        "complete": seen == eligible and errors == 0,
        "eligible": len(eligible),
        "completed": len(seen),
        "errors_this_attempt": errors,
        "backend": "vllm",
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "runtime": runtime,
        "merged_model_dir": str(merged_model_dir),
        "materialization_manifest": str(
            merged_model_dir / "vllm_materialization.json"
        ),
        "launch_contract": str(launch_path),
        "checkpoint_parameterization": parameterization,
        "adapter_dir": str(Path(args.adapter_dir).resolve()),
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "prompt_sha256": eval_prompt_sha256,
        "prompt_mode": args.prompt_mode,
        "audio_intervention": args.audio_intervention,
        "audio_intervention_definition": (
            "float32 all-zero waveform preserving the decoded matched-audio sample count"
            if args.audio_intervention == "silence_same_duration"
            else "unaltered matched audio"
        ),
        "answers_hidden": args.answers_hidden,
        "training_prompt_sha256": contract.prompt_sha256,
        "generation_contract": generation_contract,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_mode": args.shard_mode,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "enforce_eager": args.enforce_eager,
        "warmup_batches": args.warmup_batches,
        "warmup_rows": warmup_rows,
        "warmup_seconds": warmup_seconds,
        "warmup_jsonl": (
            str(Path(args.warmup_jsonl).resolve()) if args.warmup_jsonl else None
        ),
        "model_vocab_size": model_vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "logit_bias_vocab_size": logit_bias_vocab_size,
        "source_suppressed_token_count": len(source_forbidden_token_ids),
        "suppressed_token_count": len(forbidden_token_ids),
        "out_of_vocab_suppressed_token_count": len(
            out_of_vocab_forbidden_token_ids
        ),
        "out_of_vocab_suppressed_token_ids": out_of_vocab_forbidden_token_ids,
        "gpu": gpu,
        "gpu_name": gpu.get("name"),
        "gpu_uuid": gpu.get("uuid"),
        "peak_allocated_bytes": gpu["memory"].get("peak_allocated_bytes"),
        "peak_reserved_bytes": gpu["memory"].get("peak_reserved_bytes"),
        "new_rows": new,
        "generated_tokens": generated_tokens,
        "engine_load_seconds": engine_load_seconds,
        "elapsed_seconds": elapsed,
        "samples_per_second": throughput["samples_per_second"],
        "throughput": throughput,
        "finished_at": time.time(),
    }
    temporary = audit_path.with_name(f".{audit_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(audit_path)
    if not audit["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
