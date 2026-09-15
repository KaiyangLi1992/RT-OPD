#!/usr/bin/env python3
"""Canonical-prompt, shardable evaluation for a V2 LoRA checkpoint."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from ke_opd_v2.contract import BENCHMARK_LETTERS, V2Contract, benchmark_user_text
from ke_opd_v2.modeling import (
    answer_content_from_text,
    chat_text,
    load_audio,
    load_processor,
    load_thinker,
    matching_option_indices,
    normalize_option_text,
    strict_answer_content_from_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATHS = {
    "Q": PROJECT_ROOT / "models/Qwen--Qwen2.5-Omni-3B",
    "K": PROJECT_ROOT / "models/KE-Team--Ke-Omni-R-3B",
}


class CompleteAnswer(StoppingCriteria):
    """Stop each batch member independently after its complete answer tag."""

    def __init__(self, tokenizer, prompt_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens
        # Transformers 4.52.4 only replaces tokens from already-finished batch
        # members with ``pad_token_id`` when at least one stopping criterion has
        # this attribute.  Our actual stop decision remains the complete tag
        # below; the marker prevents a finished row from continuing while a
        # longer peer is still decoding.
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return torch.tensor(
            [
                "</answer>"
                in self.tokenizer.decode(
                    row[self.prompt_tokens :],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                for row in input_ids
            ],
            dtype=torch.bool,
            device=input_ids.device,
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
    """Select a shard while optionally preserving the canonical batch groups."""
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid shard count/index")
    if mode == "stable_hash":
        return [
            row
            for row in rows
            if stable_shard(str(row["id"]), count) == index
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
    parser.add_argument("--initialization", choices=("Q", "K"), required=True)
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="engineering-only evenly spaced subset; zero evaluates the full shard",
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


def prepare_batch_inputs(
    processor,
    rows: list[dict],
    sampling_rate: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Serialize and pad a variable-length audio/text batch for generation."""
    texts = []
    waveforms = []
    for row in rows:
        choices = [str(choice) for choice in row["choices"]]
        user_text = row.get("canonical_user_text") or benchmark_user_text(
            str(row["question"]), choices
        )
        # ``apply_chat_template`` in the pinned 4.52.4 Omni processor returns
        # a one-element list for this single conversation.  Flatten exactly
        # that list so the outer list remains the processor's batch dimension.
        rendered = chat_text(processor, user_text)
        if not isinstance(rendered, list) or len(rendered) != 1:
            raise ValueError("canonical chat serialization did not return one prompt")
        texts.append(rendered[0])
        waveforms.append(load_audio(row["audio_path"], sampling_rate))
    inputs = processor(
        text=texts,
        audio=waveforms,
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


def trim_generated_padding(token_ids: torch.Tensor, pad_token_id: int | None) -> torch.Tensor:
    """Remove only padding appended after an independently stopped row."""
    if pad_token_id is None:
        return token_ids
    end = int(token_ids.numel())
    while end > 0 and int(token_ids[end - 1].item()) == int(pad_token_id):
        end -= 1
    return token_ids[:end]


def main() -> None:
    from peft import PeftModel

    args = parse_args()
    if args.batch_size < 1 or args.max_samples < 0:
        raise ValueError("--batch-size must be positive and --max-samples nonnegative")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid --num-shards/--shard-index")
    contract = V2Contract()
    model_path = MODEL_PATHS[args.initialization]
    if args.adapter_dir != "NONE":
        launch_path = Path(args.adapter_dir).resolve().parent / "launch_contract.json"
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        if launch.get("prompt_sha256") != contract.prompt_sha256:
            raise ValueError(f"checkpoint was not trained under the active prompt: {launch_path}")
    # Multiple independent eval shards often cold-load the same NFS checkpoint
    # on one node. Serialize only that load/copy phase to avoid page-cache I/O
    # collapse; inference remains fully parallel after the lock is released.
    load_lock_path = Path(f"/tmp/kaiyli_ke_opd_v2_eval_load_{args.initialization}.lock")
    with load_lock_path.open("w") as load_lock:
        fcntl.flock(load_lock, fcntl.LOCK_EX)
        processor = load_processor(model_path)
        base = load_thinker(model_path)
        if args.adapter_dir == "NONE":
            model = base.to("cuda").eval()
        else:
            model = PeftModel.from_pretrained(
                base, args.adapter_dir, is_trainable=False
            ).to("cuda").eval()
        fcntl.flock(load_lock, fcntl.LOCK_UN)
    vocab = json.loads(Path(args.vocab_json).read_text(encoding="utf-8"))
    model_vocab = int(base.config.text_config.vocab_size)
    forbidden = sorted(set(range(model_vocab)) - set(map(int, vocab["valid_ids"])))
    sampling_rate = int(processor.feature_extractor.sampling_rate)
    output = Path(args.output_jsonl)
    audit_path = Path(args.audit_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(output)
    rows = [json.loads(line) for line in Path(args.input_jsonl).open(encoding="utf-8") if line.strip()]
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
    started = time.time()
    new = 0
    errors = 0
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    with output.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending_rows), args.batch_size):
            batch_rows = pending_rows[offset : offset + args.batch_size]
            try:
                inputs = prepare_batch_inputs(processor, batch_rows, sampling_rate, device)
                prompt_length = int(inputs["input_ids"].shape[1])
                with torch.inference_mode():
                    sequences = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=contract.max_completion_tokens,
                        use_cache=True,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        suppress_tokens=forbidden,
                        stopping_criteria=StoppingCriteriaList(
                            [CompleteAnswer(processor.tokenizer, prompt_length)]
                        ),
                    )
                items = []
                for index, row in enumerate(batch_rows):
                    item = {
                        "id": str(row["id"]),
                        "gold_index": int(row["gold_index"]),
                        "gold_letter": str(
                            row.get("gold_letter")
                            or BENCHMARK_LETTERS[int(row["gold_index"])]
                        ).upper(),
                        "prompt_sha256": contract.prompt_sha256,
                        "shard_index": args.shard_index,
                    }
                    choices = [str(choice) for choice in row["choices"]]
                    item["gold_answer"] = choices[item["gold_index"]]
                    generated = trim_generated_padding(
                        sequences[index, prompt_length:], processor.tokenizer.pad_token_id
                    )
                    raw = processor.tokenizer.decode(
                        generated,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
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
                    item.update(
                        {
                            "raw_output": raw,
                            "parsed_index": (
                                parsed_indices[0] if len(parsed_indices) == 1 else None
                            ),
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
                                parsed_answer is not None
                                and normalize_option_text(parsed_answer)
                                == normalize_option_text(item["gold_answer"])
                            ),
                            "generated_tokens": int(generated.numel()),
                            "error": None,
                            "backend": "transformers",
                        }
                    )
                    items.append(item)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise
            except Exception as exc:
                items = []
                for row in batch_rows:
                    items.append(
                        {
                            "id": str(row["id"]),
                            "gold_index": int(row["gold_index"]),
                            "gold_letter": str(
                                row.get("gold_letter")
                                or BENCHMARK_LETTERS[int(row["gold_index"])]
                            ).upper(),
                            "prompt_sha256": contract.prompt_sha256,
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
                            "correct": False,
                            "generated_tokens": 0,
                            "error": f"{type(exc).__name__}: {exc}",
                            "backend": "transformers",
                        }
                    )
                errors += len(batch_rows)
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                new += 1
            handle.flush()
            if new % 25 == 0:
                print(
                    f"eval shard={args.shard_index}/{args.num_shards} new={new} "
                    f"batch_size={args.batch_size} "
                    f"rate={new / max(time.time() - started, 1e-6):.3f}/s errors={errors} "
                    f"peak_allocated_mib={torch.cuda.max_memory_allocated(device) / 2**20:.0f} "
                    f"peak_reserved_mib={torch.cuda.max_memory_reserved(device) / 2**20:.0f}",
                    flush=True,
                )
    seen = completed_ids(output)
    eligible = {str(row["id"]) for row in rows}
    audit = {
        "complete": seen == eligible and errors == 0,
        "eligible": len(eligible),
        "completed": len(seen),
        "errors_this_attempt": errors,
        "backend": "transformers",
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "initialization": args.initialization,
        "adapter_dir": None if args.adapter_dir == "NONE" else str(Path(args.adapter_dir).resolve()),
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "prompt_sha256": contract.prompt_sha256,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_mode": args.shard_mode,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "new_rows": new,
        "elapsed_seconds": time.time() - started,
        "samples_per_second": new / max(time.time() - started, 1e-6),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    tmp = audit_path.with_name(f".{audit_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    tmp.replace(audit_path)
    if not audit["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
