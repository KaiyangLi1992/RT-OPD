#!/usr/bin/env python3
"""Explicit K checkpoint materialization, independent vLLM shards, and closure.

No submission API is called. MMAU-full exports a hidden-label submission;
official scores are deliberately separate from validation selection.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ["OPD_CAMPAIGN"] = "ke_grid16"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment.common import ROOT, check, config, identity, model_config, read, rows, sha, verify_prepared, verify_release, write
from experiment.grid import STEPS
from scripts.evaluate import score


@contextmanager
def gpu_claim(gpu):
    query = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if not query.startswith("GPU-") or "\n" in query or "/" in query:
        raise ValueError("Expected exactly one GPU UUID")
    directory = Path("/tmp") / f"aa-opd-gpu-locks-{os.getuid()}"
    directory.mkdir(exist_ok=True)
    with (directory / (query + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        used = int(subprocess.check_output(["nvidia-smi", "-i", query,
            "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
        if used > 2048:
            raise RuntimeError("Evaluation GPU is busy; no process was started")
        yield


def shard_rows(gold, index, count):
    if not 0 <= index < count:
        raise ValueError("Invalid shard index")
    return [r for i, r in enumerate(gold) if (i // 8) % count == index]


def checkpoint(run, step):
    release = verify_release()
    if step not in STEPS:
        raise ValueError("Checkpoint is outside the frozen grid")
    cp = run / f"checkpoint-{step}"
    manifest = read(cp / "checkpoint_manifest.json")
    launch = run / "launch_contract.json"
    if not manifest["complete"] or manifest["step"] != step or manifest["world_size"] != 4:
        raise ValueError("Incomplete or foreign checkpoint")
    if manifest["launch_contract_sha256"] != sha(launch):
        raise ValueError("Checkpoint launch identity mismatch")
    immutable = read(launch)["immutable"]
    if (immutable["experiment_contract"] != config()["campaign_id"]
        or immutable["release_sha256"] != release or immutable["engineering_only"]
        or immutable["start_step"] != 0
        or immutable["stop_step"] != config()["arms"][immutable["reported_arm"]]["stop_step"]
        or step != immutable["stop_step"]
        or immutable["models"]["student"]["files"] != model_config()["student"]["files"]):
        raise ValueError("Not a formal K-grid checkpoint")
    for name, rec in manifest["files"].items():
        check(dict(rec, path=str(cp / name)))
    cell = config()["arms"][immutable["reported_arm"]]
    if immutable["target_alpha"] != cell["alpha"] or immutable["target_arm"] != cell["target_arm"]:
        raise ValueError("Target configuration drift")
    if immutable["run"]["learning_rate"] != cell["learning_rate"] or immutable["seed"] not in config()["seeds"]:
        raise ValueError("Learning rate or seed drift")
    return dict(manifest=identity(cp / "checkpoint_manifest.json"), launch=identity(launch),
                config_id=immutable["reported_arm"], step=step, release_sha256=release)


def verify_inference_runtime():
    expected = read(ROOT / "environment/evaluation_runtime.json")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("The frozen vLLM evaluator requires Python 3.12")
    for name in ("torch", "transformers", "numpy", "librosa", "soundfile", "safetensors", "soxr", "tokenizers"):
        if importlib.metadata.version(name) != expected[name]:
            raise ValueError(f"Frozen inference runtime mismatch: {name}")
    if importlib.metadata.version("vllm") != expected["vllm_distribution"]:
        raise ValueError("Use the frozen vLLM CUDA-12.9 distribution")


def validate_part(gold, predictions, audit, benchmark, index, count):
    from scripts.grid16_inference import evaluation_prompt_sha256
    full = benchmark == "mmau_full"
    mode = "ke_mmau_v051525" if full else "v2_contract"
    expected = shard_rows(gold, index, count)
    if [r["id"] for r in expected] != [r["id"] for r in predictions]:
        raise ValueError("Missing, duplicate, reordered or foreign predictions")
    required = dict(complete=True, backend="vllm", errors_this_attempt=0, eligible=len(expected),
                    completed=len(expected), num_shards=count, shard_index=index,
                    batch_size=8, shard_mode="batch_stride", dtype="float16", enforce_eager=True,
                    max_model_len=4096, gpu_memory_utilization=.9, prompt_mode=mode,
                    audio_intervention="matched", prompt_sha256=evaluation_prompt_sha256(mode))
    if any(audit.get(k) != v for k, v in required.items()):
        raise ValueError("Incomplete or drifted inference audit")
    runtime = read(ROOT / "environment/evaluation_runtime.json")
    for name, value in runtime.items():
        if audit["runtime"].get(name) != value:
            raise ValueError(f"Evaluation runtime drift: {name}")
    generation = (dict(do_sample=False, max_new_tokens=256, stop="model EOS only", vocabulary_suppression=False)
                  if full else dict(do_sample=False, max_new_tokens=256, stop=["</answer>"],
                                    include_stop_str_in_output=True, vocabulary_suppression=True))
    if audit["generation_contract"] != generation:
        raise ValueError("Generation contract drift")
    for g, p in zip(expected, predictions):
        if (p.get("error") is not None or p["backend"] != "vllm" or p["audio_intervention"] != "matched"
            or p["prompt_sha256"] != required["prompt_sha256"]
            or type(p["generated_tokens"]) is not int or not 0 <= p["generated_tokens"] <= 256):
            raise ValueError("Invalid prediction")
        if full:
            if any(k in g for k in ("answer", "gold_index")) or p.get("correct") is not None or p.get("gold_index") is not None:
                raise ValueError("Full test labels must remain hidden")
        else:
            s = score(p["raw_output"], g)
            if any(p[k] != v for k, v in s.items()) or p["gold_index"] != g["gold_index"]:
                raise ValueError("Parser/score drift")
    return predictions


def close(out, benchmark, gold, binding):
    count = config()["evaluation"]["shards"][benchmark]
    parts, artifacts = [], {}
    for i in range(count):
        part = out / benchmark / str(i)
        if read(part / "BINDING.json") != dict(binding, benchmark=benchmark, shard=i):
            raise ValueError("Mixed checkpoint or benchmark bindings")
        predictions = rows(part / "predictions.jsonl")
        validate_part(gold, predictions, read(part / "audit.json"), benchmark, i, count)
        parts.extend(predictions)
        for name in ("BINDING.json", "predictions.jsonl", "audit.json"):
            artifacts[f"{i}/{name}"] = identity(part / name)
    by_id = {p["id"]: p for p in parts}
    if len(by_id) != len(parts) or set(by_id) != {r["id"] for r in gold}:
        raise ValueError("Shard coverage does not close")
    ordered = [by_id[r["id"]] for r in gold]
    if benchmark == "mmau_full":
        def extracted(raw):
            match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
            return (match.group(1) if match else raw).strip()
        submission = [dict(id=str(g["source_id"]), model_prediction=extracted(p["raw_output"]))
            for g, p in zip(gold, ordered)]
        write(out / benchmark / "submission.json", submission)
        artifacts["submission"] = identity(out / benchmark / "submission.json")
        result = dict(inference_complete=True, score_available=False, count=len(gold),
                      reason="Hidden labels: official score required; never substitute Mini")
    else:
        correct = sum(p["correct"] for p in ordered)
        result = dict(inference_complete=True, score_available=True, count=len(gold),
                      correct=correct, accuracy=correct / len(gold))
    write(out / benchmark / "COMPLETE.json", dict(binding=binding, benchmark=benchmark,
          result=result, artifacts=artifacts))
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("mode", choices=["materialize", "shard", "close"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=STEPS, required=True)
    parser.add_argument("--benchmark", choices=config()["evaluation"]["datasets"])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    run = args.run.resolve()
    cp = run / f"checkpoint-{args.step}"
    binding = checkpoint(run, args.step)
    out = run / f"evaluation-{args.step}"
    out.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "source/portable") + os.pathsep + str(ROOT)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HF_HUB_OFFLINE"] = "1"
    if "," in args.gpu:
        parser.error("Evaluation uses exactly one GPU")
    merged = out / "merged"
    if args.mode == "materialize":
        from experiment.common import verify_runtime
        verify_runtime()  # materialize in the TRAINING environment, not vLLM's
        wrapper = read(ROOT / "configs/ke_grid16_assets.json")["wrapper"]
        check(dict(path=str(ROOT / "assets/grid16/wrapper/config.json"),
                   sha256=wrapper["sha256"], size_bytes=wrapper["size_bytes"]))
        for name, rec in model_config()["student"]["files"].items():
            check(dict(path=str(ROOT / "models" / model_config()["student"]["directory"] / name),
                       sha256=rec["sha256"], size_bytes=rec["size"]))
        write(out / "MODEL_BINDING.json", binding)
        subprocess.run([sys.executable, str(ROOT / "source/portable/scripts/materialize_vllm_thinker.py"),
            "--project", str(ROOT), "--initialization", config()["initialization_code"], "--adapter-dir", str(cp),
            "--wrapper-config", str(ROOT / "assets/grid16/wrapper/config.json"),
            "--output-dir", str(merged)], env=env, check=True)
        return
    if args.benchmark is None:
        parser.error("--benchmark is required")
    ready = verify_prepared("evaluation")
    benchmark = args.benchmark
    source = ready["artifacts"][benchmark]
    gold = rows(source["path"])
    if len(gold) != config()["evaluation"]["datasets"][benchmark]:
        raise ValueError("Benchmark count mismatch")
    binding = dict(binding, dataset=source, evaluation=config()["evaluation"])
    if args.mode == "close":
        print(json.dumps(close(out, benchmark, gold, binding), indent=2))
        return
    verify_inference_runtime()
    count = config()["evaluation"]["shards"][benchmark]
    shard_rows(gold, args.shard_index, count)
    if read(out / "MODEL_BINDING.json") != checkpoint(run, args.step):
        raise ValueError("Materialized model binding mismatch")
    materialized = read(merged / "vllm_materialization.json")
    if not materialized["complete"]:
        raise ValueError("Materialization is incomplete")
    for name, rec in materialized["output_files"].items():
        check(dict(path=str(merged / name), sha256=rec["sha256"], size_bytes=rec["size"]))
    part = out / benchmark / str(args.shard_index)
    part.mkdir(parents=True, exist_ok=True)
    with (part / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(part / "BINDING.json", dict(binding, benchmark=benchmark, shard=args.shard_index))
        if (part / "audit.json").exists():
            validate_part(gold, rows(part / "predictions.jsonl"), read(part / "audit.json"), benchmark, args.shard_index, count)
            print("Verified existing shard; no repeated inference")
            return
        # Short absolute socket directory avoids the historical Unix socket limit.
        with gpu_claim(args.gpu), tempfile.TemporaryDirectory(prefix="opd-", dir="/tmp") as tmp:
            env.update(TMPDIR=tmp, VLLM_RPC_BASE_PATH=tmp)
            command = [sys.executable, str(ROOT / "scripts/grid16_inference.py"),
                "--merged-model-dir", str(merged), "--adapter-dir", str(cp), "--input-jsonl", source["path"],
                "--output-jsonl", str(part / "predictions.jsonl"), "--audit-json", str(part / "audit.json"),
                "--vocab-json", str(ROOT / "data/frozen/canonical_valid_vocab_v2.json"),
                "--num-shards", str(count), "--shard-index", str(args.shard_index), "--shard-mode", "batch_stride",
                "--batch-size", "8", "--max-model-len", "4096", "--gpu-memory-utilization", "0.9",
                "--dtype", "float16", "--enforce-eager", "--audio-intervention", "matched",
                "--prompt-mode", "ke_mmau_v051525" if benchmark == "mmau_full" else "v2_contract"]
            if benchmark == "mmau_full":
                command.append("--answers-hidden")
            with (part / "inference.log").open("a") as log:
                subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        validate_part(gold, rows(part / "predictions.jsonl"), read(part / "audit.json"), benchmark, args.shard_index, count)


if __name__ == "__main__":
    main()
