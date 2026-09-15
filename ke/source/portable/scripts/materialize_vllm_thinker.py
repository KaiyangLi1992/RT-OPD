#!/usr/bin/env python3
"""Materialize a PEFT checkpoint as an atomic thinker-only vLLM model."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from ke_opd_v2.modeling import load_processor, load_thinker


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATHS = {
    "Q": Path("models/Qwen--Qwen2.5-Omni-3B"),
    "K": Path("models/KE-Team--Ke-Omni-R-3B"),
}
MODEL_AUXILIARY_FILES = (
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


def native_wrapper_valid(output_dir: Path) -> bool:
    try:
        config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return (
        config.get("model_type") == "qwen2_5_omni"
        and config.get("architectures") == ["Qwen2_5OmniForConditionalGeneration"]
        and isinstance(config.get("thinker_config"), dict)
        and config["thinker_config"].get("model_type")
        == "qwen2_5_omni_thinker"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ROOT)
    parser.add_argument("--initialization", choices=("Q", "K"), required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument(
        "--adapter-dir",
        help="legacy LoRA checkpoint entry point (kept for backward compatibility)",
    )
    checkpoint.add_argument(
        "--checkpoint-dir",
        help="LoRA adapter or full-fine-tuned thinker checkpoint",
    )
    parser.add_argument(
        "--checkpoint-kind",
        choices=("auto", "lora", "full_ft"),
        default="auto",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--wrapper-config", type=Path,
                        help="Pinned full-wrapper config directory; no Q student weights are needed")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialization_runtime_identity() -> dict[str, object]:
    import librosa
    import numpy
    import peft
    import safetensors
    import soundfile
    import soxr
    import tokenizers
    import transformers

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
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "peft": str(peft.__version__),
        "safetensors": str(safetensors.__version__),
        "tokenizers": str(tokenizers.__version__),
        "numpy": str(numpy.__version__),
        "librosa": str(librosa.__version__),
        "soundfile": str(soundfile.__version__),
        "soxr": str(soxr.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version()),
        "nvidia_driver_version": versions.pop(),
        "cuda": bool(torch.cuda.is_available()),
    }


def checkpoint_kind(checkpoint_dir: Path) -> str:
    lora = all(
        (checkpoint_dir / name).is_file()
        for name in ("adapter_config.json", "adapter_model.safetensors")
    )
    full_weights = sorted(
        path
        for path in checkpoint_dir.iterdir()
        if path.is_file()
        and path.name.endswith(".safetensors")
        and path.name != "adapter_model.safetensors"
    )
    full_ft = (checkpoint_dir / "config.json").is_file() and bool(full_weights)
    if lora and full_ft:
        raise ValueError(f"ambiguous LoRA/full-FT checkpoint: {checkpoint_dir}")
    if lora:
        return "lora"
    if full_ft:
        return "full_ft"
    raise ValueError(
        "checkpoint must contain adapter_config.json + adapter_model.safetensors "
        "or config.json + non-adapter safetensor weights: "
        f"{checkpoint_dir}"
    )


def launch_parameterization(checkpoint_dir: Path) -> tuple[str | None, Path | None]:
    launch_path = checkpoint_dir.parent / "launch_contract.json"
    try:
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    immutable = launch.get("immutable")
    candidates = [launch.get("parameterization")]
    if isinstance(immutable, dict):
        candidates.append(immutable.get("parameterization"))
        resolved = immutable.get("resolved")
        if isinstance(resolved, dict):
            candidates.append(resolved.get("parameterization"))
    declared = next((value for value in candidates if isinstance(value, str)), None)
    return declared, launch_path


def resolve_checkpoint(args: argparse.Namespace) -> tuple[Path, str]:
    checkpoint_dir = Path(args.checkpoint_dir or args.adapter_dir).resolve()
    detected = checkpoint_kind(checkpoint_dir)
    requested = str(args.checkpoint_kind)
    if requested != "auto" and requested != detected:
        raise ValueError(
            f"--checkpoint-kind={requested} does not match detected {detected}: "
            f"{checkpoint_dir}"
        )
    declared, launch_path = launch_parameterization(checkpoint_dir)
    if declared is not None and declared != detected:
        raise ValueError(
            f"launch contract declares parameterization={declared}, but checkpoint "
            f"files detect {detected}: {launch_path}"
        )
    return checkpoint_dir, detected


def source_identity(
    model_dir: Path,
    checkpoint_dir: Path,
    kind: str = "lora",
    wrapper_config_path: Path | None = None,
) -> dict[str, object]:
    model_files = sorted(
        path
        for path in model_dir.iterdir()
        if path.is_file()
        and (
            path.name == "config.json"
            or path.name == "model.safetensors.index.json"
            or path.name.endswith(".safetensors")
            or path.name in MODEL_AUXILIARY_FILES
        )
    )
    if kind == "lora":
        checkpoint_files = [
            checkpoint_dir / "adapter_config.json",
            checkpoint_dir / "adapter_model.safetensors",
        ]
    elif kind == "full_ft":
        checkpoint_files = sorted(
            path
            for path in checkpoint_dir.iterdir()
            if path.is_file()
            and (
                path.name == "config.json"
                or path.name == "model.safetensors.index.json"
                or (
                    path.name.endswith(".safetensors")
                    and path.name != "adapter_model.safetensors"
                )
            )
        )
        if not checkpoint_files:
            raise ValueError(f"full-FT checkpoint has no model files: {checkpoint_dir}")
    else:
        raise ValueError(f"unsupported checkpoint kind: {kind}")
    for path in [*model_files, *checkpoint_files]:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
    source: dict[str, object] = {
        "base_model_dir": str(model_dir),
        "base_files": {
            path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in model_files
        },
    }
    if kind == "lora":
        # Keep the original source schema so outputs created through the
        # legacy --adapter-dir entry point remain cache-compatible.
        source["adapter_dir"] = str(checkpoint_dir)
        source["adapter_files"] = {
            path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in checkpoint_files
        }
    else:
        source["full_ft_dir"] = str(checkpoint_dir)
        source["full_ft_files"] = {
            path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in checkpoint_files
        }
    declared, launch_path = launch_parameterization(checkpoint_dir)
    if declared is not None and launch_path is not None:
        source["launch_contract"] = {
            "path": str(launch_path),
            "parameterization": declared,
            "size": launch_path.stat().st_size,
            "sha256": sha256_file(launch_path),
        }
    if wrapper_config_path is not None:
        wrapper_config_path = wrapper_config_path.absolute()
        if wrapper_config_path.is_symlink() or not wrapper_config_path.is_file():
            raise ValueError(
                f"wrapper config is missing, linked, or unsafe: {wrapper_config_path}"
            )
        source["wrapper_config"] = {
            "path": str(wrapper_config_path.resolve()),
            "size": wrapper_config_path.stat().st_size,
            "sha256": sha256_file(wrapper_config_path),
        }
    return source


def manifest_matches(
    output_dir: Path,
    source: dict[str, object],
    runtime: dict[str, object] | None = None,
) -> bool:
    try:
        manifest = json.loads(
            (output_dir / "vllm_materialization.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if (
        manifest.get("schema_version") != 2
        or manifest.get("source") != source
        or not manifest.get("complete")
        or (runtime is not None and manifest.get("runtime") != runtime)
    ):
        return False
    output_files = manifest.get("output_files")
    if not isinstance(output_files, dict):
        return False
    for name, row in output_files.items():
        path = output_dir / str(name)
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size"])
            or sha256_file(path) != str(row["sha256"])
        ):
            return False
    return native_wrapper_valid(output_dir)


def fast_manifest_matches(
    output_dir: Path,
    model_dir: Path,
    checkpoint_dir: Path,
    kind: str = "lora",
    full_model_dir: Path | None = None,
    runtime: dict[str, object] | None = None,
) -> bool:
    """Validate an immutable cached model without re-hashing multi-GB weights.

    Full source and output hashes are recorded when the model is materialized.
    On reuse, paths, sizes, small-file hashes, and modification times provide a
    fail-closed cache check.  Any file touched after the manifest was written
    falls back to the full verification path in ``main``.
    """
    try:
        manifest = json.loads(
            (output_dir / "vllm_materialization.json").read_text(encoding="utf-8")
        )
        source = manifest["source"]
        finished_at = float(manifest["finished_at"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False
    if (
        manifest.get("schema_version") != 2
        or not manifest.get("complete")
        or source.get("base_model_dir") != str(model_dir)
        or (runtime is not None and manifest.get("runtime") != runtime)
    ):
        return False

    if kind == "lora":
        source_directory_key = "adapter_dir"
        source_files_key = "adapter_files"
    elif kind == "full_ft":
        source_directory_key = "full_ft_dir"
        source_files_key = "full_ft_files"
    else:
        return False
    if source.get(source_directory_key) != str(checkpoint_dir):
        return False
    if manifest.get("checkpoint_kind", "lora") != kind:
        return False
    if full_model_dir is not None:
        wrapper_path = full_model_dir / "config.json"
        wrapper_record = source.get("wrapper_config")
        if not isinstance(wrapper_record, dict):
            return False
        try:
            if (
                wrapper_path.is_symlink()
                or not wrapper_path.is_file()
                or wrapper_record.get("path") != str(wrapper_path.resolve())
                or int(wrapper_record.get("size", -1)) != wrapper_path.stat().st_size
                or wrapper_record.get("sha256") != sha256_file(wrapper_path)
                or manifest.get("wrapper_config_source") != wrapper_record
            ):
                return False
        except (TypeError, ValueError, OSError):
            return False
    launch_record = source.get("launch_contract")
    if launch_record is not None:
        if not isinstance(launch_record, dict):
            return False
        try:
            launch_path = Path(str(launch_record["path"]))
            if (
                not launch_path.is_file()
                or launch_path.stat().st_size != int(launch_record["size"])
                or sha256_file(launch_path) != str(launch_record["sha256"])
            ):
                return False
        except (KeyError, TypeError, ValueError, OSError):
            return False

    groups = (
        (model_dir, source.get("base_files"), False),
        (checkpoint_dir, source.get(source_files_key), kind == "lora"),
        (output_dir, manifest.get("output_files"), False),
    )
    for directory, records, hash_weights in groups:
        if not isinstance(records, dict) or not records:
            return False
        for name, record in records.items():
            if not isinstance(record, dict):
                return False
            path = directory / str(name)
            try:
                stat = path.stat()
                expected_size = int(record["size"])
                expected_sha = str(record["sha256"])
            except (FileNotFoundError, KeyError, TypeError, ValueError, OSError):
                return False
            if stat.st_size != expected_size or stat.st_mtime > finished_at:
                return False
            # Config/index/tokenizer files are cheap to hash.  Adapter weights
            # are also hashed because they identify the checkpoint uniquely;
            # the immutable base and materialized multi-GB shards use size and
            # modification-time checks on cache hits.
            if hash_weights or not path.name.endswith(".safetensors"):
                if sha256_file(path) != expected_sha:
                    return False
    return native_wrapper_valid(output_dir)


def wrap_as_native_vllm_omni(
    directory: Path, full_model_dir: Path, thinker_config: dict[str, object]
) -> None:
    """Use vLLM's full-Omni registry name while storing thinker-only weights."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("saved thinker is missing its safetensor weight map")
    shard_names = sorted(set(map(str, weight_map.values())))
    for shard_name in shard_names:
        shard_path = directory / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        tensors = load_file(shard_path, device="cpu")
        if any(name.startswith("thinker.") for name in tensors):
            raise ValueError("saved thinker unexpectedly already has wrapper prefixes")
        prefixed = {f"thinker.{name}": value for name, value in tensors.items()}
        replacement = shard_path.with_name(f".{shard_path.name}.prefixed.tmp")
        save_file(prefixed, replacement, metadata=metadata)
        replacement.replace(shard_path)
    index["weight_map"] = {
        f"thinker.{name}": shard for name, shard in weight_map.items()
    }
    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    full_config = json.loads(
        (full_model_dir / "config.json").read_text(encoding="utf-8")
    )
    full_config.update(
        {
            "architectures": ["Qwen2_5OmniForConditionalGeneration"],
            "enable_audio_output": False,
            "enable_talker": False,
            "thinker_config": thinker_config,
        }
    )
    (directory / "config.json").write_text(
        json.dumps(full_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    project = Path(args.project).resolve()
    checkpoint_dir, kind = resolve_checkpoint(args)
    output_dir = Path(args.output_dir).resolve()
    model_dir = (project / MODEL_PATHS[args.initialization]).resolve()
    full_model_dir = (project / MODEL_PATHS["Q"]).resolve()
    if args.wrapper_config is not None:
        full_model_dir = args.wrapper_config.resolve().parent
    runtime = materialization_runtime_identity()
    if output_dir in {model_dir, checkpoint_dir, full_model_dir}:
        raise ValueError("output directory must be distinct from source directories")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.materialize.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output_dir.exists() and fast_manifest_matches(
            output_dir,
            model_dir,
            checkpoint_dir,
            kind,
            full_model_dir=full_model_dir,
            runtime=runtime,
        ):
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "cached": True,
                        "verification": "immutable_fast_path",
                        "output_dir": str(output_dir),
                    }
                ),
                flush=True,
            )
            return
        source = source_identity(
            model_dir,
            checkpoint_dir,
            kind,
            wrapper_config_path=full_model_dir / "config.json",
        )
        if output_dir.exists():
            if manifest_matches(output_dir, source, runtime):
                print(
                    json.dumps(
                        {
                            "status": "ready",
                            "cached": True,
                            "output_dir": str(output_dir),
                        }
                    ),
                    flush=True,
                )
                return
            raise FileExistsError(
                f"refusing to overwrite incomplete or mismatched output: {output_dir}"
            )

        started = time.time()
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.tmp.", dir=str(output_dir.parent)
            )
        )
        processor = load_processor(model_dir)
        if kind == "lora":
            from peft import PeftModel

            base = load_thinker(model_dir, dtype=torch.bfloat16)
            peft_model = PeftModel.from_pretrained(
                base, checkpoint_dir, is_trainable=False
            )
            merged = peft_model.merge_and_unload(safe_merge=True, progressbar=True)
        else:
            merged = load_thinker(checkpoint_dir, dtype=torch.bfloat16)
        merged.config.architectures = [
            "Qwen2_5OmniThinkerForConditionalGeneration"
        ]
        merged.config.model_type = "qwen2_5_omni_thinker"
        thinker_config = merged.config.to_dict()
        merged.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        processor.save_pretrained(temporary)
        # Ke-Omni ships a thinker-only top-level config.  vLLM's native
        # multimodal registry expects the full Qwen2.5-Omni wrapper config,
        # with the merged Ke thinker nested underneath it.
        wrap_as_native_vllm_omni(temporary, full_model_dir, thinker_config)
        output_files = {}
        for path in sorted(temporary.iterdir()):
            if path.is_file() and path.name != "vllm_materialization.json":
                output_files[path.name] = {
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        manifest = {
            "schema_version": 2,
            "complete": True,
            "initialization": args.initialization,
            "checkpoint_kind": kind,
            "source": source,
            "vllm_architecture": "Qwen2_5OmniForConditionalGeneration",
            "stored_weight_prefix": "thinker.",
            "wrapper_config_source": source["wrapper_config"],
            "output_dir": str(output_dir),
            "output_files": output_files,
            "torch_version": torch.__version__,
            "runtime": runtime,
            "elapsed_seconds": time.time() - started,
            "finished_at": time.time(),
        }
        (temporary / "vllm_materialization.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
        print(
            json.dumps(
                {
                    "status": "ready",
                    "cached": False,
                    "output_dir": str(output_dir),
                    "elapsed_seconds": manifest["elapsed_seconds"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
