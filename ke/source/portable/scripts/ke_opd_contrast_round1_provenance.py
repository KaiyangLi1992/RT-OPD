#!/usr/bin/env python3
"""Content-bound data and runtime provenance for contrast Round 1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

from ke_opd_v2.contract import sha256_file, sha256_json


PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
AUDIO_AUTHORITY_CONTRACT: Final = "v1_0_contrast_round1_audio_hash_authority_v1"
AUDIO_RECEIPT_CONTRACT: Final = "v1_0_contrast_round1_current_audio_bytes_v1"
RUNTIME_RECEIPT_CONTRACT: Final = "v1_0_contrast_round1_training_runtime_v1"
FORMAL_ROW_COUNT: Final = 10_000
FORMAL_REFERENCE_COUNT: Final = 20_000
FORMAL_UNIQUE_AUDIO_COUNT: Final = 10_000
RUNTIME_PACKAGES: Final = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
    "tokenizers",
    "numpy",
    "librosa",
    "soundfile",
    "soxr",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _duplicate_free_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json_bytes(raw: bytes, *, source: Path) -> object:
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_duplicate_free_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed JSON: {source}") from error


def _load_jsonl_bytes(raw: bytes, *, source: Path) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"JSONL is not UTF-8: {source}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_duplicate_free_object)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed JSONL row {line_number}: {source}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {source}")
        rows.append(value)
    return rows


def file_stat_guard(path: Path) -> dict[str, int]:
    supplied = path.absolute()
    try:
        value = supplied.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(supplied) from None
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise ValueError(f"provenance input is linked or non-regular: {supplied}")
    if supplied.resolve(strict=True) != supplied:
        raise ValueError(f"provenance input traverses a symlink: {supplied}")
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def stable_file_identity(path: Path) -> dict[str, object]:
    supplied = path.absolute()
    before = file_stat_guard(supplied)
    digest = sha256_file(supplied)
    after = file_stat_guard(supplied)
    if before != after:
        raise RuntimeError(f"file changed while hashing: {supplied}")
    return {
        "path": str(supplied),
        "size_bytes": after["size_bytes"],
        "sha256": digest,
    }


def stable_file_bytes(path: Path) -> tuple[bytes, dict[str, object]]:
    supplied = path.absolute()
    before = file_stat_guard(supplied)
    raw = supplied.read_bytes()
    after = file_stat_guard(supplied)
    if before != after or len(raw) != after["size_bytes"]:
        raise RuntimeError(f"file changed while reading: {supplied}")
    return raw, {
        "path": str(supplied),
        "size_bytes": after["size_bytes"],
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def receipt_file_identity(path: Path) -> dict[str, object]:
    return stable_file_identity(path)


def _validate_identity_current(identity: object, *, full_sha256: bool) -> Path:
    if not isinstance(identity, dict):
        raise ValueError("file identity is not an object")
    path = Path(str(identity.get("path", "")))
    if not path.is_absolute():
        raise ValueError("file identity path is not absolute")
    guard = file_stat_guard(path)
    if (
        int(identity.get("size_bytes", -1)) != guard["size_bytes"]
        or not isinstance(identity.get("sha256"), str)
    ):
        raise ValueError(f"file identity changed: {path}")
    if full_sha256 and sha256_file(path) != identity["sha256"]:
        raise ValueError(f"file SHA-256 changed: {path}")
    return path


def provenance_code_identity(project: Path) -> dict[str, str]:
    paths = {
        "ke_opd_contrast_round1_provenance.py": Path(__file__).resolve(),
        "run_ke_opd_contrast_round1.py": project
        / "source/portable/scripts/run_ke_opd_contrast_round1.py",
        "train_ke_opd_v1_hparam.py": project
        / "source/portable/scripts/train_ke_opd_v1_hparam.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _source_hash_mapping(rows: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row_index, row in enumerate(rows):
        for path_key, hash_key in (
            ("audio_path", "audio_sha256"),
            ("wrong_audio_path", "wrong_audio_sha256"),
        ):
            path = row.get(path_key)
            digest = row.get(hash_key)
            if (
                not isinstance(path, str)
                or not path
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"invalid source authority row {row_index}: {path_key}/{hash_key}"
                )
            previous = mapping.setdefault(path, digest)
            if previous != digest:
                raise ValueError(f"conflicting frozen hashes for audio path: {path}")
    return mapping


def _ordered_train_pairs(
    rows: list[dict[str, Any]], mapping: dict[str, str]
) -> tuple[list[dict[str, object]], set[str], set[str]]:
    pairs: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    real_paths: set[str] = set()
    wrong_paths: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = row.get("id")
        real_path = row.get("audio_path")
        wrong_path = row.get("wrong_audio_path")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
            raise ValueError(f"duplicate or invalid training ID at row {index}")
        if not isinstance(real_path, str) or not isinstance(wrong_path, str):
            raise ValueError(f"training row has invalid audio paths: {sample_id}")
        if real_path == wrong_path:
            raise ValueError(f"training row reuses Real bytes as Wrong: {sample_id}")
        if real_path in real_paths or wrong_path in wrong_paths:
            raise ValueError(f"duplicate Real/Wrong role path at row {index}")
        if real_path not in mapping or wrong_path not in mapping:
            raise ValueError(f"training audio path missing from V1 authority: {sample_id}")
        if mapping[real_path] == mapping[wrong_path]:
            raise ValueError(f"training Real/Wrong hashes collide: {sample_id}")
        seen_ids.add(sample_id)
        real_paths.add(real_path)
        wrong_paths.add(wrong_path)
        pairs.append(
            {
                "index": index,
                "id": sample_id,
                "real_path": real_path,
                "real_sha256": mapping[real_path],
                "wrong_path": wrong_path,
                "wrong_sha256": mapping[wrong_path],
            }
        )
    return pairs, real_paths, wrong_paths


def build_v1_audio_hash_authority(
    *,
    train_path: Path,
    source_path: Path,
    expected_rows: int = FORMAL_ROW_COUNT,
    expected_unique_audio: int = FORMAL_UNIQUE_AUDIO_COUNT,
) -> dict[str, object]:
    train_raw, train_identity = stable_file_bytes(train_path)
    source_raw, source_identity = stable_file_bytes(source_path)
    train_rows = _load_jsonl_bytes(train_raw, source=train_path)
    source_rows = _load_jsonl_bytes(source_raw, source=source_path)
    source_mapping = _source_hash_mapping(source_rows)
    pairs, real_paths, wrong_paths = _ordered_train_pairs(train_rows, source_mapping)
    if (
        len(train_rows) != expected_rows
        or len(pairs) != expected_rows
        or len(real_paths) != expected_unique_audio
        or len(wrong_paths) != expected_unique_audio
        or real_paths != wrong_paths
    ):
        raise ValueError("V1.0 authority extraction does not match the 10K pair contract")
    referenced = sorted(real_paths | wrong_paths)
    files = [
        {"path": path, "sha256": source_mapping[path]} for path in referenced
    ]
    return {
        "schema_version": 1,
        "complete": True,
        "contract": AUDIO_AUTHORITY_CONTRACT,
        "scope": "V1.0 train 10K ordered Real/Wrong references only",
        "builder_code_sha256": sha256_file(Path(__file__).resolve()),
        "source_hash_authority": source_identity,
        "train": train_identity,
        "counts": {
            "source_rows": len(source_rows),
            "source_unique_paths": len(source_mapping),
            "train_rows": len(train_rows),
            "ordered_references": len(pairs) * 2,
            "unique_referenced_paths": len(referenced),
        },
        "ordered_row_pair_sha256": sha256_json(pairs),
        "files_sha256": sha256_json(files),
        "files": files,
    }


def write_or_validate_v1_audio_hash_authority(
    output: Path,
    payload: dict[str, object],
    *,
    allow_replace: bool = False,
) -> dict[str, object]:
    if os.path.lexists(output):
        raw, _ = stable_file_bytes(output)
        existing = _load_json_bytes(raw, source=output)
        if existing != payload:
            if not allow_replace:
                raise ValueError(f"existing V1.0 audio authority is stale: {output}")
            atomic_json(output, payload)
    else:
        atomic_json(output, payload)
    return receipt_file_identity(output)


def validate_v1_audio_hash_authority(
    authority_path: Path,
    *,
    train_path: Path,
    expected_rows: int = FORMAL_ROW_COUNT,
    expected_unique_audio: int = FORMAL_UNIQUE_AUDIO_COUNT,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, object]]]:
    authority_raw, authority_identity = stable_file_bytes(authority_path)
    payload = _load_json_bytes(authority_raw, source=authority_path)
    if not isinstance(payload, dict):
        raise ValueError("V1.0 audio authority is not an object")
    train_raw, train_identity = stable_file_bytes(train_path)
    train_rows = _load_jsonl_bytes(train_raw, source=train_path)
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("V1.0 audio authority has no files list")
    mapping: dict[str, str] = {}
    canonical_files: list[dict[str, str]] = []
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError("malformed V1.0 audio authority file record")
        path = record.get("path")
        digest = record.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or path in mapping
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("duplicate or malformed V1.0 authority mapping")
        mapping[path] = digest
        canonical_files.append({"path": path, "sha256": digest})
    pairs, real_paths, wrong_paths = _ordered_train_pairs(train_rows, mapping)
    source_identity = payload.get("source_hash_authority")
    counts = payload.get("counts")
    if not isinstance(source_identity, dict):
        raise ValueError("V1.0 audio authority source identity is malformed")
    source_path = Path(str(source_identity.get("path", "")))
    if not source_path.is_absolute():
        raise ValueError("V1.0 audio authority source path is not absolute")
    expected_payload = build_v1_audio_hash_authority(
        train_path=train_path,
        source_path=source_path,
        expected_rows=expected_rows,
        expected_unique_audio=expected_unique_audio,
    )
    if payload != expected_payload:
        raise ValueError("V1.0 audio authority differs from its frozen source mapping")
    if (
        payload.get("schema_version") != 1
        or payload.get("complete") is not True
        or payload.get("contract") != AUDIO_AUTHORITY_CONTRACT
        or payload.get("scope")
        != "V1.0 train 10K ordered Real/Wrong references only"
        or payload.get("train") != train_identity
        or payload.get("builder_code_sha256")
        != sha256_file(Path(__file__).resolve())
        or not isinstance(counts, dict)
        or len(train_rows) != expected_rows
        or len(pairs) * 2 != expected_rows * 2
        or len(real_paths) != expected_unique_audio
        or len(wrong_paths) != expected_unique_audio
        or real_paths != wrong_paths
        or set(mapping) != real_paths | wrong_paths
        or canonical_files != sorted(canonical_files, key=lambda item: item["path"])
        or payload.get("ordered_row_pair_sha256") != sha256_json(pairs)
        or payload.get("files_sha256") != sha256_json(canonical_files)
        or counts.get("train_rows") != expected_rows
        or counts.get("ordered_references") != expected_rows * 2
        or counts.get("unique_referenced_paths") != expected_unique_audio
    ):
        raise ValueError("V1.0 audio authority contract is stale or malformed")
    payload["identity"] = authority_identity
    return payload, mapping, pairs


def _validate_pair_audit(
    pair_audit_path: Path,
    *,
    train_identity: dict[str, object],
    source_identity: object,
) -> dict[str, object]:
    raw, identity = stable_file_bytes(pair_audit_path)
    payload = _load_json_bytes(raw, source=pair_audit_path)
    if not isinstance(payload, dict) or not isinstance(source_identity, dict):
        raise ValueError("contrast pair audit/source identity is malformed")
    inputs = payload.get("inputs")
    hard_gates = payload.get("hard_gates")
    if (
        payload.get("complete") is not True
        or payload.get("pass") is not True
        or not isinstance(inputs, dict)
        or inputs.get("train", {}).get("path") != train_identity["path"]
        or inputs.get("train", {}).get("sha256") != train_identity["sha256"]
        or inputs.get("hash_authority", {}).get("path")
        != source_identity.get("path")
        or inputs.get("hash_authority", {}).get("sha256")
        != source_identity.get("sha256")
        or not isinstance(hard_gates, dict)
        or not hard_gates
        or any(value is not True for value in hard_gates.values())
    ):
        raise ValueError("contrast pair audit is stale or not closed")
    return identity


def _current_audio_guard(path_text: str) -> tuple[Path, dict[str, int]]:
    path = Path(path_text)
    if not path.is_absolute() or str(path.absolute()) != path_text:
        raise ValueError(f"training audio path is not canonical absolute: {path_text}")
    guard = file_stat_guard(path)
    return path, guard


def build_current_audio_receipt(
    *,
    project: Path,
    train_path: Path,
    authority_path: Path,
    pair_audit_path: Path,
    expected_rows: int = FORMAL_ROW_COUNT,
    expected_unique_audio: int = FORMAL_UNIQUE_AUDIO_COUNT,
) -> dict[str, object]:
    authority, mapping, pairs = validate_v1_audio_hash_authority(
        authority_path,
        train_path=train_path,
        expected_rows=expected_rows,
        expected_unique_audio=expected_unique_audio,
    )
    train_identity = authority["train"]
    pair_audit_identity = _validate_pair_audit(
        pair_audit_path,
        train_identity=train_identity,
        source_identity=authority["source_hash_authority"],
    )
    paths = sorted(mapping)
    before: dict[str, dict[str, int]] = {}
    inode_owner: dict[tuple[int, int], str] = {}
    for path_text in paths:
        _, guard = _current_audio_guard(path_text)
        inode = (guard["device"], guard["inode"])
        previous = inode_owner.setdefault(inode, path_text)
        if previous != path_text:
            raise ValueError(
                f"distinct training paths alias one audio inode: {previous}, {path_text}"
            )
        before[path_text] = guard

    records: list[dict[str, object]] = []
    digest_owner: dict[str, str] = {}
    for path_text in paths:
        digest = sha256_file(path_text)
        if digest != mapping[path_text]:
            raise ValueError(f"current training audio SHA-256 mismatch: {path_text}")
        previous = digest_owner.setdefault(digest, path_text)
        if previous != path_text:
            raise ValueError(
                f"distinct training paths have duplicate audio bytes: {previous}, {path_text}"
            )
        records.append(
            {
                "path": path_text,
                "size_bytes": before[path_text]["size_bytes"],
                "sha256": digest,
                "stat_guard": before[path_text],
            }
        )
    after = {path_text: file_stat_guard(Path(path_text)) for path_text in paths}
    if before != after:
        changed = [path for path in paths if before[path] != after[path]]
        raise RuntimeError(f"training audio changed during full hashing: {changed[:8]}")

    return {
        "schema_version": 1,
        "complete": True,
        "contract": AUDIO_RECEIPT_CONTRACT,
        "campaign": "ke_opd_v1_contrast_round1",
        "inputs": {
            "train": train_identity,
            "audio_hash_authority": authority["identity"],
            "authority_source_provenance": authority["source_hash_authority"],
            "pair_audit": pair_audit_identity,
        },
        "code_sha256": provenance_code_identity(project),
        "counts": {
            "train_rows": expected_rows,
            "ordered_references": expected_rows * 2,
            "unique_audio_files": expected_unique_audio,
        },
        "ordered_row_pair_sha256": sha256_json(pairs),
        "files_sha256": sha256_json(records),
        "files": records,
        "created_at": time.time(),
    }


def validate_current_audio_receipt(
    receipt_path: Path,
    *,
    project: Path,
    train_path: Path,
    authority_path: Path,
    pair_audit_path: Path,
    expected_rows: int = FORMAL_ROW_COUNT,
    expected_unique_audio: int = FORMAL_UNIQUE_AUDIO_COUNT,
) -> dict[str, object]:
    raw, receipt_identity = stable_file_bytes(receipt_path)
    receipt = _load_json_bytes(raw, source=receipt_path)
    if not isinstance(receipt, dict):
        raise ValueError("current-audio receipt is not an object")
    authority, mapping, pairs = validate_v1_audio_hash_authority(
        authority_path,
        train_path=train_path,
        expected_rows=expected_rows,
        expected_unique_audio=expected_unique_audio,
    )
    pair_audit_identity = _validate_pair_audit(
        pair_audit_path,
        train_identity=authority["train"],
        source_identity=authority["source_hash_authority"],
    )
    inputs = receipt.get("inputs")
    counts = receipt.get("counts")
    files = receipt.get("files")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("complete") is not True
        or receipt.get("contract") != AUDIO_RECEIPT_CONTRACT
        or receipt.get("campaign") != "ke_opd_v1_contrast_round1"
        or receipt.get("code_sha256") != provenance_code_identity(project)
        or not isinstance(inputs, dict)
        or inputs.get("train") != authority["train"]
        or inputs.get("audio_hash_authority") != authority["identity"]
        or inputs.get("authority_source_provenance")
        != authority["source_hash_authority"]
        or inputs.get("pair_audit") != pair_audit_identity
        or not isinstance(counts, dict)
        or counts.get("train_rows") != expected_rows
        or counts.get("ordered_references") != expected_rows * 2
        or counts.get("unique_audio_files") != expected_unique_audio
        or receipt.get("ordered_row_pair_sha256") != sha256_json(pairs)
        or not isinstance(files, list)
        or len(files) != expected_unique_audio
    ):
        raise ValueError("current-audio receipt contract is stale or malformed")

    canonical_records: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    seen_inodes: dict[tuple[int, int], str] = {}
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("malformed current-audio file record")
        path_text = record.get("path")
        digest = record.get("sha256")
        guard = record.get("stat_guard")
        if (
            not isinstance(path_text, str)
            or path_text in seen_paths
            or path_text not in mapping
            or digest != mapping[path_text]
            or not isinstance(guard, dict)
        ):
            raise ValueError("current-audio file mapping is stale or duplicate")
        _, current_guard = _current_audio_guard(path_text)
        if (
            current_guard != guard
            or int(record.get("size_bytes", -1)) != current_guard["size_bytes"]
        ):
            raise ValueError(f"current training audio stat changed: {path_text}")
        inode = (current_guard["device"], current_guard["inode"])
        previous = seen_inodes.setdefault(inode, path_text)
        if previous != path_text:
            raise ValueError("distinct current audio paths now alias one inode")
        seen_paths.add(path_text)
        canonical_records.append(
            {
                "path": path_text,
                "size_bytes": current_guard["size_bytes"],
                "sha256": digest,
                "stat_guard": current_guard,
            }
        )
    if (
        seen_paths != set(mapping)
        or canonical_records
        != sorted(canonical_records, key=lambda record: str(record["path"]))
        or receipt.get("files_sha256") != sha256_json(canonical_records)
    ):
        raise ValueError("current-audio file set/order/digest is stale")
    return {"payload": receipt, "identity": receipt_identity}


def ensure_current_audio_receipt(
    receipt_path: Path,
    *,
    project: Path,
    train_path: Path,
    authority_path: Path,
    pair_audit_path: Path,
    expected_rows: int = FORMAL_ROW_COUNT,
    expected_unique_audio: int = FORMAL_UNIQUE_AUDIO_COUNT,
) -> dict[str, object]:
    if not os.path.lexists(receipt_path):
        payload = build_current_audio_receipt(
            project=project,
            train_path=train_path,
            authority_path=authority_path,
            pair_audit_path=pair_audit_path,
            expected_rows=expected_rows,
            expected_unique_audio=expected_unique_audio,
        )
        atomic_json(receipt_path, payload)
    return validate_current_audio_receipt(
        receipt_path,
        project=project,
        train_path=train_path,
        authority_path=authority_path,
        pair_audit_path=pair_audit_path,
        expected_rows=expected_rows,
        expected_unique_audio=expected_unique_audio,
    )


def _normalize_gpu_uuid(value: object) -> str:
    text = str(value).strip().lower()
    return text[4:] if text.startswith("gpu-") else text


def local_python_cuda_identity() -> dict[str, object]:
    import torch

    packages = {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
    gpus: list[dict[str, object]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        uuid = getattr(properties, "uuid", None)
        if uuid is None:
            raise RuntimeError(f"torch exposes no UUID for logical GPU {index}")
        gpus.append(
            {
                "logical_index": index,
                "name": str(properties.name),
                "uuid": _normalize_gpu_uuid(uuid),
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    uname = os.uname()
    return {
        "host": {
            "nodename": uname.nodename,
            "sysname": uname.sysname,
            "release": uname.release,
            "machine": uname.machine,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "packages": packages,
        "torch": {
            "version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpus": gpus,
    }


def _physical_gpu_inventory(gpu_ids: tuple[int, ...]) -> list[dict[str, object]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    inventory: dict[int, dict[str, object]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise RuntimeError(f"unexpected nvidia-smi static GPU row: {line}")
        index = int(parts[0])
        inventory[index] = {
            "physical_index": index,
            "name": parts[1],
            "uuid": _normalize_gpu_uuid(parts[2]),
            "memory_total_mib": int(parts[3]),
            "driver_version": parts[4],
        }
    if set(gpu_ids) - set(inventory):
        raise RuntimeError("one or more formal physical GPUs are absent")
    return [inventory[index] for index in gpu_ids]


def capture_training_runtime_identity(
    *, project: Path, python: Path, gpu_ids: tuple[int, ...]
) -> dict[str, object]:
    visible = ",".join(map(str, gpu_ids))
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": visible,
            "PYTHONPATH": str(project / "source/portable"),
        }
    )
    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import json; "
                "from scripts.ke_opd_contrast_round1_provenance import "
                "local_python_cuda_identity; "
                "print(json.dumps(local_python_cuda_identity(), sort_keys=True))"
            ),
        ],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    lines = [line for line in probe.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("training runtime probe emitted no JSON")
    logical = json.loads(lines[-1])
    physical = _physical_gpu_inventory(gpu_ids)
    logical_gpus = logical.get("logical_gpus") if isinstance(logical, dict) else None
    if (
        not isinstance(logical_gpus, list)
        or len(logical_gpus) != len(gpu_ids)
        or logical.get("cuda_visible_devices") != visible
    ):
        raise ValueError("training runtime does not expose exactly eight requested GPUs")
    merged_gpus: list[dict[str, object]] = []
    for logical_index, (logical_gpu, physical_gpu) in enumerate(
        zip(logical_gpus, physical)
    ):
        if (
            not isinstance(logical_gpu, dict)
            or logical_gpu.get("logical_index") != logical_index
            or logical_gpu.get("name") != physical_gpu["name"]
            or _normalize_gpu_uuid(logical_gpu.get("uuid"))
            != physical_gpu["uuid"]
        ):
            raise ValueError("logical CUDA GPU order/UUID differs from physical topology")
        merged_gpus.append({**physical_gpu, **logical_gpu})
    return {
        "schema_version": 1,
        "host": logical["host"],
        "python": logical["python"],
        "packages": logical["packages"],
        "torch": logical["torch"],
        "cuda_visible_devices": visible,
        "physical_gpu_ids": list(gpu_ids),
        "gpus": merged_gpus,
    }


def validate_runtime_preflight_binding(
    runtime: dict[str, object], selected_gpus: object
) -> None:
    if not isinstance(selected_gpus, list):
        raise ValueError("GPU preflight has no selected inventory")
    runtime_gpus = runtime.get("gpus")
    if not isinstance(runtime_gpus, list) or len(runtime_gpus) != len(selected_gpus):
        raise ValueError("runtime/preflight GPU counts differ")
    for runtime_gpu, preflight_gpu in zip(runtime_gpus, selected_gpus):
        if not isinstance(runtime_gpu, dict) or not isinstance(preflight_gpu, dict):
            raise ValueError("runtime/preflight GPU record is malformed")
        if (
            runtime_gpu.get("physical_index") != preflight_gpu.get("index")
            or runtime_gpu.get("name") != preflight_gpu.get("name")
            or _normalize_gpu_uuid(runtime_gpu.get("uuid"))
            != _normalize_gpu_uuid(preflight_gpu.get("uuid"))
            or runtime_gpu.get("memory_total_mib")
            != preflight_gpu.get("memory_total_mib")
        ):
            raise ValueError("runtime GPU UUID/topology differs from controller preflight")


def ensure_training_runtime_receipt(
    receipt_path: Path,
    *,
    project: Path,
    python: Path,
    gpu_ids: tuple[int, ...],
) -> dict[str, object]:
    current = capture_training_runtime_identity(
        project=project, python=python, gpu_ids=gpu_ids
    )
    if not os.path.lexists(receipt_path):
        atomic_json(
            receipt_path,
            {
                "schema_version": 1,
                "complete": True,
                "contract": RUNTIME_RECEIPT_CONTRACT,
                "campaign": "ke_opd_v1_contrast_round1",
                "runtime": current,
                "code_sha256": provenance_code_identity(project),
                "created_at": time.time(),
            },
        )
    return validate_training_runtime_receipt(
        receipt_path,
        project=project,
        python=python,
        gpu_ids=gpu_ids,
        current_runtime=current,
    )


def validate_training_runtime_receipt(
    receipt_path: Path,
    *,
    project: Path,
    python: Path,
    gpu_ids: tuple[int, ...],
    current_runtime: dict[str, object] | None = None,
) -> dict[str, object]:
    raw, identity = stable_file_bytes(receipt_path)
    receipt = _load_json_bytes(raw, source=receipt_path)
    if not isinstance(receipt, dict):
        raise ValueError("training runtime receipt is not an object")
    current = current_runtime or capture_training_runtime_identity(
        project=project, python=python, gpu_ids=gpu_ids
    )
    if (
        receipt.get("schema_version") != 1
        or receipt.get("complete") is not True
        or receipt.get("contract") != RUNTIME_RECEIPT_CONTRACT
        or receipt.get("campaign") != "ke_opd_v1_contrast_round1"
        or receipt.get("runtime") != current
        or receipt.get("code_sha256") != provenance_code_identity(project)
    ):
        raise ValueError("training runtime receipt is stale or malformed")
    return {"payload": receipt, "identity": identity}


def load_bound_receipt(
    path: Path, *, expected_sha256: str, expected_contract: str
) -> dict[str, object]:
    raw, identity = stable_file_bytes(path)
    payload = _load_json_bytes(raw, source=path)
    if (
        identity.get("sha256") != expected_sha256
        or not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("complete") is not True
        or payload.get("contract") != expected_contract
        or payload.get("campaign") != "ke_opd_v1_contrast_round1"
    ):
        raise ValueError(f"campaign provenance receipt is stale: {path}")
    return {"identity": identity, "payload": payload}


def validate_local_rank_runtime_binding(
    binding: dict[str, object], *, local_rank: int, world_size: int
) -> None:
    payload = binding.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("runtime"), dict):
        raise ValueError("launch runtime binding is malformed")
    frozen = payload["runtime"]
    current = local_python_cuda_identity()
    frozen_gpus = frozen.get("gpus")
    current_gpus = current.get("logical_gpus")
    if (
        world_size != 8
        or not 0 <= local_rank < world_size
        or frozen.get("host") != current.get("host")
        or frozen.get("python") != current.get("python")
        or frozen.get("packages") != current.get("packages")
        or frozen.get("torch") != current.get("torch")
        or frozen.get("cuda_visible_devices")
        != current.get("cuda_visible_devices")
        or not isinstance(frozen_gpus, list)
        or not isinstance(current_gpus, list)
        or len(frozen_gpus) != world_size
        or len(current_gpus) != world_size
    ):
        raise ValueError("local rank runtime differs from the frozen controller runtime")
    for frozen_gpu, current_gpu in zip(frozen_gpus, current_gpus):
        if not isinstance(frozen_gpu, dict) or not isinstance(current_gpu, dict):
            raise ValueError("local/frozen GPU record is malformed")
        if any(
            frozen_gpu.get(key) != current_gpu.get(key)
            for key in ("logical_index", "name", "uuid", "total_memory_bytes")
        ):
            raise ValueError("local rank GPU UUID/topology differs from frozen runtime")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        default=PROJECT_ROOT / "data/runtime/v1_0/assets/audiomcq_aa_10k.jsonl",
    )
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument(
        "--source-authority",
        default=PROJECT_ROOT / "data/frozen/v1_2/broad_aa_mix_20k_v12.jsonl",
    )
    parser.add_argument(
        "--output",
        default=(
            PROJECT_ROOT
            / "data/frozen/v1_0/contrast_round1_audio_hash_authority.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_v1_audio_hash_authority(
        train_path=Path(args.train).absolute(),
        source_path=Path(args.source_authority).absolute(),
    )
    identity = write_or_validate_v1_audio_hash_authority(
        Path(args.output).absolute(),
        payload,
        allow_replace=bool(args.replace_existing),
    )
    print(json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
