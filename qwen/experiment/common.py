"""Portable paths, immutable inputs, and small atomic receipts."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
PORTABLE = ROOT / "source/portable"
sys.path.insert(0, str(PORTABLE))


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p), size_bytes=p.stat().st_size)


def check(record):
    p = Path(record["path"])
    if not p.is_file() or sha(p) != record["sha256"]:
        raise ValueError(f"Missing or changed artifact: {p}")
    if "size_bytes" in record and p.stat().st_size != record["size_bytes"]:
        raise ValueError(f"Size mismatch: {p}")


def write(path, value):
    """Publish atomically; never replace an existing receipt with different content."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if p.exists():
        if p.read_text() != body:
            raise FileExistsError(f"Refusing to replace different content: {p}")
        return
    temporary = p.with_name(p.name + f".tmp-{os.getpid()}")
    temporary.write_text(body)
    try:
        os.link(temporary, p)
    finally:
        temporary.unlink(missing_ok=True)


def config():
    from experiment.grid import grid_config
    return grid_config()


def model_config():
    return read(ROOT / "configs/models_ke_grid16.json")


def runtime():
    packages = ("torch", "torchvision", "torchaudio", "transformers", "peft", "accelerate",
                "numpy", "librosa", "resampy", "soundfile", "safetensors", "qwen-omni-utils")
    return dict(python=platform.python_version(), executable=sys.executable,
                packages={p: importlib.metadata.version(p) for p in packages})


def verify_release():
    manifest = read(ROOT / "provenance/release_files.json")
    for name, digest in manifest.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Release file was changed: {name}; use a new experiment version for edits")
    return sha(ROOT / "provenance/release_files.json")


def verify_runtime():
    observed = runtime()
    expected = read(ROOT / "environment/runtime.json")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use Python 3.10")
    for name, version in expected["packages"].items():
        if observed["packages"][name] != version:
            raise RuntimeError(f"{name}: expected {version}, got {observed['packages'][name]}")
    return observed


def verify_prepared(scope="training"):
    from experiment.frozen_data import ensure_frozen_data
    ensure_frozen_data(download=False)
    receipt = read(ROOT / "assets" / f"{scope}_ready.json")
    if config().get("initialization_code") == "K" and receipt.get("campaign_id") != config()["campaign_id"]:
        raise ValueError("Assets were not prepared for the K grid profile")
    if receipt["release_sha256"] != verify_release():
        raise ValueError("Prepared assets belong to a different release; rerun preparation")
    for rec in receipt["artifacts"].values():
        check(rec)
    # Check file size/mtime at launch; full byte hashes were verified at preparation.
    # Any changed audio forces re-preparation and a fresh full hash verification.
    for rec in receipt["audio"]:
        p = Path(rec["path"])
        s = p.stat()
        if s.st_size != rec["size_bytes"] or s.st_mtime_ns != rec["mtime_ns"]:
            raise ValueError(f"Audio changed since verification: {p}")
    return receipt
