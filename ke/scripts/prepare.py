#!/usr/bin/env python3
"""Download pinned public assets, reconstruct audio, and verify exact byte hashes."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment.common import ROOT, check, config, model_config, identity, read, rows, sha, verify_release, write


def hub_file(repo_id, revision, name, directory, *, dataset=False):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo_id, revision=revision, filename=name,
                repo_type="dataset" if dataset else "model", local_dir=directory))


def extract(archive, destination, *, zstd=False):
    """Extract only regular files/directories into a dedicated asset directory."""
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with Path(archive).open("rb") as source:
        if zstd:
            import zstandard
            stream = zstandard.ZstdDecompressor().stream_reader(source)
        else:
            stream = source
        try:
            with tarfile.open(fileobj=stream, mode="r|" if zstd else "r|gz") as tf:
                for member in tf:
                    path = (destination / member.name).resolve()
                    if not path.is_relative_to(base) or not (member.isfile() or member.isdir()):
                        raise ValueError(f"Unsafe archive member: {member.name}")
                    if member.isdir():
                        path.mkdir(parents=True, exist_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with tf.extractfile(member) as inp, path.open("wb") as out:
                            shutil.copyfileobj(inp, out)
        finally:
            if zstd:
                stream.close()


def download_models():
    for role, model in model_config().items():
        directory = ROOT / "models" / model["directory"]
        for name, record in model["files"].items():
            path = directory / name
            if not path.is_file() or path.stat().st_size != record["size"] or sha(path) != record["sha256"]:
                path = hub_file(model["repo_id"], model["revision"], name, directory)
            check(dict(path=str(path), sha256=record["sha256"], size_bytes=record["size"]))
        print(f"Verified {role}: {model['repo_id']}@{model['revision']}", flush=True)


def download_training():
    source = read(ROOT / "configs/training_archive.json")
    directory = ROOT / "downloads/training"
    archive = source["archive"]
    repo = source["repository"]
    path = hub_file(repo["repo_id"], repo["revision"], archive["filename"], directory, dataset=True)
    check(dict(path=str(path), sha256=archive["sha256"], size_bytes=archive["bytes"]))
    unpacked = ROOT / "downloads/training/unpacked"
    extract(path, unpacked, zstd=True)
    candidates = list(unpacked.rglob("data.jsonl"))
    if len(candidates) != 1:
        raise ValueError("Archive must contain exactly one upstream data.jsonl")
    target = ROOT / "assets/audio/training"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == candidates[0].parent.resolve():
        return
    if target.exists():
        raise FileExistsError(target)
    target.symlink_to(candidates[0].parent.resolve(), target_is_directory=True)


def download_benchmarks():
    from huggingface_hub import snapshot_download
    import numpy as np
    import pyarrow.parquet as pq
    import resampy
    import soundfile as sf

    sources = {r["target"]: r for r in read(ROOT / "configs/benchmark_sources.json")}
    audio = ROOT / "assets/audio"
    mmau = sources["MMAU-test-mini"]
    parquet = hub_file(mmau["repo_id"], mmau["revision"], "test_mini.parquet",
                       ROOT / "downloads/mmau", dataset=True)
    (audio / "mmau").mkdir(parents=True, exist_ok=True)
    expected = {str(r["source_id"]) for r in rows(ROOT / "data/frozen/mmau.jsonl")}
    observed = set()
    pf = pq.ParquetFile(parquet)
    for group in range(pf.num_row_groups):
        for row in pf.read_row_group(group, columns=["context", "other_attributes"]).to_pylist():
            sid = str(json.loads(row["other_attributes"])["id"])
            if sid not in expected or sid in observed:
                raise ValueError(f"Unexpected/duplicate MMAU ID: {sid}")
            observed.add(sid)
            (audio / "mmau" / (sid + ".wav")).write_bytes(row["context"]["bytes"])
    if observed != expected:
        raise ValueError("MMAU ID coverage mismatch")

    mmar = sources["MMAR"]
    archive = hub_file(mmar["repo_id"], mmar["revision"], "mmar-audio.tar.gz",
                       ROOT / "downloads/mmar", dataset=True)
    unpacked = ROOT / "downloads/mmar/unpacked"
    extract(archive, unpacked)
    files = {}
    for path in unpacked.rglob("*.wav"):
        if path.name in files:
            raise ValueError(f"Duplicate MMAR basename: {path.name}")
        files[path.name] = path
    (audio / "mmar").mkdir(parents=True, exist_ok=True)
    for row in rows(ROOT / "data/frozen/mmar.jsonl"):
        name = Path(row["audio_path"]).name
        shutil.copyfile(files[name], audio / "mmar" / name)

    adqa = sources["DCASE2026-Task5-DevSet"]
    repo = ROOT / "downloads/adqa"
    snapshot_download(repo_id=adqa["repo_id"], revision=adqa["revision"], repo_type="dataset",
                      local_dir=repo, max_workers=4)
    (audio / "adqa").mkdir(parents=True, exist_ok=True)
    required = {str(r["source_id"]) for r in rows(ROOT / "data/frozen/adqa.jsonl")}
    for row in rows(repo / "dev.jsonl"):
        sid = str(row["id"])
        if sid not in required:
            continue
        waveform, rate = sf.read(repo / row["audio_path"], dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if rate != 32000:
            waveform = resampy.resample(waveform.astype("float32"), rate, 32000)
        sf.write(audio / "adqa" / (sid + ".wav"), np.asarray(waveform, dtype="float32"),
                 32000, subtype="PCM_16")


def prepare(scope):
    release = verify_release()
    names = ["train", "balanced"] if scope == "training" else ["balanced", "mmar", "mmau", "adqa"]
    if True:
        names = ["train"] if scope == "training" else list(config()["evaluation"]["datasets"])
    expected_audio = read(ROOT / "data/frozen/audio_files.json")
    if True and scope == "evaluation":
        from experiment.grid_assets import full_audio_identities
        expected_audio.update(full_audio_identities())
    audio_records = {}
    artifacts = {}
    for name in names:
        data = rows(ROOT / ("assets/grid16/mmau_full.jsonl" if name == "mmau_full" else f"data/frozen/{name}.jsonl"))
        for row in data:
            for key in ("audio_path", "wrong_audio_path"):
                if key not in row:
                    continue
                relative = row[key]
                path = ROOT / "assets/audio" / relative
                if relative not in audio_records:
                    check(dict(expected_audio[relative], path=str(path)))
                    stat = path.stat()
                    audio_records[relative] = dict(path=str(path.resolve()), **expected_audio[relative], mtime_ns=stat.st_mtime_ns)
                row[key] = str(path.resolve())
        path = ROOT / "assets/manifests" / (name + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
        if path.exists() and path.read_text() != body:
            raise ValueError(f"Different prepared manifest already exists: {path}")
        path.write_text(body)
        artifacts[name] = identity(path)
    if scope == "training":
        from ke_opd_v2.contract import V2Contract
        for name, file in [("teacher_gate", "teacher_gate.jsonl"), ("valid_vocab", "canonical_valid_vocab_v2.json")]:
            artifacts[name] = identity(ROOT / "data/frozen" / file)
        bundle = dict(status="frozen_complete", prompt_sha256=V2Contract().prompt_sha256,
                      assets={key: artifacts[key] for key in ("train", "teacher_gate", "valid_vocab")})
        bundle_path = ROOT / "assets/manifests/bundle.json"
        write(bundle_path, bundle)
        artifacts["bundle"] = identity(bundle_path)
        train = rows(artifacts["train"]["path"])
        by_id = {r["id"]: r for r in train}
        verified_hashes = {r["path"]: r["sha256"] for r in audio_records.values()}
        gates = dict(rows_10000=len(train) == len(by_id) == 10000,
                     donor_bijection=len({r["wrong_id"] for r in train}) == 10000,
                     donors_in_training=all(r["wrong_id"] in by_id for r in train),
                     distinct_audio=all(verified_hashes[r["audio_path"]] != verified_hashes[r["wrong_audio_path"]] for r in train),
                     donor_paths_agree=all(r["wrong_audio_path"] == by_id[r["wrong_id"]]["audio_path"] for r in train))
        if not all(gates.values()):
            raise ValueError(f"Donor pair validation failed: {gates}")
        pair_path = ROOT / "assets/manifests/pairs.json"
        write(pair_path, {"complete": True, "pass": True, "hard_gates": gates,
                          "inputs": {"train": artifacts["train"]}})
        artifacts["pair_audit"] = identity(pair_path)
        for role, model in model_config().items():
            for filename, record in model["files"].items():
                path = ROOT / "models" / model["directory"] / filename
                check(dict(path=str(path), sha256=record["sha256"], size_bytes=record["size"]))
                artifacts[f"model/{role}/{filename}"] = identity(path)
    receipt = dict(complete=True, scope=scope, release_sha256=release, campaign_id=config()["campaign_id"],
                   artifacts=artifacts, audio=list(audio_records.values()))
    if True and scope == "training":
        from experiment.grid import order_paths
        for seed in config()["seeds"]:
            for epoch, path in enumerate(order_paths(seed)):
                artifacts[f"order/{seed}/{epoch}"] = identity(path)
        artifacts["wrapper"] = identity(ROOT / "assets/grid16/wrapper/config.json")
    # A re-verification may refresh audio timestamps, but never touches a run receipt.
    path = ROOT / "assets" / (scope + "_ready.json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    print(f"Verified {scope}: {len(audio_records)} exact audio files; wrote {path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=["training", "evaluation", "all"], default="training")
    p.add_argument("--verify-only", action="store_true", help="Use existing assets/audio and models directories; no network")
    args = p.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    (ROOT / "assets").mkdir(exist_ok=True)
    with (ROOT / "assets/.prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_release()
        from experiment.frozen_data import ensure_frozen_data
        ensure_frozen_data(download=not args.verify_only)
        if args.scope in ("training", "all"):
            if True:
                from experiment.grid_assets import prepare_orders, prepare_wrapper
                prepare_orders()
                prepare_wrapper(download=not args.verify_only)
            if not args.verify_only:
                download_models()
                download_training()
            prepare("training")
        if args.scope in ("evaluation", "all"):
            if True:
                from experiment.grid_assets import prepare_full_manifest
                prepare_full_manifest(download=not args.verify_only)
            if not args.verify_only:
                download_benchmarks()
                if True:
                    from experiment.full_assets import download_full
                    download_full()
            prepare("evaluation")


if __name__ == "__main__":
    main()
