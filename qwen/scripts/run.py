#!/usr/bin/env python3
"""Three independent four-GPU lanes, or one scheduler-allocated four-GPU job."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment.common import ROOT, check, config, identity, read, rows, sha, verify_release, verify_runtime, write


def tasks():
    return config()["tasks"]


def sidecar(output, suffix):
    return output.with_name(output.name + "." + suffix)


def environment(gpus):
    env = os.environ.copy()
    if gpus != "auto":
        ids = gpus.split(",")
        if len(ids) != 4 or len(set(ids)) != 4:
            raise ValueError("Each job must expose four distinct GPUs")
        env["CUDA_VISIBLE_DEVICES"] = gpus
    elif len(env.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 4:
        raise ValueError("--gpus auto requires four GPUs in CUDA_VISIBLE_DEVICES from the scheduler")
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    env.update(PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    # A6000 defaults use NVIDIA's transport selection; host-specific Ada settings
    # are never baked in. Explicit NCCL settings from the allocation are recorded.
    return env


def close_training(output, task, *, smoke, binding):
    complete = read(output / "TRAIN_COMPLETE.json")
    c = config()
    stop = 2 if smoke else c["arms"][task["arm"]]["stop_step"]
    cp = output / f"checkpoint-{stop}"
    manifest = read(cp / "checkpoint_manifest.json")
    if not manifest["complete"] or manifest["step"] != stop:
        raise ValueError("Checkpoint not complete")
    for name, rec in manifest["files"].items():
        check(dict(rec, path=str(cp / name)))
    audits = []
    initial = set()
    all_ids = []
    ordered = []
    for rank in range(4):
        audit = read(output / f"RANK_AUDIT_{rank}.json")
        if audit["start"] != 0 or audit["stop"] != manifest["step"] or audit["arm"] != task["arm"]:
            raise ValueError("Run audit disagrees with task")
        if smoke and audit["memory_headroom_bytes"] < 2 * 1024**3:
            raise RuntimeError("Smoke finished but left less than 2 GiB VRAM headroom")
        audits.append(audit)
        initial.add(read(output / f"START_RANK_{rank}.json")["initial_adapter_sha256"])
        for row in rows(output / f"rows-rank-{rank}.jsonl"):
            ordered.append((row["step"], row["slot"], rank, row["ids"]))
    for _, _, _, ids in sorted(ordered):
        all_ids.extend(ids)
    if True:
        from experiment.grid import order_paths
        visits = stop * 32 - (stop // 313) * 16
        expected = [item for p in order_paths(task["seed"]) for item in read(p)["ids"]][:visits]
    else:
        expected = read(ROOT / f"data/frozen/order_s{task['seed']}.json")["ids"][:64 if smoke else 10000]
    if all_ids != expected or len(initial) != 1:
        raise ValueError("Row coverage or common initialization verification failed")
    payload = dict(complete=True, task=task, engineering_only=smoke, start_step=0,
                   stop_step=manifest["step"], binding=binding, rows=len(all_ids),
                   initial_adapter_sha256=initial.pop(), checkpoint=str(cp),
                   checkpoint_manifest=identity(cp / "checkpoint_manifest.json"),
                   training_complete=identity(output / "TRAIN_COMPLETE.json"),
                   audits={str(r): identity(output / f"RANK_AUDIT_{r}.json") for r in range(4)},
                   launch=identity(sidecar(output, "LAUNCH.json")))
    payload["checkpoints"] = {}
    for step in [stop]:
        checkpoint = output / f"checkpoint-{step}"
        record = read(checkpoint / "checkpoint_manifest.json")
        if not record["complete"] or record["step"] != step:
            raise ValueError("Missing retained checkpoint")
        for name, rec in record["files"].items():
            check(dict(rec, path=str(checkpoint / name)))
        payload["checkpoints"][str(step)] = identity(checkpoint / "checkpoint_manifest.json")
    write(output / "VERIFIED.json", payload)
    return payload


def verify_completed(output, task, binding):
    value = read(output / "VERIFIED.json")
    if value["task"] != task or value["binding"] != binding or not value["complete"]:
        raise ValueError(f"Existing receipt belongs to different inputs: {output}")
    check(value["checkpoint_manifest"])
    cp = Path(value["checkpoint"])
    for name, rec in read(value["checkpoint_manifest"]["path"])["files"].items():
        check(dict(rec, path=str(cp / name)))
    for rec in [value["training_complete"], value["launch"], *value["audits"].values()]:
        check(rec)
    for step, record in value.get("checkpoints", {}).items():
        check(record)
        for name, rec in read(record["path"])["files"].items():
            check(dict(rec, path=str(Path(record["path"]).parent / name)))
    return value


def execute(task, gpus, *, smoke_only=False, attempt="attempt1"):
    env = environment(gpus)
    release = verify_release()
    runtime = verify_runtime()
    # Query in a child so the controller retains no CUDA contexts.
    hardware = json.loads(subprocess.check_output(
        [sys.executable, str(ROOT / "scripts/preflight.py"), "--hardware-only"], env=env, text=True))
    uuids = [d["uuid"] for d in hardware["devices"]]
    devices = [{k: v for k, v in d.items() if k != "memory_used_mib"} for d in hardware["devices"]]
    binding = dict(release_sha256=release, runtime=runtime, devices=devices,
                   driver=hardware["driver"], host=socket.gethostname(),
                   nccl={k: v for k, v in env.items() if k.startswith("NCCL_")})
    lane_id = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()[:16]
    root = ROOT / "runs"
    root.mkdir(exist_ok=True)
    if shutil.disk_usage(root).free < 25 * 1024**3:
        raise RuntimeError("Less than 25 GiB free before a training job")
    locks = []
    try:
        directory = Path("/tmp") / f"aa-opd-gpu-locks-{os.getuid()}"
        directory.mkdir(exist_ok=True)
        for uuid in sorted(uuids):
            lock = (directory / (uuid + ".lock")).open("a")
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Recheck after taking the device claims, immediately before launch.
        subprocess.run([sys.executable, str(ROOT / "scripts/preflight.py"), "--hardware-only"],
                       env=env, stdout=subprocess.DEVNULL, check=True)
        for smoke in ([True] if smoke_only else [True, False]):
            unit = dict(task)
            if smoke:
                unit.update(seed=config()["seeds"][0], task_id=f"smoke-{task['arm']}", index=-1, lane=-1)
                output = root / "qualification" / lane_id / task["arm"] / attempt
            else:
                output = root / "training" / task["task_id"] / attempt
            if (output / "VERIFIED.json").exists():
                verify_completed(output, unit, binding)
                print(f"Verified existing completion: {output}", flush=True)
                continue
            if output.exists():
                raise FileExistsError(f"Incomplete attempt preserved at {output}; inspect logs and use a new --attempt after fixing the cause")
            output.mkdir(parents=True, exist_ok=False)
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                       "--nproc_per_node=4", "--max_restarts=0", "--module", "experiment.train",
                       "--arm", task["arm"], "--seed", str(unit["seed"]), "--output", str(output)]
            if smoke:
                command += ["--smoke"]
            write(sidecar(output, "LAUNCH.json"), dict(task=unit, argv=command, binding=binding,
                 environment={k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONPATH")},
                 at=time.time(), parent_checkpoint=None, engineering_only=smoke))
            log_path = sidecar(output, "train.log")
            print(f"Starting {'smoke' if smoke else 'formal'} {unit['task_id']} on {gpus}; log={log_path}", flush=True)
            with log_path.open("x") as log:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try:
                    code = proc.wait()
                except BaseException:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait()
                    raise
            if code:
                write(output / "FAILED.json", dict(returncode=code, at=time.time(), log=str(log_path)))
                raise RuntimeError(f"Training failed; see {log_path}")
            close_training(output, unit, smoke=smoke, binding=binding)
            print(f"Completed {unit['task_id']}", flush=True)
    finally:
        for lock in reversed(locks):
            lock.close()

