#!/usr/bin/env python3
"""Read-only release, runtime, device, and asset checks."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment.common import ROOT, runtime, verify_prepared, verify_release, verify_runtime


def hardware(*, require_idle=True):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("Expose exactly four GPUs to each training job")
    devices = []
    for i in range(4):
        p = torch.cuda.get_device_properties(i)
        # Thor's RTX 6000 Ada cards expose 44.39 GiB with ECC enabled.  The
        # campaign-specific smoke remains authoritative and still requires
        # at least 2 GiB reserved-memory headroom after two real updates.
        if p.total_memory < 44 * 1024**3 or p.major < 8:
            raise RuntimeError("Each GPU needs >=44 GiB VRAM and Ampere-or-newer BF16 support")
        uuid = str(p.uuid)
        if not uuid.startswith("GPU-"):
            uuid = "GPU-" + uuid
        used = int(subprocess.check_output(["nvidia-smi", "-i", uuid,
            "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
        if require_idle and used > 2048:
            raise RuntimeError(f"GPU {uuid} is busy ({used} MiB used); no job was launched")
        devices.append(dict(logical_index=i, uuid=uuid, name=p.name,
                            total_memory=p.total_memory, memory_used_mib=used,
                            compute_capability=[p.major, p.minor]))
    if len({x["name"] for x in devices}) != 1:
        raise RuntimeError("All four GPUs in a job must match")
    return dict(devices=devices, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                torch_cuda=torch.version.cuda,
                driver=subprocess.check_output(["nvidia-smi", "-i", devices[0]["uuid"],
                    "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--assets", action="store_true")
    p.add_argument("--hardware-only", action="store_true")
    a = p.parse_args()
    if a.hardware_only:
        print(json.dumps(hardware()))
        return
    value = dict(passed=True, release_sha256=verify_release(), runtime=verify_runtime(),
                 free_disk_gib=shutil.disk_usage(ROOT).free / 1024**3)
    if a.assets:
        value["prepared"] = verify_prepared()["complete"]
    if not a.cpu_only:
        value["hardware"] = hardware()
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
