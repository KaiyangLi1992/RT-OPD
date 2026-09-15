#!/usr/bin/env python3
"""Explicit, once-only submission of a closed 9000-item MMAU-full endpoint."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

os.environ["OPD_CAMPAIGN"] = "ke_grid16"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment.common import check, identity, read, write
from scripts.grid16_eval import checkpoint

SPACE = "https://sonalkum-mmau-eval.hf.space"


def request(argv):
    # Never retry POST: ambiguous completion requires manual event recovery.
    return subprocess.check_output(["curl", "-fsS", "--connect-timeout", "30", "--max-time", "300", *argv], text=True)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--step", type=int, choices=[313, 432, 626], required=True)
    p.add_argument("--submit", action="store_true", help="Explicitly authorize uploading this endpoint's predictions")
    a = p.parse_args()
    run = a.run.resolve()
    base_binding = checkpoint(run, a.step)
    root = run / f"evaluation-{a.step}/mmau_full"
    complete = read(root / "COMPLETE.json")
    if any(complete["binding"][k] != v for k, v in base_binding.items()):
        raise ValueError("Foreign inference result")
    for r in complete["artifacts"].values():
        check(r)
    submission = root / "submission.json"
    data = read(submission)
    if len(data) != 9000 or len({r["id"] for r in data}) != 9000:
        raise ValueError("Expected exactly 9000 unique predictions")
    binding = dict(checkpoint=base_binding, inference=identity(root / "COMPLETE.json"), submission=identity(submission))
    with (root / ".official.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "OFFICIAL.json").exists():
            result = read(root / "OFFICIAL.json")
            if result["binding"] != binding:
                raise ValueError("Foreign official result")
            check(result["response"])
            print(json.dumps(result, indent=2))
            return
        event = root / "OFFICIAL_EVENT.json"
        intent = root / "OFFICIAL_INTENT.json"
        if not event.exists():
            if intent.exists():
                raise RuntimeError("Submission intent exists without event receipt. Recover the event manually; do not resubmit.")
            if not a.submit:
                p.error("Use --submit to authorize one external scoring submission")
            write(intent, binding)
            uploaded = json.loads(request(["-X", "POST", SPACE + "/gradio_api/upload",
                                          "-F", f"files=@{submission};type=application/json"]))
            if not isinstance(uploaded, list) or len(uploaded) != 1:
                raise ValueError("Unexpected upload response; inspect intent before recovery")
            payload = dict(data=[dict(path=uploaded[0], orig_name=submission.name, size=submission.stat().st_size,
                mime_type="application/json", is_stream=False, meta={"_type": "gradio.FileData"})])
            result = json.loads(request(["-H", "Content-Type: application/json", "--data-binary", json.dumps(payload),
                                         SPACE + "/gradio_api/call/predict"]))
            write(event, dict(event_id=result["event_id"], binding=binding))
        e = read(event)
        if e["binding"] != binding or read(intent) != binding or not re.fullmatch(r"[A-Za-z0-9_-]+", e["event_id"]):
            raise ValueError("Event binding mismatch")
        raw = request([SPACE + "/gradio_api/call/predict/" + e["event_id"]])
        lines = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
        returned = json.loads(lines[-1]) if lines else None
        report = str(returned[0] if isinstance(returned, list) else returned)
        match = re.search(r"Total Accuracy:\s*([0-9.]+)%\s*\((\d+)/(\d+)\)", report)
        if not match or int(match[3]) != 9000 or not 0 <= int(match[2]) <= 9000:
            raise ValueError("Official result is pending or invalid; retry the saved event only")
        correct = int(match[2])
        if not math.isclose(float(match[1]), 100 * correct / 9000, abs_tol=.011):
            raise ValueError("Official reported percentage disagrees with counts")
        write(root / "OFFICIAL_RESPONSE.json", dict(event_id=e["event_id"], response=raw, report=report))
        result = dict(binding=binding, score_available=True, count=9000, correct=correct,
                      accuracy=correct / 9000, response=identity(root / "OFFICIAL_RESPONSE.json"))
        write(root / "OFFICIAL.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
