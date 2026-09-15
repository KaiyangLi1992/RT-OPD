#!/usr/bin/env python3
"""Maintainer-only hash refresh after a reviewed code/config change; never launches."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    a = p.parse_args()
    names = []
    for directory in ("configs", "environment", "experiment", "scripts", "source", "tests"):
        names.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix != ".pyc")
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(names)}
    target = ROOT / "provenance/release_files.json"
    if a.write:
        target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    elif json.loads(target.read_text()) != manifest:
        raise SystemExit("Release hashes differ; only a reviewed maintainer change may refresh them")
    print(f"Verified {len(manifest)} release files")


if __name__ == "__main__":
    main()
