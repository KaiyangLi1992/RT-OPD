"""Reconstruct runtime-only assets; publish only algorithms and aggregate hashes."""
import hashlib
import json
from pathlib import Path
from .common import ROOT, read, rows, check, sha, write


def prepare_orders():
    from scripts.fast4_cost_balanced_sampler import CostBalancedRandomSampler, load_teacher_gate_costs
    from .grid import order_paths
    from .common import config
    data = rows(ROOT / "data/frozen/train.jsonl")
    costs = load_teacher_gate_costs(ROOT / "data/frozen/teacher_gate.jsonl", 10000)
    expected = read(ROOT / "configs/ke_grid16_assets.json")["repeat_orders"]
    for seed in config()["seeds"]:
        sampler = CostBalancedRandomSampler(data, costs=[costs[str(r["id"])] for r in data], seed=seed,
                                            world_size=4, microbatch=4, gradient_accumulation_steps=2)
        for epoch, path in enumerate(order_paths(seed)):
            sampler.set_epoch(epoch)
            indices = list(sampler)
            if sorted(indices) != list(range(10000)):
                raise ValueError("Sampler failed to cover all training rows")
            write(path, dict(seed=seed, epoch=epoch, indices=indices, ids=[str(data[i]["id"]) for i in indices]))
            check(dict(expected[str(seed)][str(epoch)], path=str(path)))


def prepare_wrapper(download):
    from scripts.prepare import hub_file
    rec = read(ROOT / "configs/ke_grid16_assets.json")["wrapper"]
    path = ROOT / "assets/grid16/wrapper/config.json"
    if download:
        hub_file(rec["repo_id"], rec["revision"], rec["filename"], path.parent)
    check(dict(path=str(path), sha256=rec["sha256"], size_bytes=rec["size_bytes"]))


def full_row(source):
    """Same hidden-label schema as the existing Nitro2 full-input manifest."""
    if source.get("answer") not in (None, ""):
        raise ValueError("Expected a hidden-answer test row")
    sid = str(source["id"])
    question, choices = source["question"], source["choices"]
    return dict(answers_hidden=True, audio_path=f"mmau_full/{sid}.wav", benchmark="mmau_full",
        category=source["category"], choices=choices, dataset_version="MMAU-v05.15.25",
        difficulty=source["difficulty"], id=f"mmau_full:{sid}", num_choices=len(choices),
        prompt=f"{question} Please choose the answer from the following options: {choices!r}. Output the final answer in <answer> </answer>.",
        question=question, source_dataset=source["dataset"], source_id=sid, split=source["split"],
        sub_category=source["sub-category"],
        system_prompt="You are an audio understanding model that answers multiple choice questions based on audio content.",
        task=source["task"])


def prepare_full_manifest(download):
    import pyarrow.parquet as pq
    from scripts.prepare import hub_file
    c = read(ROOT / "configs/mmau_full.json")
    expected = read(ROOT / "configs/ke_grid16_assets.json")
    parquet = ROOT / "downloads/mmau_full/test.parquet"
    if download:
        hub_file(c["repo_id"], c["revision"], "test.parquet", parquet.parent, dataset=True)
    check(dict(expected["full_parquet"], path=str(parquet)))
    data = [full_row(r) for r in pq.read_table(parquet).to_pylist()]
    if len(data) != 9000 or len({r["id"] for r in data}) != 9000:
        raise ValueError("MMAU-full coverage mismatch")
    path = ROOT / "assets/grid16/mmau_full.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in data)
    if path.exists() and path.read_text() != body:
        raise ValueError("Existing full manifest differs; preserve it for audit")
    if not path.exists():
        path.write_text(body)
    check(dict(expected["full_manifest"], path=str(path)))
    return path


def full_audio_identities():
    records = {}
    for row in rows(ROOT / "assets/grid16/mmau_full.jsonl"):
        relative = row["audio_path"]
        path = ROOT / "assets/audio" / relative
        records[relative] = dict(sha256=sha(path), size_bytes=path.stat().st_size)
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != read(ROOT / "configs/ke_grid16_assets.json")["full_audio_tree_sha256"]:
        raise ValueError("Full-test audio differs from the frozen reference inventory")
    return records
