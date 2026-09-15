"""Pinned hidden-label MMAU-full audio; never downloads or invents gold answers."""
from .common import ROOT, read, rows, check


def download_full():
    from scripts.prepare import hub_file, extract
    import shutil
    c = read(ROOT / "configs/mmau_full.json")
    archive = hub_file(c["repo_id"], c["revision"], "test-audios.tar.gz",
                       ROOT / "downloads/mmau_full", dataset=True)
    check(dict(c["archive"], path=str(archive)))
    unpacked = ROOT / "downloads/mmau_full/unpacked"
    extract(archive, unpacked)
    paths = {}
    for path in unpacked.rglob("*.wav"):
        if path.name in paths:
            raise ValueError("Duplicate full-audio filename")
        paths[path.name] = path
    dest = ROOT / "assets/audio/mmau_full"
    dest.mkdir(parents=True, exist_ok=True)
    for row in rows(ROOT / "assets/grid16/mmau_full.jsonl"):
        name = row["source_id"] + ".wav"
        shutil.copyfile(paths[name], dest / name)
