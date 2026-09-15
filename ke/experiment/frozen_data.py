"""Fetch the shared fixed manifests at a pinned revision and verify every byte."""
import hashlib,json,tarfile,tempfile,os
from pathlib import Path
from .common import ROOT,read,sha


def ensure_frozen_data(*,download=True):
    c=read(ROOT/'configs/frozen_data.json')
    dest=ROOT/'data/frozen'
    def valid():
        return all((dest/n).is_file() and (dest/n).stat().st_size==r['bytes'] and sha(dest/n)==r['sha256'] for n,r in c['files'].items())
    if valid():return dest
    if not download:raise ValueError('Frozen manifests are absent or changed; run scripts/prepare.py with HF authentication')
    from huggingface_hub import hf_hub_download
    archive=Path(hf_hub_download(repo_id=c['repo_id'],revision=c['revision'],filename=c['filename']))
    if archive.stat().st_size!=c['bytes'] or sha(archive)!=c['sha256']:raise ValueError('Frozen data archive identity mismatch')
    dest.mkdir(parents=True,exist_ok=True)
    seen=set()
    with tarfile.open(archive,'r:gz') as t:
        for m in t:
            if not m.isfile() or m.name not in c['files'] or m.name in seen:raise ValueError('Unexpected or duplicate manifest archive member')
            seen.add(m.name)
            data=t.extractfile(m).read();expected=c['files'][m.name]
            if len(data)!=expected['bytes'] or hashlib.sha256(data).hexdigest()!=expected['sha256']:raise ValueError('Frozen data member hash mismatch')
            p=dest/m.name
            if p.exists() and p.read_bytes()!=data:raise ValueError('Changed local manifest preserved: '+str(p))
            if not p.exists():
                with tempfile.NamedTemporaryFile(dir=dest,delete=False) as f:
                    f.write(data);tmp=Path(f.name)
                try:os.link(tmp,p)
                except FileExistsError:
                    if p.read_bytes()!=data:raise ValueError('Concurrent manifest mismatch')
                finally:tmp.unlink(missing_ok=True)
    if seen!=set(c['files']) or not valid():raise ValueError('Incomplete frozen data archive')
    return dest
