#!/usr/bin/env python3
"""Check committed file hashes and fixed main-method settings without a GPU."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for profile,seeds,lr in [('ke',list(range(82,87)),7.5e-5),('qwen',list(range(92,97)),2.5e-5)]:
 d=root/profile
 c=json.loads((d/'configs/ke_grid16.json').read_text())
 assert c['seeds']==seeds and set(c['arms'])=={'N'}
 assert c['arms']['N']==dict(alpha=1.0,learning_rate=lr,stop_step=626,checkpoint_steps=[626],target_arm='linear_noaudio')
 assert c['world_size']*c['microbatch']*c['gradient_accumulation_steps']==32
 manifest=json.loads((d/'provenance/release_files.json').read_text())
 for name,digest in manifest.items():
  assert hashlib.sha256((d/name).read_bytes()).hexdigest()==digest,name
 print(profile,len(manifest),'committed files verified; all five main seeds and parameters verified')
a=json.loads((root/'ke/configs/frozen_data.json').read_text())
b=json.loads((root/'qwen/configs/frozen_data.json').read_text())
assert a==b
print('Ke/Qwen fixed data source and hash inventories are identical')
