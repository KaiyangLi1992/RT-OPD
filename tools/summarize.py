#!/usr/bin/env python3
"""Recompute per-seed Macro-3, means and sample standard deviations."""
import json,statistics
from pathlib import Path
rows=json.loads((Path(__file__).resolve().parents[1]/'results/main_results.json').read_text())
for student in ['ke','qwen']:
 r=[x for x in rows if x['student']==student]
 print(student.upper())
 for x in r:
  macro=statistics.mean(s['accuracy'] for s in x['scores'].values())
  assert abs(macro-x['macro3'])<1e-12
  print('seed',x['seed'],'Macro-3',f'{100*macro:.4f}%')
 print('Mean +/- sample SD:',f'{100*statistics.mean(x["macro3"] for x in r):.4f}',f'+/- {100*statistics.stdev(x["macro3"] for x in r):.4f}')
 print('Best released candidate:',max(r,key=lambda x:x['macro3'])['seed'])
