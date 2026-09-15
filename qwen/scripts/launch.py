#!/usr/bin/env python3
"""Reviewable launch for one seed. The default prints the plan only."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment.common import config
from scripts.run import execute
p=argparse.ArgumentParser()
p.add_argument("mode",choices=["plan","smoke","train"],default="plan",nargs="?")
p.add_argument("--seed",type=int,choices=config()["seeds"],default=92)
p.add_argument("--gpus",default="0,1,2,3")
p.add_argument("--attempt",default="attempt1")
a=p.parse_args()
if not a.attempt.isalnum():p.error("--attempt must be alphanumeric")
task=next(t for t in config()["tasks"] if t["seed"]==a.seed)
if a.mode=="plan":
 print(json.dumps(dict(task=task,recipe=config(),gpus=a.gpus,launch=False),indent=2))
else:
 execute(task,a.gpus,smoke_only=a.mode=="smoke",attempt=a.attempt)
