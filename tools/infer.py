#!/usr/bin/env python3
"""One-audio convenience inference; use the frozen evaluator for paper scores."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--student',choices=['ke','qwen'],default='ke')
p.add_argument('--adapter',default='KaiyangLi/RT-OPD-Ke-3B')
p.add_argument('--revision',default='main')
p.add_argument('--audio',type=Path,required=True)
p.add_argument('--question',required=True)
p.add_argument('--choices',nargs='+',required=True)
p.add_argument('--device',default='cuda:0')
p.add_argument('--dtype',choices=['bfloat16','float16'],default='bfloat16')
p.add_argument('--base-model-dir',type=Path)
a=p.parse_args()
if not a.audio.is_file():p.error('Audio file does not exist')
if len(a.choices)<2:p.error('Provide at least two answer choices')
sys.path.insert(0,str(ROOT/a.student))
sys.path.insert(0,str(ROOT/a.student/'source/portable'))
from experiment.frozen_data import ensure_frozen_data
ensure_frozen_data(download=True)
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import StoppingCriteriaList
from scripts.eval_ke_opd_v2 import CompleteAnswer
from ke_opd_v2.modeling import load_thinker,load_processor,load_audio,prepare_prompt_inputs
from ke_opd_v2.contract import canonical_user_text
config=json.loads((ROOT/a.student/'configs/models_ke_grid16.json').read_text())['student']
base=str(a.base_model_dir.resolve()) if a.base_model_dir else snapshot_download(config['repo_id'],revision=config['revision'],allow_patterns=list(config['files']))
if a.base_model_dir:
 import hashlib
 for name,record in config['files'].items():
  h=hashlib.sha256()
  with (Path(base)/name).open('rb') as f:
   for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
  if h.hexdigest()!=record['sha256']:raise ValueError('Base snapshot hash mismatch: '+name)
adapter=str(Path(a.adapter).resolve()) if Path(a.adapter).is_dir() else snapshot_download(a.adapter,revision=a.revision,allow_patterns=['adapter_config.json','adapter_model.safetensors'])
ac=json.loads((Path(adapter)/'adapter_config.json').read_text())
if ac.get('base_model_name_or_path')!=config['repo_id']:
 launch=Path(adapter).parent/'launch_contract.json'
 bound=json.loads(launch.read_text()).get('immutable',{}) if launch.is_file() else {}
 if bound.get('models',{}).get('student',{}).get('files')!=config['files']:
  raise ValueError('Adapter base does not match --student; use the matching Ke or Qwen adapter')
processor=load_processor(base)
model=PeftModel.from_pretrained(load_thinker(base,dtype=getattr(torch,a.dtype)),adapter,is_trainable=False).to(a.device).eval()
sr=processor.feature_extractor.sampling_rate
inputs=prepare_prompt_inputs(processor,canonical_user_text(a.question,a.choices),load_audio(a.audio,sr),sr,torch.device(a.device))
inputs={k:(v.to(getattr(torch,a.dtype)) if torch.is_floating_point(v) else v) for k,v in inputs.items()}
vocab=json.loads((ROOT/a.student/'data/frozen/canonical_valid_vocab_v2.json').read_text())
with torch.inference_mode():
 seq=model.generate(**inputs,do_sample=False,max_new_tokens=256,
  suppress_tokens=vocab['forbidden_ids'],pad_token_id=processor.tokenizer.pad_token_id,
  stopping_criteria=StoppingCriteriaList([CompleteAnswer(processor.tokenizer,inputs['input_ids'].shape[1])]))
print(processor.tokenizer.decode(seq[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True))
