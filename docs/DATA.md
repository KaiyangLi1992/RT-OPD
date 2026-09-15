# Exact data and model assets

The release downloads fixed training/evaluation JSONL manifests, the frozen
teacher gate, canonical valid vocabulary and audio hashes from a shared 4.5 MB
archive in the HF model repository. Authenticate with `<profile>/.venv/bin/hf
auth login` first. `configs/frozen_data.json` pins the archive revision, hash
and every member hash; all epoch-order hashes are committed with the code. Audio and base weights download from immutable upstream revisions.
The same 10,000 training rows and frozen gate are used by both student profiles.

| Asset | Source | Frozen revision |
|---|---|---|
| Ke student | [KE-Team/Ke-Omni-R-3B](https://huggingface.co/KE-Team/Ke-Omni-R-3B) | `54a602334a1379d9321cf26db6982e29f25bfeaf` |
| Qwen student | [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | `f75b40e3da2003cdd6e1829b1f420ca70797c34e` |
| Frozen teacher | [KE-Team/Ke-Omni-R](https://huggingface.co/KE-Team/Ke-Omni-R) | `a8c65184c6f6e5e8ab3fccd29b4111e6e6b8bfc4` |
| Training audio archive | [frozen AudioMCQ archive](https://huggingface.co/datasets/bmmv-9x2q7/aa-opd-v1-training-audio-cb33687) | `600d3d8d7630e87367934bdeb25a802da8a3c87e` |
| Training upstream | [AudioMCQ-StrongAC-GeminiCoT](https://huggingface.co/datasets/AudioLLMs/dcase2026_task5_AudioMCQ-StrongAC-GeminiCoT) | `cb33687ce4dc3dace4e203a2dd584fa46ae312da` |
| MMAU full | [gamma-lab-umd/MMAU-test](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test) | `8e835a9f64ed6c703b3c9ddb6d423d9ab697061e` |
| MMAU mini | [gamma-lab-umd/MMAU-test-mini](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini) | `ccd9696c0111ea7060827598f310558df0b71b0a` |
| MMAR | [BoJack/MMAR](https://huggingface.co/datasets/BoJack/MMAR) | `3bd051123480e80d273ae9e8e9f1653f49010ac7` |
| ADQA dev | [Harland/DCASE2026-Task5-DevSet](https://huggingface.co/datasets/Harland/DCASE2026-Task5-DevSet) | `3280a1aebcd3a8b542b80a1aa175694dd4dfebca` |

For Ke, run `ke/.venv/bin/python ke/scripts/prepare.py --scope all`.
For Qwen, replace `ke` by `qwen`. The preparation step performs:

1. Reconstruct the cost-balanced permutations for both epochs of every seed;
   compare exact SHA-256 against the original experiments.
2. Download and hash every required base-model file.
3. Download the 8.17 GB training audio archive, verify its SHA-256, and extract it.
4. Resolve the frozen manifests to local audio paths and verify audio byte hashes.
5. Reconstruct benchmark audio, including ADQA at 32 kHz mono PCM16, and verify it.
6. Download the full MMAU input/audio, preserving hidden gold labels.
7. Write local preparation receipts under `<profile>/assets/`.

`--scope training` avoids benchmark downloads; `--scope evaluation` prepares the
benchmark data; `--verify-only` checks already prepared local assets without
network calls. If a file differs, preparation fails instead of silently using a
new dataset version. Download caches can be reused through the usual HF cache.

The archived dataset has more rows than the experiment uses: `data/frozen/train.jsonl`
defines the exact 10,000-row snapshot. `teacher_gate.jsonl` defines the frozen gate.
Some inherited manifest fields identify paired donor audio. They are retained
for source integrity and existing data auditing but the `linear_noaudio` main
trainer does not feed donor audio to the teacher.

Training/model configurations list exact content hashes, including transformed
local Ke/Qwen model configuration bytes. Dataset permissions remain those of
upstream: the source manifests record Apache-2.0 for training/ADQA and
CC-BY-NC-4.0 for MMAR/MMAU-mini. Consult the linked source cards for full terms.
No blanket new license is applied to third-party data or model weights.
