# RT-OPD: Reward-Tilted On-Policy Distillation

Reproducibility package for the **main absent-audio RT-OPD experiment** with
Ke-Omni-R-3B and Qwen2.5-Omni-3B students. Both use the same frozen Ke-Omni-R 7B
teacher. Each run starts from its original student with fresh LoRA and trains
for **626 steps / two epochs** on the same 10,000 examples.

**Code:** [KaiyangLi1992/RT-OPD](https://github.com/KaiyangLi1992/RT-OPD). **Best Ke adapter:**
[KaiyangLi/RT-OPD-Ke-3B](https://huggingface.co/KaiyangLi/RT-OPD-Ke-3B).
These repositories are initially private and require authorized access. This is
an identified research release, not an anonymous review package.

## Start here

| Goal | Entry point |
|---|---|
| Understand the method and map equations to code | [Reviewer guide](docs/REVIEWER_GUIDE.md) |
| Download the exact data and model versions | [Data guide](docs/DATA.md) |
| Reproduce training | Ke and Qwen commands below |
| Run the released Ke adapter on one audio file | [Inference](#use-the-released-ke-model) |
| Reproduce benchmark evaluation / export MMAU submission | [Evaluation guide](docs/EVALUATION.md) |
| Recompute published aggregates without a GPU | `python3 tools/summarize.py` |
| See what has actually been checked | [Validation](docs/VALIDATION.md) |

## Environment and hardware

Linux x86-64, Python **3.10**, four matching NVIDIA GPUs with at least **44 GiB
visible VRAM each** and native BF16 support (48GB A6000/RTX 6000 Ada class or
larger), and a CUDA-12.4-compatible NVIDIA driver. The original training used
four RTX 6000 Ada GPUs. Use an existing allocation of idle GPUs.

The setup script creates an isolated environment with PyTorch 2.5.1+cu124,
Transformers 4.52.4, PEFT 0.19.1 and the complete pinned dependency list. Model
weights, archives, extracted audio, evaluation materializations and five runs
require substantial disk space; allow **at least 150 GB per profile** for a
complete local workflow. Paths are relative to the clone; no original server
or home directory is required.

Clone with an authenticated Git client:

```bash
git clone https://github.com/KaiyangLi1992/RT-OPD.git
cd RT-OPD
```

### Ke student

```bash
bash ke/environment/setup.sh
ke/.venv/bin/hf auth login
ke/.venv/bin/python ke/scripts/prepare.py --scope training
./launch_ke.sh plan --seed 85
./launch_ke.sh smoke --seed 85 --gpus 0,1,2,3
./launch_ke.sh train --seed 85 --gpus 0,1,2,3
```

### Qwen student

```bash
bash qwen/environment/setup.sh
qwen/.venv/bin/hf auth login
qwen/.venv/bin/python qwen/scripts/prepare.py --scope training
./launch_qwen.sh plan --seed 92
./launch_qwen.sh smoke --seed 92 --gpus 0,1,2,3
./launch_qwen.sh train --seed 92 --gpus 0,1,2,3
```

`plan` prints every effective setting and allocates no GPU. `smoke` performs
2 real optimizer updates and saves an engineering checkpoint. `train` first
runs or verifies that smoke, then starts fresh formal training. The separate
smoke always uses the profile's first seed; it is not the requested formal
seed's scientific result. Never report its checkpoint as a main result.

To reproduce all five seeds, run sequentially on the same allocation:

```bash
for seed in 82 83 84 85 86; do ./launch_ke.sh train --seed "$seed" --gpus 0,1,2,3; done
for seed in 92 93 94 95 96; do ./launch_qwen.sh train --seed "$seed" --gpus 0,1,2,3; done
```

Outputs are `ke/runs/training/rtopd-ke-s85/attempt1/checkpoint-626` and, for
example, `qwen/runs/training/rtopd-qwen-s92/attempt1/checkpoint-626`.
Interrupted attempts are preserved. After fixing the cause, use `--attempt
attempt2` to begin a new run. Formal training does not silently resume or reuse
an intermediate checkpoint. `RTOPD_PYTHON=/absolute/path/python` overrides the
wrapper's environment path.

## Main results

Accuracies in percent; Macro-3 is the unweighted mean of the three benchmark
accuracies. ± is the **sample standard deviation of per-seed Macro-3**.

| Student / result | MMAU full (9,000) | MMAR (1,000) | ADQA-cl (1,577) | Macro-3 |
|---|---:|---:|---:|---:|
| Ke: mean of seeds 82–86 | 72.72 | 60.08 | 56.45 | 63.08 ± 0.36 |
| Qwen: mean of seeds 92–96 | 72.18 | 58.90 | 55.70 | 62.26 ± 0.15 |
| Released Ke seed 85 | 72.78 | 61.10 | 57.07 | 63.65 |

The released adapter is the **post-hoc highest-Macro-3 seed**, not an average of
weights or an independent unbiased test estimate. All five final checkpoints
were fixed at step 626. Seed 86 is best on MMAU alone (73.12%); it is not the
best on Macro-3. Exact counts/accuracies and source hashes are in
[main_results.json](results/main_results.json).

## Use the released Ke model

After creating the Ke environment, authenticate to the private HF repository
using `ke/.venv/bin/hf auth login` (enter credentials interactively).

```bash
ke/.venv/bin/python tools/infer.py \
  --adapter KaiyangLi/RT-OPD-Ke-3B \
  --revision 9428480fdbbef20fbbc9f9c621c2be3a30f88f80 \
  --audio /absolute/path/example.wav \
  --question "Which sound is audible?" \
  --choices "A dog barking" "A piano playing"
```

The helper downloads the pinned base and attaches the adapter to its Thinker.
It needs no teacher or training data. Use `--dtype float16` on older GPUs;
this convenience inference is separate from the frozen paper evaluator.
For a locally trained Qwen adapter, pass `--student qwen --adapter
/absolute/path/to/checkpoint-626`; the base must match the selected student.

## Scope and provenance

The original loss, model-loading code and training algorithm are preserved.
This package fixes packaging and launch issues: explicit student profiles,
all-seed order reconstruction, Qwen asset preparation, portable base paths,
and the Qwen materialization initialization. Legacy grid selection and
machine-specific launchers are omitted. Both profiles include their complete
runtime dependencies and download the same exact frozen data manifests from a
pinned archive in the HF model repository.

This is the main-method release. Vanilla OPD, CAAD, donor, sharpening and other
ablations are not advertised as main RT-OPD runs. The source includes inherited
utility classes for those research projects because the actual main trainer
imports them; inactive branches do not change the published settings.
