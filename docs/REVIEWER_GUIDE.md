# Reviewer guide

## What the method does

At a prefix sampled from the current student, one frozen teacher produces
probabilities with the true audio (`p_audio`) and with audio omitted (`p_noaudio`).
Both receive the same question and continuation token sequence. With alpha=1,
RT-OPD uses:

```
log q = log_softmax(log p_audio + alpha * (log p_audio - log p_noaudio))
loss  = answer CE + 0.25 * frozen_gate * KL(p_student || q)
```

The distributions use the frozen valid-vocabulary mask. The teacher target is
detached from autograd. A frozen gate enables the distillation term only for
teacher-correct and parseable training examples; all examples retain gold-answer
CE. The rollout is online and is not a precomputed teacher response.

## Code map

Paths below are under either `ke/` or `qwen/`.

| Component | Code |
|---|---|
| Exact effective hyperparameters, seeds and tasks | `configs/ke_grid16.json` (historical filename) |
| Present/absent teacher inputs and target construction | `experiment/target_trainer.py` |
| Reverse KL and online rollout integration | `experiment/target_trainer.py`, imported `source/portable/ke_opd_v2/` |
| Fresh initialization, optimizer, two epochs, row audits | `experiment/train.py` |
| Last partial batch (16 rows) and epoch continuation | `experiment/data_continuation.py`, `experiment/grid.py` |
| Cost-balanced deterministic row order | `experiment/grid_assets.py`, `source/portable/scripts/fast4_cost_balanced_sampler.py` |
| Download pinned model/audio versions and verify bytes | `scripts/prepare.py` |
| One-seed launch with preflight and two-step smoke | `scripts/launch.py`, `scripts/run.py` |
| Exact checkpoint materialization and evaluation | `scripts/grid16_eval.py`, `scripts/grid16_inference.py` |

## Effective recipe

| Setting | Ke | Qwen |
|---|---|---|
| Initial student | Ke-Omni-R-3B | Qwen2.5-Omni-3B |
| Frozen teacher | Ke-Omni-R 7B | Same |
| Seeds | 82, 83, 84, 85, 86 | 92, 93, 94, 95, 96 |
| Learning rate | 7.5e-5 | 2.5e-5 |
| Final checkpoint | 626 | 626 |
| Epochs / rows | 2 / 10,000 | Same |
| Global batch | 32 | Same |
| GPUs × microbatch × accumulation | 4 × 4 × 2 | Same |
| LoRA rank / alpha / dropout | 64 / 128 / 0.05 | Same |
| Weight decay / gradient norm clip | 0.01 / 1.0 | Same |
| LR scheduler / horizon / warmup | cosine / 626 / 32 updates | Same |
| RT alpha / KL temperature / KL coefficient | 1.0 / 1.0 / 0.25 | Same |
| Rollout temperature / top-p / top-k / length | 1.0 / 0.95 / 64 / 96 tokens | Same |
| Precision / attention | BF16 / SDPA | Same |

The inherited run name `N` means `linear_noaudio`, the main RT-OPD method.
`initialization_code=K` selects Ke and `Q` selects Qwen. Historical file/class
names are not separate methods. All meaningful run settings are printed by
`launch_ke.sh plan` and `launch_qwen.sh plan` and captured in the launch contract.

## Interpreting results

The two five-seed means match the manuscript table. The uploaded Ke seed 85 is a
single model selected post hoc using Macro-3. It must be labeled separately
from the mean. MMAU full uses 9,000 hidden-label test examples and an official
scorer. The 1,000-example mini set is separate and is not substituted in Macro-3.
Neither MMAR nor ADQA selects the training step in these replication runs.

Reproduction means matching the prescribed data, settings and evaluation
protocol. Hardware-dependent floating-point behavior can change individual
rollouts and exact final scores. The included historical counts provide the
reported results; a new run must be evaluated and reported as a new run.
