# Release validation — 2026-09-15

## Completed

| Check | Evidence / result |
|---|---|
| Original Ke source manifest | Every original recorded source hash matched on the experiment host |
| Original best checkpoint | All 9 files in the original checkpoint manifest passed SHA-256 verification |
| HF upload | Adapter size and remote LFS SHA-256 match the original trained bytes |
| HF re-download | Downloaded the published adapter again and matched SHA-256 |
| CPU numerical/input tests | 20 passed for Ke and 20 passed for Qwen |
| Actual HF training-loop tail | CPU test verified the 313-update half-batch boundary against an explicit gradient reference |
| Original data order | All 20 epoch permutations (10 seeds × 2 epochs) exactly match original hashes |
| Clean source package | A source-only archive downloaded the pinned HF manifests, reconstructed both profiles' orders, verified release hashes and passed all 40 tests |
| Upstream configuration download | Pinned Ke student and teacher config bytes match expected hashes |
| MMAU input reconstruction | Pinned public Parquet converted to exactly the expected 9,000 hidden-label rows and manifest hash |
| Evaluation audio verification | 12,577 exact audio files verified across MMAU-full, MMAU-mini, MMAR and ADQA |
| Aggregate results | Per-seed Macro-3 and five-seed mean/sample-SD reproduce both main paper rows |
| Python / shell syntax and launch plans | Passed for both profiles |
| Published adapter GPU load and inference | HF adapter loaded with exact base on a TITAN RTX using float16; one real ADQA audio generated an answer |

The clean-source check uses the same files delivered by GitHub, without any
preexisting `data/frozen` directory. Its successful HF download proves that the
frozen-data dependency is self-contained at the pinned repository revision.

The training runtime's core versions matched the original launch contract. CPU
tests used the existing Python 3.10 environment with temporary pytest/pyarrow
packages as needed. That existing environment required its existing
`pkg_resources` compatibility overlay for librosa; the release setup pins
setuptools 65.5.0, which provides `pkg_resources` directly. No shared environment
was modified during validation.

## Limits

No new 626-step training or four-GPU BF16 qualification was run for this
packaging release. Nitro2 has 24GB TITAN RTX GPUs, below the training preflight
requirement; the original RTX 6000 Ada training host was unreachable. The
launchers still require a real two-update qualification on the reviewer's
four-GPU machine before formal training.

Mantis was successfully accessed on 2026-09-15 and its Slurm node inventory was
checked. Its A100 nodes expose one GPU each; its largest L40S nodes
(`mantis-034` through `mantis-036`) expose three GPUs each. No advertised node
provides the four matching GPUs required by the unchanged single-node launcher.
A scheduler-only check (`sbatch --test-only --account=pi-ji --qos=general
--partition=general --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=56G
--gres=gpu:4 --time=00:10:00 --wrap='hostname'`) returned
`Requested node configuration is not available`. No training job was submitted
and this resource check is not a passed GPU qualification. Multi-node execution
would require a separately validated launcher adaptation.

The GPU loading example used float16, one audio, and the Transformers
convenience path. It does not establish the paper's vLLM accuracy or rerun a full
benchmark. Historical results are supported by the original run records and
hashes in `results/main_results.json`.

The complete audio archives and every model shard were not downloaded again.
Existing benchmark audio was hash-verified, pinned small upstream configuration
and Parquet downloads were replayed, and the newly published HF adapter/data
archive was downloaded and checked. A new complete training/evaluation virtual
environment was not installed from scratch during this release; the provided
install scripts remain part of the new-machine workflow.

## Recheck locally

```bash
python3 tools/verify_release.py
python3 tools/summarize.py
ke/.venv/bin/python -m pytest ke/tests -q
qwen/.venv/bin/python -m pytest qwen/tests -q
```

Data byte verification runs in `scripts/prepare.py`; the GPU smoke runs via
`launch_ke.sh smoke` or `launch_qwen.sh smoke`. Do not treat CPU tests as proof
that a new GPU topology has qualified.
