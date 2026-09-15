# Reproduce benchmark evaluation

## Separate environments

Training and materialization use Python 3.10 / Transformers 4.52.4.
The paper benchmark pipeline uses Python 3.12 / vLLM 0.25.1+cu129 /
Transformers 5.14.1 / PyTorch 2.11.0+cu129. Keep them separate:

```bash
bash tools/setup_evaluation.sh
```

The script installs the exact CUDA 12.9 wheel from the official
[vLLM release](https://github.com/vllm-project/vllm/releases/tag/v0.25.1).
A CUDA-12.9-compatible driver is required for this environment. The inference
runtime check fails if its recorded core package versions do not match.

## Evaluate a freshly reproduced checkpoint

For Ke seed 85, after training has completed:

```bash
ke/.venv/bin/python ke/scripts/prepare.py --scope evaluation
RUN="$PWD/ke/runs/training/rtopd-ke-s85/attempt1"
ke/.venv/bin/python ke/scripts/grid16_eval.py materialize \
  --run "$RUN" --step 626 --gpu 0

for benchmark in mmar adqa; do
  .venv-eval/bin/python ke/scripts/grid16_eval.py shard \
    --run "$RUN" --step 626 --benchmark "$benchmark" --shard-index 0 --gpu 0
  .venv-eval/bin/python ke/scripts/grid16_eval.py close \
    --run "$RUN" --step 626 --benchmark "$benchmark"
done

for shard in 0 1 2 3 4; do
  .venv-eval/bin/python ke/scripts/grid16_eval.py shard \
    --run "$RUN" --step 626 --benchmark mmau_full --shard-index "$shard" --gpu 0
done
.venv-eval/bin/python ke/scripts/grid16_eval.py close \
  --run "$RUN" --step 626 --benchmark mmau_full
```

For Qwen, use `qwen/.venv/bin/python`, `qwen/scripts/...`, and a matching run such
as `$PWD/qwen/runs/training/rtopd-qwen-s92/attempt1`. Both evaluator profiles
explicitly select the correct student when merging its adapter.

The example executes all shards sequentially on one idle GPU. Separate shards
may be assigned distinct free GPUs. Sharding is frozen by batch stride, with
batch size 8; changing batching, generation parameters or prompt mode can
change results.

## Outputs and metrics

- MMAR and ADQA: `evaluation-626/<benchmark>/COMPLETE.json` contains local accuracy;
  malformed answers remain in the denominator.
- MMAU full: `evaluation-626/mmau_full/submission.json` contains the 9,000 predictions.
  Submit that file to the [official MMAU scorer](https://huggingface.co/spaces/sonalkum/MMAU-Eval).
  The package does not invent hidden labels or infer full-test accuracy locally.
- `evaluation-626/<benchmark>/<shard>/` contains predictions, audit and source bindings.

MMAU full: greedy, 256 tokens, model EOS only, no vocabulary suppression,
official Ke MMAU prompt. MMAR/ADQA: greedy, 256 tokens, closing `</answer>` stop,
canonical V2 prompt and valid-vocabulary suppression. Backend dtype is float16,
eager execution, max model length 4096 and GPU-memory utilization 0.9.

Macro-3 is `(MMAU_full_accuracy + MMAR_accuracy + ADQA_clean_accuracy) / 3`.
Report the mean and sample standard deviation across all five seeds. The mini
split is not part of the paper's Macro-3. Historical official scores and their
source-file hashes are recorded in `results/main_results.json`.

## Released HF adapter

`tools/infer.py` is the supported one-audio loading example for the published
adapter. It verifies the student choice and can use a hash-verified local base
via `--base-model-dir`. This convenience path is **not** the paper evaluator.

The strict `grid16_eval.py` commands above bind to newly generated local training
receipts and will reject a downloaded adapter lacking that run history. Do not
fabricate a `VERIFIED.json` or alter the published adapter to bypass those checks.
For inference from the released adapter, use the loading example; for an
end-to-end reproduction of main results, run the training and evaluator above.
