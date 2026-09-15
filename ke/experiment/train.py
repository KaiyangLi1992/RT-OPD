"""Fresh 0->626 teacher-target OPD; four ranks, exact frozen row order, no resume."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

from experiment.common import ROOT, check, config, identity, model_config, read, rows, sha, verify_prepared, verify_release, verify_runtime, write


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--arm", choices=list(config()["arms"]), required=True)
    p.add_argument("--seed", choices=config()["seeds"], type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    c = config()
    grid = True
    cell = c["arms"][args.arm]
    initialization = c.get("initialization_code", "Q")
    lr = cell.get("learning_rate", c["learning_rate"])
    release = verify_release()
    observed_runtime = verify_runtime()
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or os.environ["RANK"] != os.environ["LOCAL_RANK"]:
        raise ValueError("Exactly four local ranks on a single machine are required")
    receipt = verify_prepared() if rank == 0 else read(ROOT / "assets/training_ready.json")
    assets = receipt["artifacts"]
    stop = 2 if args.smoke else cell["stop_step"]
    checkpoints = (stop,) if args.smoke else tuple(cell["checkpoint_steps"])
    output = args.output.resolve()
    if list(output.glob("checkpoint-*")) or (output / "TRAIN_COMPLETE.json").exists():
        raise FileExistsError("Fresh training never loads an existing checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    data = rows(assets["train"]["path"])
    from experiment.grid import order_paths, row_window
    paths = order_paths(args.seed) if grid else [ROOT / f"data/frozen/order_s{args.seed}.json"]
    if grid:
        frozen_orders = read(ROOT / "configs/ke_grid16_assets.json")["repeat_orders"][str(args.seed)]
        for epoch, path in enumerate(paths):
            check(dict(frozen_orders[str(epoch)], path=str(path)))
    orders = [read(path) for path in paths]
    for order in orders:
        if sorted(order["indices"]) != list(range(10000)) or [str(data[i]["id"]) for i in order["indices"]] != order["ids"]:
            raise ValueError("Frozen data order does not match training rows")
    order_path, order = paths[0], orders[0]

    import torch
    from torch.utils.data import SequentialSampler
    from transformers import TrainerCallback
    from scripts import train_ke_opd_v1_hparam as base
    from experiment.target_trainer import AudioTargetOPDTrainer
    from experiment.data_continuation import OrderedRemainingRows, accumulation_multiplier
    view = OrderedRemainingRows(data, order["indices"], 0)
    target_arm = c["arms"][args.arm]["target_arm"]
    models = model_config()
    run_name = f"step0-s{args.seed}-{args.arm}" + ("-smoke" if args.smoke else "")
    resolved = base.ResolvedRun(
        run_name=run_name, initialization=initialization, learning_rate=lr,
        epoch="fresh_step0", max_steps=stop, checkpoint_steps=checkpoints,
        schedule_profile="contrast_round1_h626_cut470", scheduler_horizon=626,
        seed=args.seed, aa_top_fraction=.30, aa_selection_unit="wordwise", lambda_opd=.25,
        parameterization="lora", lora_rank=64, lora_alpha=128,
        per_device_train_batch_size=4, gradient_accumulation_steps=2, world_size=4,
        engineering_calibration=False, engineering_save_load_checkpoint=False,
        performance_warmup_steps=0, contrast_mode="none", contrast_beta=0., contrast_margin=.10,
        contrast_teacher_ratio_threshold=2., contrast_temperature=1.,
        contrast_ramp_start=32, contrast_ramp_end=80, contrast_smoke=False)
    sys.argv = [__file__, "--run-name", run_name, "--initialization", initialization,
        "--output-dir", str(output), "--learning-rate", str(lr), "--max-steps", str(stop),
        "--checkpoint-steps", ",".join(map(str, checkpoints)), "--schedule-profile", resolved.schedule_profile,
        "--seed", "44", "--aa-top-fraction", ".30", "--lambda-opd", ".25",
        "--lora-rank", "64", "--lora-alpha", "128", "--per-device-train-batch-size", "4",
        "--gradient-accumulation-steps", "2", "--world-size", "4", "--thor2-four-gpu-extension",
        "--train-jsonl", assets["train"]["path"], "--teacher-gate", assets["teacher_gate"]["path"],
        "--vocab-json", assets["valid_vocab"]["path"], "--asset-bundle", assets["bundle"]["path"],
        "--contrast-pair-audit", assets["pair_audit"]["path"],
        "--student-model-dir", str((ROOT / "models" / models["student"]["directory"]).resolve()),
        "--teacher-model-dir", str((ROOT / "models" / models["teacher"]["directory"]).resolve()),
        "--attempt-id", "smoke" if args.smoke else "formal"]
    original_parse = base.parse_args

    def parse():
        parsed = original_parse()
        parsed.seed = args.seed
        return parsed

    def topology(observed, *, enabled, visible_gpu_names):
        if not enabled or observed != resolved or len(visible_gpu_names) != 4 or len(set(visible_gpu_names)) != 1:
            raise ValueError("Training requires four matching GPUs")
        return dict(campaign_id=c["campaign_id"], host=socket.gethostname(),
                    physical_gpu_ids=os.environ["CUDA_VISIBLE_DEVICES"], gpu_names=visible_gpu_names,
                    world_size=4, effective_batch=32, release_sha256=release, qualification=args.smoke)

    original_contract = base.V2Contract

    def contract(**kw):
        return original_contract(**dict(kw, version=c["campaign_id"], lora_dropout=.05))

    original_manifest = base.build_immutable_manifest

    def manifest(**kw):
        m = original_manifest(**kw)
        for role in ("student", "teacher"):
            if m["models"][role]["files"] != models[role]["files"]:
                raise ValueError(f"Model identity mismatch: {role}")
        m.update(experiment_contract=c["campaign_id"], release_sha256=release,
                 experiment=identity(ROOT / ("configs/ke_grid16.json" if grid else "configs/experiment.json")),
                 target_arm=target_arm, reported_arm=args.arm, seed=args.seed,
                 start_step=0, stop_step=stop, parent_checkpoint=None,
                 order=identity(order_path), aa_enabled=False, pac_enabled=False,
                 epoch_orders=[identity(path) for path in paths], target_alpha=cell["alpha"],
                 runtime=observed_runtime, engineering_only=args.smoke,
                 final_partial_batch="16 rows; compensate HF fixed /2 at each 313-update epoch tail")
        m["run"]["arm"] = "U"
        m["selector"] = {"enabled": False, "opd_token_weighting": "uniform"}
        return m

    def fresh_only(output_arg, **kw):
        if kw["explicit"] is not None or Path(output_arg) != output or list(output.glob("checkpoint-*")):
            raise ValueError("Resume and warm-start are disabled in this from-step-0 release")
        return None, None, {"fresh_start": True}

    class Trainer(AudioTargetOPDTrainer):
        def __init__(self, *pos, **kw):
            super().__init__(*pos, **kw)
            self.target_arm = target_arm
            self.target_alpha = cell["alpha"]
            self.model_accepts_loss_kwargs = False
            if self.accelerator.gradient_accumulation_steps != 1:
                raise ValueError("Unexpected Accelerate gradient accumulation behavior")
            self.executed_microbatches = 0
            self.opd_rows = 0
            self.add_callback(Audit(self))

        def get_train_dataloader(self):
            if self.args.dataloader_num_workers != 0:
                raise ValueError("Epoch order switching requires synchronous data loading")
            return super().get_train_dataloader()

        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        def _opd_loss(self, *pos, **kw):
            value = super()._opd_loss(*pos, **kw)
            self.opd_rows += 1
            return value

        def compute_loss(self, model, inputs, **kw):
            step = int(self.state.global_step) + 1
            epoch, local_step, slot, offset = row_window(step, self.executed_microbatches, rank)
            ids = [str(row["id"]) for row in inputs]
            if ids != orders[epoch]["ids"][offset:offset + 4]:
                raise ValueError(f"Data order mismatch: rank={rank}, step={step}, slot={slot}")
            loss = super().compute_loss(model, inputs, **kw)
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Nonfinite loss")
            multiplier = accumulation_multiplier(local_step)
            self.executed_microbatches += 1
            with (output / f"rows-rank-{rank}.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(step=step, slot=slot, ids=ids, loss=float(loss.detach()),
                                            accumulation_multiplier=multiplier), allow_nan=False) + "\n")
            return loss * multiplier

    class Audit(TrainerCallback):
        def __init__(self, trainer):
            self.t = trainer
            self.norms = []

        def on_train_begin(self, training_args, state, control, **kw):
            t = self.t
            if state.global_step != 0 or t.lr_scheduler.last_epoch != 0 or t.optimizer.state:
                raise ValueError("Model/optimizer/scheduler must start fresh at step 0")
            from peft import get_peft_model_state_dict
            h = hashlib.sha256()
            for key, tensor in sorted(get_peft_model_state_dict(t.model).items()):
                h.update(key.encode())
                h.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
            write(output / f"START_RANK_{rank}.json",
                  dict(step=0, optimizer_states=0, scheduler_step=0, seed=args.seed,
                       initial_adapter_sha256=h.hexdigest(), parent_checkpoint=None))

        def on_epoch_begin(self, training_args, state, control, **kw):
            epoch = int(state.global_step) // 313
            if epoch >= len(orders):
                raise ValueError("Unexpected extra training epoch")
            view.rows = [data[i] for i in orders[epoch]["indices"]]

        def on_pre_optimizer_step(self, training_args, state, control, **kw):
            t = self.t
            if any(p.grad is not None for p in t.v2_teacher.parameters()):
                raise ValueError("Frozen teacher received gradients")
            grads = [p.grad for p in t.model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise FloatingPointError("Missing or nonfinite trainable gradients")
            norm = float(torch.stack([g.detach().float().norm() for g in grads]).norm())
            if norm <= 0:
                raise FloatingPointError("Zero trainable gradient")
            self.norms.append(norm)

        def on_train_end(self, training_args, state, control, **kw):
            t = self.t
            expected_micro = stop * 2 - stop // 313
            if state.global_step != stop or len(self.norms) != stop or t.executed_microbatches != expected_micro:
                raise ValueError("Incomplete optimizer or sample coverage")
            if not t.opd_rows:
                raise ValueError("No gated OPD rows exercised")
            total = torch.cuda.get_device_properties(training_args.device).total_memory
            reserved = torch.cuda.max_memory_reserved(training_args.device)
            write(output / f"RANK_AUDIT_{rank}.json", dict(
                seed=args.seed, arm=args.arm, target_arm=target_arm, start=0, stop=stop,
                optimizer_updates=len(self.norms), microbatches=t.executed_microbatches,
                opd_rows=t.opd_rows, gradient_norms_after_clip=self.norms,
                teacher_gradients_absent=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(training_args.device),
                peak_reserved_bytes=reserved, total_device_bytes=total,
                memory_headroom_bytes=total-reserved))

    base.parse_args = parse
    base.validate_and_resolve = lambda parsed: resolved
    base.validate_thor2_four_gpu_extension = topology
    base.V2Contract = contract
    base.build_immutable_manifest = manifest
    base.recover_interrupted_checkpoints_and_resolve = fresh_only
    base.JsonlRows = lambda path: view
    base.KeOPDV2Trainer = Trainer
    base.main()


if __name__ == "__main__":
    main()
