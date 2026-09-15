import pytest
import torch
from torch.utils.data import SequentialSampler
from transformers import Trainer, TrainingArguments
from experiment.data_continuation import accumulation_multiplier
from scripts.evaluate import score, validate_predictions


def test_actual_hf_training_loop_uses_the_final_half_batch(tmp_path):
    # 625 one-example microbatches represent the real 625 four-rank microbatches.
    # Compare all 313 HF optimizer updates to an explicit mean-gradient reference.
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))
        def forward(self, x):
            return (self.weight - x / 625).square().mean()
    class MeanTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset)
        def compute_loss(self, model, inputs, **kw):
            self.seen.extend(int(x) for x in inputs["x"])
            return model(inputs["x"]) * accumulation_multiplier(int(self.state.global_step) + 1,
                total_rows=625, microbatch=1, world_size=1, ga=2)
    t = MeanTrainer(model=Model(), train_dataset=[dict(x=float(i)) for i in range(625)],
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, max_steps=313,
            per_device_train_batch_size=1, gradient_accumulation_steps=2, learning_rate=.01,
            lr_scheduler_type="constant", optim="sgd", max_grad_norm=0., weight_decay=0.,
            save_strategy="no", logging_strategy="no", remove_unused_columns=False,
            report_to="none", disable_tqdm=True, dataloader_pin_memory=False))
    t.model_accepts_loss_kwargs = False
    t.seen = []
    t.train()
    assert t.seen == list(range(625))
    assert t.state.global_step == t.lr_scheduler.last_epoch == 313
    expected = .1
    for start in range(0, 625, 2):
        window = list(range(start, min(start + 2, 625)))
        expected -= .02 * (expected - sum(i / 625 for i in window) / len(window))
    assert float(t.model.weight) == pytest.approx(expected, abs=1e-6)

def test_parser_failure_stays_in_denominator_and_duplicate_ids_fail():
    gold = [dict(id="a", choices=["dog", "cat"], gold_index=0),
            dict(id="b", choices=["dog", "cat"], gold_index=1)]
    pred = [dict(id=r["id"], gold_index=r["gold_index"], raw_output=text, generated_tokens=8, **score(text, r))
            for r, text in zip(gold, ["<answer>dog</answer>", "unparseable"])]
    assert validate_predictions(pred, gold)["accuracy"] == .5
    with pytest.raises(ValueError):
        validate_predictions([pred[0], pred[0]], gold)
