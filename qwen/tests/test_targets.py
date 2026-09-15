"""CPU numerical and input-contract tests; no model download or GPU use."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

from ke_opd_v2.losses import reverse_kl_per_position
from experiment.target_trainer import build_teacher_target, prepare_noaudio_prompt_inputs, reverse_kl_to_target


def tensors():
    g = torch.Generator().manual_seed(37)
    return (torch.randn(2, 7, 19, generator=g), torch.randn(2, 7, 19, generator=g),
            torch.randn(2, 7, 19, generator=g), torch.tensor([0, 2, 4, 7, 9, 11, 13, 18]))


@pytest.mark.parametrize('arm', ['uniform', 'linear_noaudio', 'linear_donor', 'sigmoid_donor'])
@pytest.mark.parametrize('same_audio', [False, True])
def test_uniform_recovery_value_and_student_gradient(arm, same_audio):
    student, real, negative, ids = tensors()
    negative = real.clone() if same_audio else negative
    student.requires_grad_(True)
    reference = reverse_kl_per_position(student, real, ids)
    expected_grad = torch.autograd.grad(reference.sum(), student)[0]
    target, _ = build_teacher_target(real, negative, ids, arm, alpha=None if same_audio else 0.0)
    actual = reverse_kl_to_target(student, target, ids)
    actual_grad = torch.autograd.grad(actual.sum(), student)[0]
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=5e-7)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-6, atol=5e-7)


@pytest.mark.parametrize('arm', ['linear_noaudio', 'linear_donor', 'sigmoid_donor'])
def test_normalized_probability_difference_is_logit_shift_invariant(arm):
    _, real, negative, ids = tensors()
    first, _ = build_teacher_target(real, negative, ids, arm)
    second, _ = build_teacher_target(real + 23.0, negative - 11.0, ids, arm)
    torch.testing.assert_close(first, second, rtol=2e-6, atol=4e-6)
    torch.testing.assert_close(first.exp().sum(-1), torch.ones_like(first[..., 0]))


@pytest.mark.parametrize('arm', ['linear_donor', 'sigmoid_donor'])
def test_exact_formula_and_chunk_independence(arm):
    _, real, negative, ids = tensors()
    lm = F.log_softmax(real.index_select(-1, ids), -1)
    ln = F.log_softmax(negative.index_select(-1, ids), -1)
    correction = 0.5 * (lm - ln) if arm == 'linear_donor' else F.logsigmoid(lm - ln)
    expected = F.log_softmax(lm + correction, -1)
    actual, stats = build_teacher_target(real, negative, ids, arm, position_chunk_size=1)
    other, other_stats = build_teacher_target(real, negative, ids, arm, position_chunk_size=8)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, other)
    assert stats == pytest.approx(other_stats)
    assert stats['target_kl_q_to_teacher_real'] >= -2e-6


@pytest.mark.parametrize('arm', ['linear_donor', 'sigmoid_donor'])
def test_extreme_finite_and_teacher_detached(arm):
    real = torch.tensor([[[10000.0, -10000.0, 0.0]]], requires_grad=True)
    negative = torch.tensor([[[-10000.0, 10000.0, 0.0]]], requires_grad=True)
    ids = torch.arange(3)
    target, stats = build_teacher_target(real, negative, ids, arm)
    student = torch.zeros_like(real, requires_grad=True)
    loss = reverse_kl_to_target(student, target, ids).mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
    assert not target.requires_grad and real.grad is None and negative.grad is None
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())


def test_noaudio_omits_content_placeholder_and_audio_argument():
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{'role': 'user', 'content': [{'type': 'text', 'text': 'same question'}]}]
            assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
            return 'text-only rendered prefix'

        def __call__(self, **kwargs):
            assert kwargs == dict(text='text-only rendered prefix', return_tensors='pt', padding=True)
            return dict(input_ids=torch.tensor([[3, 4]]), attention_mask=torch.ones(1, 2, dtype=torch.long))

    result = prepare_noaudio_prompt_inputs(Processor(), 'same question', torch.device('cpu'))
    assert set(result) == {'input_ids', 'attention_mask'}


def test_rejects_nonfinite_and_misaligned_inputs():
    _, real, negative, ids = tensors()
    with pytest.raises(ValueError):
        build_teacher_target(real, negative[:, :-1], ids, 'linear_donor')
    real[0, 0, 0] = float('nan')
    with pytest.raises(FloatingPointError):
        build_teacher_target(real, negative, ids, 'sigmoid_donor')


def test_hf_microbatch_mean_ga2_matches_full_batch_gradient(tmp_path):
    """Exercise real HF training_step/Accelerate, not only algebraic division."""
    from transformers import Trainer, TrainingArguments

    class MeanLossTrainer(Trainer):
        def compute_loss(self, model, inputs, **kwargs):
            del kwargs
            return F.mse_loss(model(inputs['x']), inputs['y'], reduction='mean')

    generator = torch.Generator().manual_seed(91)
    model = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.3, 0.5]]))
    inputs = dict(x=torch.randn(8, 3, generator=generator), y=torch.randn(8, 1, generator=generator))
    expected = torch.autograd.grad(F.mse_loss(model(inputs['x']), inputs['y']), model.weight)[0]
    trainer = MeanLossTrainer(model=model, args=TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, gradient_accumulation_steps=2,
        report_to=[], disable_tqdm=True,
    ))
    trainer.model_accepts_loss_kwargs = False
    for begin in (0, 4):
        trainer.training_step(model, {key: value[begin:begin + 4] for key, value in inputs.items()})
    torch.testing.assert_close(model.weight.grad, expected, rtol=2e-6, atol=2e-7)
