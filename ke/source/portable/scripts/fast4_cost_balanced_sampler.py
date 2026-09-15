"""Deterministic cost balancing for the four-rank fast4 data path.

The sampler deliberately keeps the random membership of every complete
optimizer-step block unchanged.  It only changes which rank receives each
member of that block.  Its output is interleaved for Accelerate's
``BatchSamplerShard(split_batches=False)`` behavior: consecutive dataloader
batches are assigned to ranks by batch index modulo world size.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import RandomSampler, Sampler


COST_PROXY_VERSION = "teacher-gate-correct-plus-generated-tokens-v1"


def teacher_gate_cost(row: dict[str, Any]) -> int:
    """Return a fail-closed integer workload proxy for one frozen gate row.

    Every row pays one unit for the direct-answer forward.  A Teacher-correct,
    parseable row also pays one fixed OPD unit plus its frozen Teacher
    generation length.  The latter is a deterministic proxy for the variable
    rollout/sequence work performed only for gated rows at training time.
    """

    correct = row.get("correct")
    parseable = row.get("parseable")
    generated_tokens = row.get("generated_tokens")
    if not isinstance(correct, bool) or not isinstance(parseable, bool):
        raise ValueError("teacher gate correct/parseable fields must be booleans")
    if isinstance(generated_tokens, bool) or not isinstance(generated_tokens, int):
        raise ValueError("teacher gate generated_tokens must be an integer")
    if generated_tokens < 0:
        raise ValueError("teacher gate generated_tokens must be non-negative")
    gated = correct and parseable
    return 1 + (1 + generated_tokens if gated else 0)


def load_teacher_gate_costs(path: str | Path, expected_count: int) -> dict[str, int]:
    """Load one deterministic cost per unique Teacher-gate sample ID."""

    costs: dict[str, int] = {}
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["id"])
        if sample_id in costs:
            raise ValueError(f"duplicate teacher-gate cost ID: {sample_id}")
        costs[sample_id] = teacher_gate_cost(row)
    if len(costs) != expected_count:
        raise ValueError(
            f"teacher-gate costs contain {len(costs)} rows, expected {expected_count}"
        )
    return costs


def seeded_random_order(length: int, *, seed: int, epoch: int) -> list[int]:
    """Match Accelerate's SeedableRandomSampler order for this dataset length."""

    if length < 0:
        raise ValueError("sampler length cannot be negative")
    if epoch < 0:
        raise ValueError("sampler epoch cannot be negative")
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(epoch))
    return list(RandomSampler(range(length), generator=generator))


def accelerate_rank_assignments(
    block: Sequence[int],
    *,
    world_size: int,
    microbatch: int,
) -> tuple[tuple[int, ...], ...]:
    """Model Accelerate 1.13's no-split batch-index-mod-world sharding."""

    if world_size <= 0 or microbatch <= 0:
        raise ValueError("world_size and microbatch must be positive")
    if len(block) % (world_size * microbatch) != 0:
        raise ValueError("block must contain complete batches for every rank")
    ranks: list[list[int]] = [[] for _ in range(world_size)]
    for batch_index, offset in enumerate(range(0, len(block), microbatch)):
        rank = batch_index % world_size
        ranks[rank].extend(block[offset : offset + microbatch])
    return tuple(tuple(items) for items in ranks)


def rank_cost_spread(
    assignments: Sequence[Sequence[int]], costs: Sequence[int | float]
) -> float:
    loads = [sum(float(costs[index]) for index in rank) for rank in assignments]
    return max(loads) - min(loads) if loads else 0.0


def greedy_balance_block(
    block: Sequence[int],
    costs: Sequence[int | float],
    *,
    world_size: int,
    microbatch: int,
) -> list[int]:
    """Capacity-constrained deterministic LPT assignment, then interleave ranks.

    Relative random order is retained within each assigned rank.  If the
    greedy candidate ever has a larger rank-cost spread than Accelerate's
    unmodified assignment, the original order is retained.
    """

    if len(block) % world_size != 0:
        raise ValueError("balanced block must divide evenly across ranks")
    per_rank = len(block) // world_size
    if per_rank == 0 or per_rank % microbatch != 0:
        raise ValueError("each rank must receive a positive whole number of batches")
    if any(index < 0 or index >= len(costs) for index in block):
        raise IndexError("balanced block contains an out-of-range dataset index")
    if any(not math.isfinite(float(costs[index])) or float(costs[index]) < 0 for index in block):
        raise ValueError("sampler costs must be finite and non-negative")

    original = accelerate_rank_assignments(
        block, world_size=world_size, microbatch=microbatch
    )
    positions = {index: position for position, index in enumerate(block)}
    if len(positions) != len(block):
        raise ValueError("optimizer block must not contain duplicate dataset indices")
    descending = sorted(
        block,
        key=lambda index: (-float(costs[index]), positions[index]),
    )
    buckets: list[list[int]] = [[] for _ in range(world_size)]
    loads = [0.0] * world_size
    for index in descending:
        eligible = [rank for rank in range(world_size) if len(buckets[rank]) < per_rank]
        rank = min(eligible, key=lambda item: (loads[item], len(buckets[item]), item))
        buckets[rank].append(index)
        loads[rank] += float(costs[index])

    for bucket in buckets:
        bucket.sort(key=positions.__getitem__)
    candidate = tuple(tuple(bucket) for bucket in buckets)
    if rank_cost_spread(candidate, costs) >= rank_cost_spread(original, costs):
        return list(block)

    interleaved: list[int] = []
    for offset in range(0, per_rank, microbatch):
        for rank in range(world_size):
            interleaved.extend(buckets[rank][offset : offset + microbatch])
    if sorted(interleaved) != sorted(block):
        raise AssertionError("cost balancing changed optimizer-block membership")
    return interleaved


class CostBalancedRandomSampler(Sampler[int]):
    """Seeded random sampler with optimizer-block-local rank balancing.

    This intentionally does not subclass ``RandomSampler``.  Accelerate must
    leave it intact instead of replacing it with ``SeedableRandomSampler``.
    """

    def __init__(
        self,
        data_source: Sequence[Any],
        *,
        costs: Sequence[int | float],
        seed: int,
        world_size: int = 4,
        microbatch: int = 2,
        gradient_accumulation_steps: int = 4,
    ) -> None:
        if len(costs) != len(data_source):
            raise ValueError("sampler costs must match the dataset length")
        if world_size <= 0 or microbatch <= 0 or gradient_accumulation_steps <= 0:
            raise ValueError("sampler topology values must be positive")
        self.data_source = data_source
        self.costs = tuple(float(value) for value in costs)
        if any(not math.isfinite(value) or value < 0 for value in self.costs):
            raise ValueError("sampler costs must be finite and non-negative")
        self.seed = int(seed)
        self.world_size = int(world_size)
        self.microbatch = int(microbatch)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.optimizer_block_size = (
            self.world_size * self.microbatch * self.gradient_accumulation_steps
        )
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("sampler epoch cannot be negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        epoch = self.epoch
        original = seeded_random_order(len(self), seed=self.seed, epoch=epoch)
        balanced: list[int] = []
        full_stop = len(original) - len(original) % self.optimizer_block_size
        for offset in range(0, full_stop, self.optimizer_block_size):
            balanced.extend(
                greedy_balance_block(
                    original[offset : offset + self.optimizer_block_size],
                    self.costs,
                    world_size=self.world_size,
                    microbatch=self.microbatch,
                )
            )

        # Preserve the original tail membership.  Balance it only when it is a
        # complete equal number of dataloader batches for all ranks (10K with
        # fast4 has a 16-row, four-rows-per-rank final partial update).
        tail = original[full_stop:]
        if tail and len(tail) % (self.world_size * self.microbatch) == 0:
            balanced.extend(
                greedy_balance_block(
                    tail,
                    self.costs,
                    world_size=self.world_size,
                    microbatch=self.microbatch,
                )
            )
        else:
            balanced.extend(tail)
        if len(balanced) != len(original):
            raise AssertionError("cost-balanced sampler changed epoch length")
        self.epoch = epoch + 1
        return iter(balanced)


__all__ = [
    "COST_PROXY_VERSION",
    "CostBalancedRandomSampler",
    "accelerate_rank_assignments",
    "greedy_balance_block",
    "load_teacher_gate_costs",
    "rank_cost_spread",
    "seeded_random_order",
    "teacher_gate_cost",
]
