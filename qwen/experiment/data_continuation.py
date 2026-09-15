"""Explicit remaining-row view and final-window normalization for all Phase2/3 arms.

HF4.52's resume skip retains the pre-skip steps_in_epoch, which can miss the
final odd microbatch. Give it only the remaining rows and ignore_data_skip=True.
Optimizer/scheduler/global_step/RNG still restore normally. No library patch.
"""
from math import ceil
from torch.utils.data import Dataset


class OrderedRemainingRows(Dataset):
    def __init__(self, rows, order, start_step, effective_batch=32):
        assert len(rows) == len(order) and sorted(order) == list(range(len(rows)))
        assert isinstance(start_step, int) and 0 <= start_step * effective_batch < len(rows)
        self.rows = [rows[i] for i in order[start_step * effective_batch:]]
        self.consumed_before = start_step * effective_batch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def accumulation_multiplier(step, total_rows=10000, microbatch=4, world_size=4, ga=2):
    """Compensate HF's fixed /GA only for an incomplete final accumulation window.

    Each microbatch has the same per-rank number of samples. The final 16-row
    global window has one microbatch per rank, so scale by2 before HF divides
    by2. All complete 32-row windows are unchanged.
    """
    global_micro = microbatch * world_size
    assert total_rows % global_micro == 0, 'unequal per-rank tail requires a separate contract'
    updates = ceil(total_rows / (global_micro * ga))
    assert isinstance(step, int) and 1 <= step <= updates
    remaining = total_rows - (step-1) * global_micro * ga
    actual = min(ga, remaining // global_micro)
    assert 1 <= actual <= ga
    return ga / actual
