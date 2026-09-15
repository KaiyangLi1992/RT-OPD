"""Frozen K-student grid; pure helpers also used by CPU qualification tests."""
from __future__ import annotations
from .common import ROOT, read

LRS = (1e-5, 2.5e-5, 5e-5, 7.5e-5)
ALPHAS = (0., .25, .5, 1.)
STEPS = (432, 626)
COUNTS = {"mmau": 1000, "mmau_full": 9000, "mmar": 1000, "adqa": 1577}


def grid_config():
    c = read(ROOT / "configs/ke_grid16.json")
    assert c["campaign_id"] == "rtopd_qwen_release_20260915"
    assert c["start_step"] == 0 and c["scheduler_horizon"] == 626
    assert c["world_size"] * c["microbatch"] * c["gradient_accumulation_steps"] == 32
    assert c["evaluation"]["datasets"] == COUNTS
    return c


def order_paths(seed):
    return [ROOT / f"assets/grid16/order_s{seed}_epoch{epoch}.json" for epoch in range(2)]


def row_window(step, microbatch_index, rank):
    """HF makes 625 microbatches per epoch, not 626. Reset slots after each tail."""
    if not 1 <= step <= 626 or rank not in range(4):
        raise ValueError("Invalid training position")
    epoch, local_step = divmod(step - 1, 313)
    slot = (microbatch_index % 625) % 2
    offset = local_step * 32 + slot * 16 + rank * 4
    return epoch, local_step + 1, slot, offset

