import numpy as np
import pytest

from optimizer.response import (
    NO_SUPPRESSION,
    GridContext,
    decode,
    empty_schedule,
    suppression_schedule,
    wind_vector,
)


def test_valid_cell_is_kept(small_ctx):
    plan = decode(np.array([10.2, 10.7]), small_ctx)
    assert plan.cells.tolist() == [[10, 11]]
    assert plan.n_repaired == 0


def test_non_burnable_cell_is_moved_to_nearest_valid(small_ctx):
    plan = decode(np.array([6.0, 22.0]), small_ctx)  # middle of the rocky patch
    r, c = plan.cells[0]
    assert small_ctx.valid[r, c]
    assert plan.n_repaired == 1
    assert abs(r - 6) + abs(c - 22) <= 3


def test_unreachable_cell_is_moved(small_ctx):
    plan = decode(np.array([15.0, 0.0]), small_ctx)
    r, c = plan.cells[0]
    assert np.isfinite(small_ctx.travel_time[r, c])
    assert c == 3


def test_out_of_grid_values_are_clipped(small_ctx):
    plan = decode(np.array([-50.0, 500.0]), small_ctx)
    r, c = plan.cells[0]
    assert 0 <= r < 30 and 0 <= c < 30 and small_ctx.valid[r, c]


def test_every_random_vector_decodes_to_valid_cells(small_ctx):
    rng = np.random.default_rng(0)
    lower, upper = small_ctx.bounds(4)
    for _ in range(200):
        plan = decode(rng.uniform(lower - 5, upper + 5), small_ctx)
        assert small_ctx.valid[plan.cells[:, 0], plan.cells[:, 1]].all()


def test_duplicates_and_arrival(small_ctx):
    plan = decode(np.array([10, 10, 10, 10]), small_ctx)
    assert plan.n_duplicates == 1
    assert plan.arrival_steps.tolist() == [4, 4]          # 20 min / 5 min per step
    assert plan.total_travel_minutes == pytest.approx(40.0)


def test_schedule_marks_3x3_block_from_arrival(small_ctx):
    sched = suppression_schedule(decode(np.array([10, 10]), small_ctx), small_ctx)
    block = sched[9:12, 9:12]
    assert (block == 4).all()
    assert (sched == 4).sum() == 9
    assert sched[0, 0] == NO_SUPPRESSION


def test_schedule_skips_non_burnable_and_keeps_earliest(small_ctx):
    travel = small_ctx.travel_time.copy()
    travel[10, 11] = 5.0
    ctx = GridContext(small_ctx.burnable, travel, minutes_per_step=5.0)
    sched = suppression_schedule(decode(np.array([10, 10, 10, 11]), ctx), ctx)
    assert sched[10, 10] == 1          # unit 2 arrives at step 1 and overlaps
    near_rock = suppression_schedule(decode(np.array([8, 21]), ctx), ctx)
    assert near_rock[7, 21] == NO_SUPPRESSION   # rock cell is never "suppressed"


def test_empty_schedule(small_ctx):
    assert (empty_schedule(small_ctx) == NO_SUPPRESSION).all()


def test_wind_vector_directions():
    assert np.allclose(wind_vector(0), (-1, 0))    # north = up = smaller row
    assert np.allclose(wind_vector(90), (0, 1))    # east = larger column


def test_grid_without_valid_cells_is_rejected():
    with pytest.raises(ValueError):
        GridContext(np.zeros((3, 3), bool), np.zeros((3, 3)))
