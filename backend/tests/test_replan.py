import numpy as np
import pytest

from optimizer.fitness import Fitness
from optimizer.pso import pso
from optimizer.replan import (
    ReplanFitness,
    free_dims,
    replan,
    run_wind_change,
    situation_at,
    warm_start_swarm,
)
from optimizer.response import GridContext, decode, wind_direction_at


@pytest.fixture
def graded_ctx():
    """Travel time grows with the column: 5 min per column, so arrival step == column."""
    rows, cols = 30, 30
    burnable = np.ones((rows, cols), dtype=bool)
    travel = np.tile(np.arange(cols, dtype=float) * 5.0, (rows, 1))
    travel[:, 0] = np.inf
    return GridContext(burnable=burnable, travel_time=travel, minutes_per_step=5.0)


@pytest.fixture
def graded_scenario(graded_ctx):
    return {"burnable": graded_ctx.burnable, "ignition": (15, 12), "wind_dir_deg": 90.0,
            "wind_strength": 0.7, "p_base": 0.45, "steps": 24}


def test_wind_direction_at_switches_at_the_change_step():
    sc = {"wind_dir_deg": 90.0, "wind_change": {"step": 5, "wind_dir_deg": 180.0}}
    assert [wind_direction_at(sc, t) for t in (0, 4, 5, 9)] == [90.0, 90.0, 180.0, 180.0]
    assert wind_direction_at({"wind_dir_deg": 45.0}, 100) == 45.0


def test_situation_marks_units_not_yet_arrived(graded_ctx):
    plan = decode(np.array([10, 4, 10, 20, 12, 8]), graded_ctx)   # arrival steps 4, 20, 8
    sit = situation_at(plan, t_change=8)
    assert plan.arrival_steps.tolist() == [4, 20, 8]
    assert sit.free.tolist() == [False, True, False]
    assert free_dims(sit.free).tolist() == [2, 3]


def test_full_plan_keeps_fixed_units_and_delays_redirected_ones(graded_ctx, graded_scenario, fire_model):
    plan = decode(np.array([10, 4, 10, 20, 20, 25]), graded_ctx)  # arrivals 4, 20, 25
    sit = situation_at(plan, t_change=6)                           # units 1 and 2 free
    rfit = ReplanFitness(Fitness(graded_ctx, fire_model, graded_scenario), sit)

    full = rfit.full_plan(np.array([10, 20, 5, 9]))   # unit 1 stays, unit 2 moves to (5, 9)
    assert full.cells.tolist() == [[10, 4], [10, 20], [5, 9]]
    assert full.arrival_steps.tolist() == [4, 20, 6 + 9]           # kept, kept, t_change + ceil(45/5)


def test_warm_start_swarm_starts_from_the_current_plan(small_ctx):
    lo, hi = small_ctx.bounds(3)
    prev = pso(lambda x: float(np.sum(x)), lo, hi, seed=0, iterations=3, patience=None)
    free = np.array([True, False, True])
    lo2, hi2 = small_ctx.bounds(2)
    z0 = np.array([4.0, 5.0, 6.0, 7.0])
    swarm = warm_start_swarm(prev, z0, free, lo2, hi2, n_particles=30, rng=np.random.default_rng(1))
    assert swarm.shape == (30, 4)
    assert np.array_equal(swarm[0], z0)
    assert np.all(swarm >= lo2) and np.all(swarm <= hi2)
    # elites come from the old swarm's best, projected onto the free units' coordinates
    best_free = prev.pbest_x[np.argmin(prev.pbest_f)][free_dims(free)]
    assert np.linalg.norm(swarm[1] - best_free) < 0.1 * np.linalg.norm(hi2 - lo2)


def test_replan_is_never_worse_than_the_fixed_plan(graded_ctx, graded_scenario, fire_model):
    lo, hi = graded_ctx.bounds(4)
    fit0 = Fitness(graded_ctx, fire_model, graded_scenario)
    first = pso(fit0, lo, hi, seed=3, iterations=8)
    plan = decode(first.best_x, graded_ctx)
    changed = dict(graded_scenario, wind_change={"step": 6, "wind_dir_deg": 0.0})
    sit = situation_at(plan, 6)
    assert sit.n_free > 0

    fixed = Fitness(graded_ctx, fire_model, changed).evaluate_plan(plan)
    res = replan(sit, changed, graded_ctx, fire_model, previous=first, seed=3,
                 pso_kwargs={"iterations": 6})
    assert res.history[0] <= fixed["fitness"] + 1e-9          # current plan is in the swarm
    assert res.fitness <= fixed["fitness"] + 1e-9
    assert np.all(res.cells[~sit.free] == plan.cells[~sit.free])   # arrived units never move
    assert not res.moved[~sit.free].any()


def test_replan_with_every_unit_arrived_returns_the_plan(small_ctx, scenario, fire_model):
    plan = decode(np.array([10, 10, 20, 20]), small_ctx)       # travel 20 min -> arrive at step 4
    sit = situation_at(plan, 10)
    changed = dict(scenario, wind_change={"step": 10, "wind_dir_deg": 270.0})
    res = replan(sit, changed, small_ctx, fire_model, previous=None)
    assert sit.n_free == 0 and res.n_evals == 1
    assert np.array_equal(res.cells, plan.cells)


def test_warm_start_needs_the_previous_result(graded_ctx, graded_scenario, fire_model):
    plan = decode(np.array([10, 25]), graded_ctx)
    with pytest.raises(ValueError):
        replan(situation_at(plan, 3), graded_scenario, graded_ctx, fire_model, previous=None, warm=True)


def test_run_wind_change_row(graded_ctx, graded_scenario, fire_model):
    row = run_wind_change(graded_scenario, graded_ctx, fire_model, t_change=6, new_wind_dir_deg=0.0,
                          n_units=3, seed=2, pso_kwargs={"iterations": 4},
                          replan_kwargs={"iterations": 3})
    for key in ("fixed_burned", "warm_burned", "cold_burned", "warm_history", "cold_history",
                "reduction_vs_fixed_pct", "n_free"):
        assert key in row
    assert row["warm_fitness"] <= row["fixed_fitness"] + 1e-9
    assert row["warm_evals"] <= 30 * 4 + 1 and row["cold_evals"] <= 30 * 4 + 1
