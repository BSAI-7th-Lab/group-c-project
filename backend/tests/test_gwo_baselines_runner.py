import numpy as np
import pytest

from optimizer.baselines import greedy_downwind_plan, random_plan
from optimizer.gwo import gwo
from optimizer.pso import pso
from optimizer.response import decode
from optimizer.runner import METHODS, run


def sphere(x):
    return float(np.sum(x ** 2))


def test_gwo_minimises_sphere():
    res = gwo(sphere, np.full(5, -5.0), np.full(5, 5.0), seed=1, iterations=100, patience=None)
    assert res.best_f < 1e-3
    assert np.all(np.diff(res.history) <= 0)


def test_gwo_and_pso_use_the_same_budget():
    lo, hi = np.full(3, -1.0), np.full(3, 1.0)
    a = pso(sphere, lo, hi, seed=0, iterations=20, patience=None)
    b = gwo(sphere, lo, hi, seed=0, iterations=20, patience=None)
    assert a.n_evals == b.n_evals == 30 * 21


def test_random_plan_is_valid(small_ctx):
    x = random_plan(small_ctx, 5, np.random.default_rng(0))
    plan = decode(x, small_ctx)
    assert plan.n_repaired == 0 and len(plan.cells) == 5


def test_greedy_arc_is_downwind(small_ctx):
    x = greedy_downwind_plan(small_ctx, 3, (15, 15), wind_dir_deg=90.0, radius_cells=6)
    cells = decode(x, small_ctx).cells
    assert (cells[:, 1] > 15).all()                  # east of the ignition
    assert cells[1].tolist() == [15, 21]             # middle unit straight downwind


@pytest.mark.parametrize("method", METHODS)
def test_runner_every_method(method, small_ctx, scenario, fire_model):
    row = run(method, scenario, small_ctx, fire_model, n_units=3, seed=1,
              optimiser_kwargs={"iterations": 4} if method in ("pso", "gwo") else None)
    assert row["method"] == method
    assert row["burned_cells"] > 0
    assert len(row["cells"]) == (0 if method == "none" else 3)
    if method in ("pso", "gwo"):
        assert len(row["history"]) >= 2


def test_runner_rejects_unknown_method(small_ctx, scenario, fire_model):
    with pytest.raises(ValueError):
        run("abc", scenario, small_ctx, fire_model)
