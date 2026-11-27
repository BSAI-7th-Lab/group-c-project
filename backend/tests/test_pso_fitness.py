import numpy as np

from optimizer.fitness import Fitness
from optimizer.pso import pso
from optimizer.response import decode


def sphere(x):
    return float(np.sum(x ** 2))


def test_pso_minimises_sphere():
    lo, hi = np.full(5, -5.0), np.full(5, 5.0)
    res = pso(sphere, lo, hi, seed=1, iterations=100, patience=None)
    assert res.best_f < 1e-3
    assert np.all(np.diff(res.history) <= 0)          # best value never gets worse
    assert res.n_evals == 30 * 101


def test_pso_respects_bounds_and_seed():
    lo, hi = np.array([2.0, 2.0]), np.array([3.0, 3.0])
    a = pso(sphere, lo, hi, seed=7, iterations=20)
    b = pso(sphere, lo, hi, seed=7, iterations=20)
    assert np.all(a.positions >= lo) and np.all(a.positions <= hi)
    assert np.allclose(a.best_x, [2.0, 2.0], atol=1e-6)   # optimum is the corner
    assert a.best_f == b.best_f                           # same seed, same result


def test_pso_early_stop():
    res = pso(lambda x: 1.0, np.zeros(2), np.ones(2), seed=0, iterations=60, patience=5)
    assert len(res.history) == 1 + 5


def test_pso_warm_start_uses_given_swarm():
    start = np.array([[0.1, 0.1], [0.2, -0.1]])
    res = pso(sphere, np.full(2, -1.0), np.full(2, 1.0), seed=0, iterations=3, init_positions=start)
    assert len(res.positions) == 2


def test_fitness_uses_common_seeds(small_ctx, scenario, fire_model):
    fit = Fitness(small_ctx, fire_model, scenario)
    x = np.array([15, 18, 12, 20, 18, 20])
    assert fit(x) == fit(x)                       # same plan, same seeds -> same value
    assert fit.n_evals == 2


def test_fitness_units_reduce_burned_area(small_ctx, scenario, fire_model):
    fit = Fitness(small_ctx, fire_model, scenario)
    free = fit.no_response()
    wall = np.array([15, 17, 14, 17, 16, 17, 13, 17])   # units just downwind of the ignition
    blocked = fit.evaluate_plan(decode(wall, small_ctx))
    assert blocked["burned"] < free


def test_fitness_penalises_repair(small_ctx, scenario, fire_model):
    fit = Fitness(small_ctx, fire_model, scenario)
    assert fit.evaluate_plan(decode(np.array([25.0, 25.0]), small_ctx))["penalty"] == 0.0
    assert fit.evaluate_plan(decode(np.array([6.0, 22.0]), small_ctx))["penalty"] == 5.0    # repaired
    assert fit.evaluate_plan(decode(np.array([25, 25, 25, 25]), small_ctx))["penalty"] == 5.0  # stacked


def test_pso_beats_random_placement(small_ctx, scenario, fire_model):
    fit = Fitness(small_ctx, fire_model, scenario)
    lo, hi = small_ctx.bounds(3)
    res = pso(fit, lo, hi, n_particles=12, iterations=15, seed=3)
    rng = np.random.default_rng(3)
    random_vals = [fit(rng.uniform(lo, hi)) for _ in range(12)]
    assert res.best_f < np.median(random_vals)
    assert res.best_f < fit.no_response()
