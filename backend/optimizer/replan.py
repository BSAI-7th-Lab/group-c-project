"""Adaptive re-planning after a wind change, with warm-start PSO (Jira GC-35).

Situation: a plan for K units was found with the forecast wind. At step
t_change the wind turns. By then some units are already working at their
target (arrival step <= t_change); they stay where they are. The others are
still on the way and can be sent somewhere better for the new wind.

Re-planning therefore searches only over the units that have not arrived:

    z = (r_i, c_i  for every unit i not yet arrived)

and scores the full plan (fixed units + re-planned units) on the scenario
with the wind change, using the same fitness F as the first plan.

Arrival of a re-planned unit
    kept on its old target   -> keeps its old arrival step
    sent to a new target     -> t_change + ceil(travel_time(new) / minutes_per_step)
This treats a redirected crew as re-dispatched from the station at t_change,
which is conservative: it can only make moving look worse, never better. When
the ACO routes with fire-proximity cost land (GC-28) this can be swapped for the
real redirect time.

Warm start
    The new swarm is built from the old one instead of from scratch:
      particle 0          the current plan (so re-planning is never worse
                          than keeping the plan, on the same seeds)
      next ~2/3           the best personal bests of the first run, slightly
                          jittered, projected onto the free units
      the rest            random plans, to keep diversity for the new wind
    A cold start (all random) is kept for comparison in the experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .fitness import FireModel, Fitness
from .pso import PSOResult, pso
from .response import DecodedPlan, GridContext, decode


@dataclass
class Situation:
    """The state of the response when the wind changes."""

    plan: DecodedPlan          # the plan being carried out
    t_change: int              # step at which the wind turns
    free: np.ndarray           # (K,) bool, True for units not yet arrived

    @property
    def free_index(self) -> np.ndarray:
        return np.flatnonzero(self.free)

    @property
    def n_free(self) -> int:
        return int(self.free.sum())


def situation_at(plan: DecodedPlan, t_change: int) -> Situation:
    """Units that have not reached their target by t_change are free to move."""
    if t_change < 0:
        raise ValueError("t_change must be >= 0")
    return Situation(plan=plan, t_change=int(t_change), free=plan.arrival_steps > t_change)


def free_dims(free: np.ndarray) -> np.ndarray:
    """Indices into a full plan vector (r1, c1, ..., rK, cK) for the free units."""
    idx = np.flatnonzero(free)
    return np.stack([2 * idx, 2 * idx + 1], axis=1).ravel()


@dataclass
class ReplanFitness:
    """F for the free units only; fixed units are part of every plan."""

    fitness: Fitness           # built on the scenario WITH the wind change
    situation: Situation

    def full_plan(self, z: np.ndarray) -> DecodedPlan:
        sit, ctx = self.situation, self.fitness.ctx
        old = sit.plan
        new_free = decode(np.asarray(z, dtype=float), ctx)

        cells = old.cells.copy()
        arrival = old.arrival_steps.copy()
        idx = sit.free_index
        cells[idx] = new_free.cells
        moved = np.any(new_free.cells != old.cells[idx], axis=1)
        redirect = sit.t_change + np.ceil(
            ctx.travel_time[new_free.cells[:, 0], new_free.cells[:, 1]] / ctx.minutes_per_step
        ).astype(int)
        arrival[idx] = np.where(moved, redirect, old.arrival_steps[idx])

        _, first = np.unique(cells, axis=0, return_index=True)
        return DecodedPlan(
            cells=cells,
            n_repaired=new_free.n_repaired,
            n_duplicates=int(len(cells) - len(first)),
            arrival_steps=arrival,
            total_travel_minutes=float(ctx.travel_time[cells[:, 0], cells[:, 1]].sum()),
        )

    def evaluate(self, z: np.ndarray) -> dict:
        return self.fitness.evaluate_plan(self.full_plan(z))

    def __call__(self, z: np.ndarray) -> float:
        self.fitness.n_evals += 1
        return self.evaluate(z)["fitness"]


def warm_start_swarm(
    previous: PSOResult,
    current_z: np.ndarray,
    free: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_particles: int = 30,
    random_fraction: float = 1 / 3,
    jitter: float = 0.03,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Initial swarm for re-planning, seeded from the previous PSO run."""
    rng = rng or np.random.default_rng()
    span = upper - lower
    dims = free_dims(free)
    n_random = int(round(random_fraction * n_particles))
    n_elite = max(n_particles - 1 - n_random, 0)

    order = np.argsort(previous.pbest_f)
    # best personal bests first, then the final positions; repeat if the old swarm was smaller
    pool = np.vstack([previous.pbest_x[order], previous.positions])[:, dims]
    elites = pool[np.arange(n_elite) % len(pool)]
    elites = elites + rng.normal(0.0, jitter, elites.shape) * span

    randoms = lower + rng.random((n_particles - 1 - len(elites), len(lower))) * span
    swarm = np.vstack([np.asarray(current_z, dtype=float)[None, :], elites, randoms])
    return np.clip(swarm, lower, upper)


@dataclass
class ReplanResult:
    cells: np.ndarray            # (K, 2) final target of every unit
    moved: np.ndarray            # (K,) bool, unit sent to a new target
    free: np.ndarray             # (K,) bool, unit was allowed to move
    arrival_steps: np.ndarray
    fitness: float
    burned: float
    history: list[float]
    n_evals: int
    runtime_s: float


def replan(
    situation: Situation,
    scenario_changed: dict,
    ctx: GridContext,
    fire_model: FireModel,
    previous: PSOResult | None = None,
    warm: bool = True,
    seed: int | None = None,
    fitness_kwargs: dict | None = None,
    pso_kwargs: dict | None = None,
) -> ReplanResult:
    """Re-plan the units that have not arrived, for the scenario with the wind change.

    pso_kwargs default to a shorter budget than the first plan (re-planning has
    to be quick): 30 particles, 25 iterations, early stop after 8 without gain.
    """
    start = time.perf_counter()
    fit = Fitness(ctx, fire_model, scenario_changed, **(fitness_kwargs or {}))
    rfit = ReplanFitness(fit, situation)
    old = situation.plan
    current_z = old.cells[situation.free_index].astype(float).ravel()

    if situation.n_free == 0:
        row = rfit.evaluate(current_z)
        return ReplanResult(old.cells.copy(), np.zeros(len(old.cells), bool), situation.free.copy(),
                            old.arrival_steps.copy(), row["fitness"], row["burned"], [row["fitness"]],
                            1, time.perf_counter() - start)

    kw = {"n_particles": 30, "iterations": 25, "patience": 8}
    kw.update(pso_kwargs or {})
    lower, upper = ctx.bounds(situation.n_free)
    init = None
    if warm:
        if previous is None:
            raise ValueError("a warm start needs the previous PSO result")
        init = warm_start_swarm(previous, current_z, situation.free, lower, upper,
                                n_particles=kw["n_particles"], rng=np.random.default_rng(seed))
    res = pso(rfit, lower, upper, seed=seed, init_positions=init, **kw)

    plan = rfit.full_plan(res.best_x)
    row = fit.evaluate_plan(plan)
    moved = np.any(plan.cells != old.cells, axis=1)
    return ReplanResult(plan.cells, moved, situation.free.copy(), plan.arrival_steps, row["fitness"],
                        row["burned"], res.history, fit.n_evals, time.perf_counter() - start)


def run_wind_change(
    scenario: dict,
    ctx: GridContext,
    fire_model: FireModel,
    t_change: int,
    new_wind_dir_deg: float,
    n_units: int = 4,
    seed: int = 0,
    fitness_kwargs: dict | None = None,
    pso_kwargs: dict | None = None,
    replan_kwargs: dict | None = None,
    with_cold: bool = True,
) -> dict:
    """One wind-change experiment: fixed plan vs warm re-plan (vs cold re-plan).

    1. PSO plans with the forecast wind (no change known).
    2. The wind turns at t_change; the fixed plan is scored on the changed scenario.
    3. The units not yet arrived are re-planned, warm-started from step 1's swarm.
    4. Optionally the same re-plan from a random swarm, same budget (cold).
    Returns a dict ready to become one row of the re-planning results table.
    """
    fit0 = Fitness(ctx, fire_model, scenario, **(fitness_kwargs or {}))
    lower, upper = ctx.bounds(n_units)
    first = pso(fit0, lower, upper, seed=seed, **(pso_kwargs or {}))
    plan = decode(first.best_x, ctx)

    changed = dict(scenario, wind_change={"step": int(t_change), "wind_dir_deg": float(new_wind_dir_deg)})
    sit = situation_at(plan, t_change)
    fixed = Fitness(ctx, fire_model, changed, **(fitness_kwargs or {})).evaluate_plan(plan)

    warm = replan(sit, changed, ctx, fire_model, previous=first, warm=True, seed=seed,
                  fitness_kwargs=fitness_kwargs, pso_kwargs=replan_kwargs)
    row = {
        "seed": seed,
        "t_change": int(t_change),
        "old_wind_deg": float(scenario["wind_dir_deg"]),
        "new_wind_deg": float(new_wind_dir_deg),
        "n_units": n_units,
        "n_free": sit.n_free,
        "fixed_burned": fixed["burned"],
        "fixed_fitness": fixed["fitness"],
        "warm_burned": warm.burned,
        "warm_fitness": warm.fitness,
        "warm_evals": warm.n_evals,
        "warm_moved": int(warm.moved.sum()),
        "warm_history": warm.history,
        "reduction_vs_fixed_pct": (
            100.0 * (fixed["burned"] - warm.burned) / fixed["burned"] if fixed["burned"] else 0.0
        ),
    }
    if with_cold:
        cold = replan(sit, changed, ctx, fire_model, warm=False, seed=seed,
                      fitness_kwargs=fitness_kwargs, pso_kwargs=replan_kwargs)
        row.update({"cold_burned": cold.burned, "cold_fitness": cold.fitness,
                    "cold_evals": cold.n_evals, "cold_history": cold.history})
    return row
