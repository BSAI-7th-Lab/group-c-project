"""One interface for every method, used by the experiments (Jira GC-26 -> GC-38).

    result = run("pso", scenario, ctx, fire_model, n_units=4, seed=0)

method   "pso" | "gwo" | "greedy" | "random" | "none"
scenario dict with at least: ignition (row, col), wind_dir_deg (direction the
         wind blows toward) plus whatever the fire model needs
seed     controls the optimiser / random placement only. The fire randomness
         always uses the Fitness object's common seeds, so every method is
         scored on the same fires.

Returns a dict ready to become one row of results.csv.
"""

from __future__ import annotations

import time

import numpy as np

from .baselines import greedy_downwind_plan, random_plan
from .fitness import FireModel, Fitness
from .gwo import gwo
from .pso import pso
from .response import GridContext, decode

METHODS = ("pso", "gwo", "greedy", "random", "none")


def run(
    method: str,
    scenario: dict,
    ctx: GridContext,
    fire_model: FireModel,
    n_units: int = 4,
    seed: int = 0,
    fitness_kwargs: dict | None = None,
    optimiser_kwargs: dict | None = None,
) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    fit = Fitness(ctx, fire_model, scenario, **(fitness_kwargs or {}))
    opt_kw = dict(optimiser_kwargs or {})
    lower, upper = ctx.bounds(n_units)
    history: list[float] = []
    start = time.perf_counter()

    if method == "none":
        burned = fit.no_response()
        row = {"fitness": burned, "burned": burned, "travel": 0.0, "penalty": 0.0}
        cells = np.empty((0, 2), dtype=int)
    else:
        if method == "pso":
            res = pso(fit, lower, upper, seed=seed, **opt_kw)
            x, history = res.best_x, res.history
        elif method == "gwo":
            res = gwo(fit, lower, upper, seed=seed, **opt_kw)
            x, history = res.best_x, res.history
        elif method == "random":
            x = random_plan(ctx, n_units, np.random.default_rng(seed))
        else:  # greedy
            x = greedy_downwind_plan(ctx, n_units, scenario["ignition"], scenario["wind_dir_deg"])
        plan = decode(x, ctx)
        row = fit.evaluate_plan(plan)
        cells = plan.cells

    return {
        "method": method,
        "seed": seed,
        "n_units": n_units if method != "none" else 0,
        "burned_cells": row["burned"],
        "fitness": row["fitness"],
        "travel_minutes": row["travel"],
        "penalty": row["penalty"],
        "n_evals": fit.n_evals,
        "runtime_s": time.perf_counter() - start,
        "cells": cells.tolist(),
        "history": history,
    }
