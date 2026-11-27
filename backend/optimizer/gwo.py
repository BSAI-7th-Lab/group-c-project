"""Grey Wolf Optimizer, written from scratch (Jira GC-26).

Mirjalili, Mirjalili & Lewis (2014). The three best wolves (alpha, beta,
delta) lead the pack; every wolf moves to the average of three positions,
one pulled toward each leader:

    A = 2a*r1 - a,   C = 2*r2,   a falls linearly from 2 to 0
    X1 = X_alpha - A1 * |C1 * X_alpha - X|      (same for beta, delta)
    X  = (X1 + X2 + X3) / 3

|A| > 1 makes wolves search away from the leaders (exploration, early);
|A| < 1 makes them close in (exploitation, late).

For a fair comparison it uses the same population size, iterations, early
stop and therefore the same number of fitness evaluations as pso().
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .pso import PSOResult


def gwo(
    objective: Callable[[np.ndarray], float],
    lower: np.ndarray,
    upper: np.ndarray,
    n_wolves: int = 30,
    iterations: int = 60,
    patience: int | None = 15,
    seed: int | None = None,
) -> PSOResult:
    rng = np.random.default_rng(seed)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    dim = lower.size

    x = lower + rng.random((n_wolves, dim)) * (upper - lower)
    f = np.array([objective(p) for p in x])
    n_evals = n_wolves

    best_x, best_f = x[np.argmin(f)].copy(), float(f.min())
    history = [best_f]
    stall = 0

    for t in range(iterations):
        order = np.argsort(f)
        leaders = x[order[:3]].copy()          # alpha, beta, delta
        a = 2.0 - 2.0 * t / max(iterations - 1, 1)

        new = np.zeros_like(x)
        for leader in leaders:
            r1 = rng.random((n_wolves, dim))
            r2 = rng.random((n_wolves, dim))
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            new += leader - A * np.abs(C * leader - x)
        x = np.clip(new / 3.0, lower, upper)

        f = np.array([objective(p) for p in x])
        n_evals += n_wolves
        i = int(np.argmin(f))
        if f[i] < best_f:
            best_x, best_f = x[i].copy(), float(f[i])
            stall = 0
        else:
            stall += 1
        history.append(best_f)
        if patience is not None and stall >= patience:
            break

    return PSOResult(best_x, best_f, history, n_evals, x, np.zeros_like(x), x.copy(), f.copy())
