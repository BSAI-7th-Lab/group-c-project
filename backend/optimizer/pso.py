"""Particle Swarm Optimization, written from scratch (Jira GC-25).

    v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    x <- x + v

Settings from the proposal: 30 particles, 60 iterations, c1 = c2 = 2.0,
inertia w decreasing linearly from 0.9 to 0.4 (Shi & Eberhart 1998), and a
velocity limit of 20 % of each dimension's range so particles cannot jump
across the park in one step. The search stops early when the best value has
not improved for `patience` iterations.

Minimises `objective`. The same function is used for unit placement, and can
also be used for the risk-map weights and fire-model calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PSOResult:
    best_x: np.ndarray
    best_f: float
    history: list[float]          # best value after each iteration (convergence curve)
    n_evals: int
    positions: np.ndarray         # final swarm, for warm-start re-planning
    velocities: np.ndarray
    pbest_x: np.ndarray
    pbest_f: np.ndarray


def pso(
    objective: Callable[[np.ndarray], float],
    lower: np.ndarray,
    upper: np.ndarray,
    n_particles: int = 30,
    iterations: int = 60,
    c1: float = 2.0,
    c2: float = 2.0,
    w_start: float = 0.9,
    w_end: float = 0.4,
    vmax_fraction: float = 0.2,
    patience: int | None = 15,
    seed: int | None = None,
    init_positions: np.ndarray | None = None,
) -> PSOResult:
    """Minimise objective over the box [lower, upper].

    init_positions lets a new run start from a previous swarm (warm start).
    """
    rng = np.random.default_rng(seed)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    dim = lower.size
    span = upper - lower
    vmax = vmax_fraction * span

    if init_positions is not None:
        x = np.clip(np.array(init_positions, dtype=float), lower, upper)
        n_particles = len(x)
    else:
        x = lower + rng.random((n_particles, dim)) * span
    v = rng.uniform(-vmax, vmax, size=(n_particles, dim)) * 0.1

    f = np.array([objective(p) for p in x])
    n_evals = n_particles
    pbest_x, pbest_f = x.copy(), f.copy()
    g = int(np.argmin(pbest_f))
    gbest_x, gbest_f = pbest_x[g].copy(), float(pbest_f[g])
    history = [gbest_f]
    stall = 0

    for t in range(iterations):
        w = w_start - (w_start - w_end) * t / max(iterations - 1, 1)
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        v = np.clip(v, -vmax, vmax)
        x = np.clip(x + v, lower, upper)

        f = np.array([objective(p) for p in x])
        n_evals += n_particles
        better = f < pbest_f                      # minimisation: smaller is better
        pbest_x[better] = x[better]
        pbest_f[better] = f[better]

        g = int(np.argmin(pbest_f))
        if pbest_f[g] < gbest_f:
            gbest_x, gbest_f = pbest_x[g].copy(), float(pbest_f[g])
            stall = 0
        else:
            stall += 1
        history.append(gbest_f)
        if patience is not None and stall >= patience:
            break

    return PSOResult(gbest_x, gbest_f, history, n_evals, x, v, pbest_x, pbest_f)
