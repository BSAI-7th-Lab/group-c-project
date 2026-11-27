"""Non-intelligent baselines the swarm methods must beat (Jira GC-26).

none    no response: the fire spreads freely
random  K units on random reachable, burnable cells
greedy  K units in an arc on the downwind side of the ignition point, as a
        person might place them by intuition

Each returns a plan vector (r1, c1, ..., rK, cK), so it is decoded, repaired
and scored exactly like a PSO or GWO plan.
"""

from __future__ import annotations

import math

import numpy as np

from .response import GridContext


def random_plan(ctx: GridContext, n_units: int, rng: np.random.Generator) -> np.ndarray:
    rows, cols = np.nonzero(ctx.valid)
    pick = rng.choice(len(rows), size=n_units, replace=len(rows) < n_units)
    return np.column_stack([rows[pick], cols[pick]]).astype(float).ravel()


def greedy_downwind_plan(
    ctx: GridContext,
    n_units: int,
    ignition: tuple[int, int],
    wind_dir_deg: float,
    radius_cells: float = 6.0,
    arc_deg: float = 120.0,
) -> np.ndarray:
    """Spread the units evenly over an arc of `arc_deg` centred on the downwind
    direction, `radius_cells` from the ignition. wind_dir_deg is the direction
    the wind blows TOWARD (0 = north/up, 90 = east/right)."""
    r0, c0 = ignition
    if n_units == 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-arc_deg / 2, arc_deg / 2, n_units)
    points = []
    for off in offsets:
        ang = math.radians(wind_dir_deg + off)
        points += [r0 - radius_cells * math.cos(ang), c0 + radius_cells * math.sin(ang)]
    return np.array(points)
