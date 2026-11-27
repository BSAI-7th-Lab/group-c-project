"""Response model for firefighting-unit placement (Jira GC-16).

A response plan places K units on the park grid. During the search a plan is a
continuous vector

    x = (r1, c1, r2, c2, ..., rK, cK)

so that PSO and GWO can move it freely. Before a plan is scored it is decoded:
each (r, c) pair is rounded to a grid cell, kept inside the grid, and, if that
cell cannot be used (non-burnable, or not reachable by a crew), moved to the
nearest usable cell. This "repair" step means every particle always stands for
a feasible plan.

Each unit, once it arrives, suppresses a 3 x 3 block of cells around its target
cell. The arrival step comes from the crew travel-time map, so a unit sent deep
into the hills starts working later.

Interface agreed with the fire model (Zain, GC-13/GC-14):

    suppress_from : int array, same shape as the grid
        suppress_from[r, c] = first simulation step from which cell (r, c) is
        suppressed (cannot ignite). NO_SUPPRESSION where no unit works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy import ndimage

NO_SUPPRESSION = np.iinfo(np.int32).max


@dataclass
class GridContext:
    """The parts of the park grid the response model needs.

    burnable      : bool array, True where vegetation can burn.
    travel_time   : float array, minutes for a crew to reach each cell from the
                    nearest fire station (np.inf where unreachable).
    minutes_per_step : how many minutes one fire-model time step represents.
    block_radius  : 1 gives a 3 x 3 suppressed block per unit.
    """

    burnable: np.ndarray
    travel_time: np.ndarray
    minutes_per_step: float = 5.0
    block_radius: int = 1
    _nearest: tuple[np.ndarray, np.ndarray] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.burnable = np.asarray(self.burnable, dtype=bool)
        self.travel_time = np.asarray(self.travel_time, dtype=float)
        if self.burnable.shape != self.travel_time.shape:
            raise ValueError("burnable and travel_time must have the same shape")
        if self.minutes_per_step <= 0:
            raise ValueError("minutes_per_step must be positive")
        if not self.valid.any():
            raise ValueError("the grid has no cell that is both burnable and reachable")

    @property
    def shape(self) -> tuple[int, int]:
        return self.burnable.shape

    @property
    def valid(self) -> np.ndarray:
        """Cells where a unit may be placed: burnable and reachable."""
        return self.burnable & np.isfinite(self.travel_time)

    def nearest_valid(self) -> tuple[np.ndarray, np.ndarray]:
        """For every cell, the row/column of the nearest valid cell (cached)."""
        if self._nearest is None:
            _, idx = ndimage.distance_transform_edt(~self.valid, return_indices=True)
            self._nearest = (idx[0], idx[1])
        return self._nearest

    def bounds(self, n_units: int) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds of a plan vector for n_units units."""
        rows, cols = self.shape
        lower = np.zeros(2 * n_units)
        upper = np.tile([rows - 1, cols - 1], n_units).astype(float)
        return lower, upper


@dataclass
class DecodedPlan:
    cells: np.ndarray          # (K, 2) int, the repaired target cell of each unit
    n_repaired: int            # units whose rounded cell had to be moved
    n_duplicates: int          # units sharing a target cell with an earlier unit
    arrival_steps: np.ndarray  # (K,) int, step at which each unit starts work
    total_travel_minutes: float


def decode(x: np.ndarray, ctx: GridContext) -> DecodedPlan:
    """Turn a continuous plan vector into K valid target cells."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size % 2:
        raise ValueError("a plan vector must be 1-D with an even length (r, c pairs)")
    rows, cols = ctx.shape
    pairs = x.reshape(-1, 2)
    r = np.clip(np.rint(pairs[:, 0]), 0, rows - 1).astype(int)
    c = np.clip(np.rint(pairs[:, 1]), 0, cols - 1).astype(int)

    valid = ctx.valid
    needs_repair = ~valid[r, c]
    if needs_repair.any():
        nr, nc = ctx.nearest_valid()
        r_fixed, c_fixed = nr[r, c], nc[r, c]
        r = np.where(needs_repair, r_fixed, r)
        c = np.where(needs_repair, c_fixed, c)

    cells = np.stack([r, c], axis=1)
    _, first_index = np.unique(cells, axis=0, return_index=True)
    n_duplicates = len(cells) - len(first_index)

    travel = ctx.travel_time[r, c]
    arrival = np.ceil(travel / ctx.minutes_per_step).astype(int)
    return DecodedPlan(
        cells=cells,
        n_repaired=int(needs_repair.sum()),
        n_duplicates=int(n_duplicates),
        arrival_steps=arrival,
        total_travel_minutes=float(travel.sum()),
    )


def suppression_schedule(plan: DecodedPlan, ctx: GridContext) -> np.ndarray:
    """Build the suppress_from grid for the fire model.

    Each unit suppresses the burnable cells in a (2*block_radius+1)^2 block
    around its target from its arrival step. Where blocks overlap, the earliest
    arrival wins.
    """
    rows, cols = ctx.shape
    out = np.full(ctx.shape, NO_SUPPRESSION, dtype=np.int64)
    k = ctx.block_radius
    for (r, c), step in zip(plan.cells, plan.arrival_steps):
        r0, r1 = max(r - k, 0), min(r + k + 1, rows)
        c0, c1 = max(c - k, 0), min(c + k + 1, cols)
        block = out[r0:r1, c0:c1]
        np.minimum(block, step, out=block)
    out[~ctx.burnable] = NO_SUPPRESSION
    return out


def empty_schedule(ctx: GridContext) -> np.ndarray:
    """No units at all (the 'no response' baseline)."""
    return np.full(ctx.shape, NO_SUPPRESSION, dtype=np.int64)


def wind_vector(direction_deg: float) -> tuple[float, float]:
    """Grid (d_row, d_col) unit vector for a wind blowing TOWARD direction_deg.

    0 deg = north = towards row 0, 90 deg = east = increasing column. Open-Meteo
    reports the direction the wind comes FROM; add 180 deg before calling this.
    """
    rad = math.radians(direction_deg)
    return -math.cos(rad), math.sin(rad)
