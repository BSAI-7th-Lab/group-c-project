"""Shared test fixtures.

`toy_fire_model` is a deliberately simple stand-in so the optimiser can be
tested before the real cellular-automaton model (GC-13/GC-14) is ready. It is
NOT the project's fire model: it spreads to the 8 neighbours with a fixed
probability, a little more easily downwind, and respects the suppression
schedule. It follows the same call signature the real model will expose.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optimizer.response import GridContext, wind_direction_at, wind_vector  # noqa: E402

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def toy_fire_model(scenario, suppress_from, seed):
    """Return the burned-out mask after scenario['steps'] steps."""
    burnable = scenario["burnable"]
    rows, cols = burnable.shape
    rng = np.random.default_rng(seed)
    burning = np.zeros_like(burnable)
    burned = np.zeros_like(burnable)
    r0, c0 = scenario["ignition"]
    burning[r0, c0] = True
    for t in range(scenario["steps"]):
        wr, wc = wind_vector(wind_direction_at(scenario, t))   # supports a wind change (GC-35)
        new = np.zeros_like(burning)
        for dr, dc in NEIGHBOURS:
            align = (dr * wr + dc * wc) / np.hypot(dr, dc)
            p = scenario["p_base"] * (1.0 + scenario["wind_strength"] * align)
            src = np.roll(np.roll(burning, dr, axis=0), dc, axis=1)
            # undo wrap-around at the edges
            if dr > 0:
                src[:dr, :] = False
            if dr < 0:
                src[dr:, :] = False
            if dc > 0:
                src[:, :dc] = False
            if dc < 0:
                src[:, dc:] = False
            new |= src & (rng.random(burnable.shape) < p)
        can_ignite = burnable & ~burning & ~burned & (suppress_from > t)
        burned |= burning
        burning = new & can_ignite
    return burned | burning


@pytest.fixture
def small_ctx():
    rows, cols = 30, 30
    burnable = np.ones((rows, cols), dtype=bool)
    burnable[5:8, 20:25] = False          # a rocky patch
    travel = np.full((rows, cols), 20.0)
    travel[:, :3] = np.inf                # a strip no crew can reach
    return GridContext(burnable=burnable, travel_time=travel, minutes_per_step=5.0)


@pytest.fixture
def scenario(small_ctx):
    return {
        "burnable": small_ctx.burnable,
        "ignition": (15, 15),
        "wind_dir_deg": 90.0,   # blowing toward the east (increasing column)
        "wind_strength": 0.6,
        "p_base": 0.45,
        "steps": 25,
    }


@pytest.fixture
def fire_model():
    return toy_fire_model
