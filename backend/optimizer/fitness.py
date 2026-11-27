"""Fitness of a response plan (Jira GC-25).

    F(x) = (1/S) * sum_s B_s(x) + lambda * D(x) + P(x)        (lower is better)

B_s(x)  burned cells after the simulation with random seed s
D(x)    total crew travel time of all units, in minutes
P(x)    penalty: penalty_per_fix for every unit that had to be repaired or that
        shares a cell with another unit (discourages particles from sitting in
        invalid areas or stacking units)

Every plan is evaluated on the SAME S seeds (common random numbers), so two
plans are compared on identical fire randomness. This removes the luck of a
single run and makes the comparison fair.

The fire model is any callable  fire_model(scenario, suppress_from, seed) -> bool
mask of burned cells  (see response.py for suppress_from).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .response import DecodedPlan, GridContext, decode, empty_schedule, suppression_schedule

FireModel = Callable[[dict, np.ndarray, int], np.ndarray]


@dataclass
class Fitness:
    ctx: GridContext
    fire_model: FireModel
    scenario: dict
    seeds: tuple[int, ...] = (11, 22, 33)
    travel_weight: float = 0.02          # lambda; tune so lambda*D is ~1-5 % of a typical burned area
    penalty_per_fix: float = 5.0
    n_evals: int = field(default=0, init=False)

    def burned_cells(self, suppress_from: np.ndarray) -> float:
        """Mean burned cells over the common seeds."""
        total = 0
        for s in self.seeds:
            total += int(np.count_nonzero(self.fire_model(self.scenario, suppress_from, s)))
        return total / len(self.seeds)

    def evaluate_plan(self, plan: DecodedPlan) -> dict:
        burned = self.burned_cells(suppression_schedule(plan, self.ctx))
        penalty = self.penalty_per_fix * (plan.n_repaired + plan.n_duplicates)
        value = burned + self.travel_weight * plan.total_travel_minutes + penalty
        return {"fitness": value, "burned": burned, "travel": plan.total_travel_minutes, "penalty": penalty}

    def __call__(self, x: np.ndarray) -> float:
        self.n_evals += 1
        return self.evaluate_plan(decode(x, self.ctx))["fitness"]

    def no_response(self) -> float:
        """Burned cells with no units at all (reference for % reduction)."""
        return self.burned_cells(empty_schedule(self.ctx))

    def suggest_travel_weight(self, share: float = 0.03, n_units: int = 4) -> float:
        """A lambda that makes the travel term about `share` of the no-response burned area
        for a plan whose units each need the median valid travel time."""
        median_travel = float(np.median(self.ctx.travel_time[self.ctx.valid]))
        if median_travel <= 0:
            return 0.0
        return share * self.no_response() / (n_units * median_travel)
