"""Greedy baseline: run everything whenever power is available.

This is the comparison the challenge brief names explicitly. It is not a
caricature -- it is what an unsupervised plant with a simple power-follow
controller actually does, and on a clear day in July it performs perfectly well.

Its failures are the interesting part, and they are all failures of *timing*
rather than of instantaneous decision-making:

    it starts on any sunlight, so a broken-cloud morning costs it several cold
    starts and the energy that goes with them

    it drains the battery to keep running into the evening, then has nothing
    left to ride through the next morning's cloud

    it never anticipates: a controller that cannot see tomorrow cannot decide
    that today's marginal kilowatt-hour is worth more tomorrow

Every one of those is a scheduling error that the hierarchical planner in M4 is
designed to avoid, so the gap between this and the autonomous strategy is a
direct measure of what the planning layer is worth.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sfp.control.base import ControlContext, Controller
from sfp.sim.bus import Request


class GreedyController(Controller):
    """Power-follow: take everything the bus can offer, all the time."""

    name = "greedy"
    description = "Run every subsystem whenever power is available (brief's suggested baseline)"

    def __init__(self, use_battery_to_run: bool = True) -> None:
        self.use_battery_to_run = use_battery_to_run
        self._rated_W = 1.0

    def reset(self, context: ControlContext) -> None:
        super().reset(context)
        self._rated_W = context.plant["process"].rated_power_W
        self._battery = context.plant["battery"]
        self._dt = context.dt_s

    def act(
        self,
        t: float,
        state: Mapping[str, np.ndarray],
        measurement: Mapping[str, Any],
        forecast: Any = None,
    ) -> Request:
        pv_available = float(measurement["pv_available_W"])

        offered = pv_available
        if self.use_battery_to_run:
            offered += float(self._battery.max_discharge_power_W(state["battery"], self._dt))

        load = float(np.clip(offered / self._rated_W, 0.0, 1.0))

        # Charge with whatever is left over; the bus caps this to the true
        # surplus, so asking for the full rating costs nothing.
        return Request(
            load_fraction=load,
            battery_charge_W=self._battery.max_power_W,
            battery_discharge_W=self._battery.max_power_W if self.use_battery_to_run else 0.0,
            curtail_fraction=0.0,
        )
