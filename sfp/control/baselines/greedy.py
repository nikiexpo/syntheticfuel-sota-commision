"""Greedy baseline: run everything, flat out, whenever there is power.

The comparison the challenge brief names explicitly. It is not a caricature --
it is what an unsupervised plant with a simple power-follow controller does, and
on a clear day in July it works perfectly well.

Its failures are all failures of *timing*:

    it calcines whenever the sun is out, burning sorbent cycles for methane it
    did not need to make today

    it runs the contactor fans flat out, paying 453 kWh/tCO2 where 126 would
    have done, because fan power goes as the cube of airflow

    it pushes the electrolyser to 100 % load, giving up the part-load
    efficiency peak at ~53 %

None of those are visible to a controller that only asks "is there power right
now?". The gap is what the planning layer is for.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sfp.control.base import ControlContext, Controller
from sfp.sim.bus import CONTROLLABLE, Request


class GreedyController(Controller):
    """Power-follow: every setpoint at maximum, all the time."""

    name = "greedy"
    description = "Run every subsystem flat out whenever power is available (brief's baseline)"

    def __init__(self, use_battery_to_run: bool = True) -> None:
        self.use_battery_to_run = use_battery_to_run

    def reset(self, context: ControlContext) -> None:
        super().reset(context)
        self._battery = context.plant["battery"]

    def act(
        self,
        t: float,
        state: Mapping[str, np.ndarray],
        measurement: Mapping[str, Any],
        forecast: Any = None,
    ) -> Request:
        # Ask for everything; the bus sheds what cannot be served. No
        # anticipation at all -- feasibility is left entirely to the safety layer.
        return Request(
            setpoints={key: 1.0 for key in CONTROLLABLE},
            enables={key: 1.0 for key in CONTROLLABLE},
            battery_charge_W=self._battery.max_power_W,
            battery_discharge_W=(
                self._battery.max_power_W if self.use_battery_to_run else 0.0
            ),
            curtail_fraction=0.0,
        )
