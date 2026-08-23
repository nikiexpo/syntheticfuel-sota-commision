"""Rule-based baseline: a credible PLC-style supervisory controller.

This is the strawman worth beating. It is roughly what a competent controls
engineer would ship without an optimiser: state-of-charge bands, start/stop
hysteresis, and a reserve held back for the morning. It fixes the greedy
controller's two worst habits -- chasing clouds and flattening the battery at
dusk -- using nothing but local rules and a clock.

What it still cannot do is anticipate. It has no forecast, so it cannot know
that tomorrow is cloudy and today's surplus should have been banked, nor that
tomorrow is clear and the battery may safely be spent tonight. That gap is
precisely the value of the planning layer, and keeping this baseline strong is
what makes that comparison mean something.

The rules, in order:

    1. never start below `soc_start`, and once running never continue below
       `soc_stop` -- a hysteresis band, so a passing cloud cannot cause a stop
    2. hold a reserve that grows towards evening, so the plant wakes up with
       enough charge to start the next morning without waiting for the sun
    3. modulate load with available power rather than running flat out, so the
       battery is cycled less
    4. once stopped, stay stopped for `min_downtime_s` -- start-ups cost money
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sfp.control.base import ControlContext, Controller
from sfp.sim.bus import Request


class RuleBasedController(Controller):
    """SoC-banded power-follow with start/stop hysteresis and an evening reserve."""

    name = "rule-based"
    description = "PLC-style: SoC bands, start/stop hysteresis, evening reserve"

    def __init__(
        self,
        soc_start: float = 0.35,
        soc_stop: float = 0.18,
        soc_charge_target: float = 0.85,
        soc_charge_priority: float = 0.55,
        min_runtime_s: float = 3600.0,
        min_downtime_s: float = 1800.0,
        evening_reserve: float = 0.30,
    ) -> None:
        self.soc_start = soc_start
        self.soc_stop = soc_stop
        self.soc_charge_target = soc_charge_target
        self.soc_charge_priority = soc_charge_priority
        self.min_runtime_s = min_runtime_s
        self.min_downtime_s = min_downtime_s
        self.evening_reserve = evening_reserve

        self._running = False
        self._last_transition_s = -1e9

    def reset(self, context: ControlContext) -> None:
        super().reset(context)
        self._process = context.plant["process"]
        self._battery = context.plant["battery"]
        self._rated_W = self._process.rated_power_W
        self._dt = context.dt_s
        self._running = False
        self._last_transition_s = -1e9

    # --- helpers ----------------------------------------------------------
    def _reserve_soc(self, measurement: Mapping[str, Any]) -> float:
        """Charge to hold back for tomorrow morning, rising as the sun drops.

        Uses only the instantaneous solar elevation -- no forecast -- so this
        stays an honest no-foresight baseline. Around solar noon the reserve is
        zero; in the last hours of daylight it climbs to `evening_reserve`.
        """
        cos_zenith = float(measurement.get("cos_zenith", 0.0))
        daylight = np.clip(cos_zenith / 0.35, 0.0, 1.0)
        return self.evening_reserve * (1.0 - daylight)

    def _commit(self, t: float, want_running: bool) -> bool:
        """Apply minimum run-time and down-time to a desired on/off state."""
        elapsed = t - self._last_transition_s
        if want_running == self._running:
            return self._running
        dwell = self.min_runtime_s if self._running else self.min_downtime_s
        if elapsed < dwell:
            return self._running
        self._running = want_running
        self._last_transition_s = t
        return self._running

    # --- control ----------------------------------------------------------
    def act(
        self,
        t: float,
        state: Mapping[str, np.ndarray],
        measurement: Mapping[str, Any],
        forecast: Any = None,
    ) -> Request:
        pv_available = float(measurement["pv_available_W"])
        soc = float(state["battery"][0])
        reserve = self._reserve_soc(measurement)

        # battery power we are willing to spend on the process, above the reserve
        spendable_soc = max(soc - max(self.soc_stop, reserve), 0.0)
        usable_J = float(self._battery.usable_energy_J(state["battery"][1]))
        spendable_W = min(
            self._battery.max_power_W,
            spendable_soc * usable_J * self._battery.p.eta_discharge / 3600.0,
        )

        offered = pv_available + spendable_W

        # start/stop decision with hysteresis
        can_run = offered >= self._process.min_power_W
        if self._running:
            want = can_run and soc > self.soc_stop
        else:
            want = can_run and soc > self.soc_start
        running = self._commit(t, want)

        if not running:
            # Stay energised only if there is power to spare for the parasitics;
            # otherwise let the plant go dark rather than bleed the battery.
            keep_warm = pv_available > self._process.parasitic_power_W(state["process"]) or (
                soc > self.soc_start
            )
            return Request(
                load_fraction=0.0,
                battery_charge_W=self._battery.max_power_W,
                battery_discharge_W=0.0,
                enable=1.0 if keep_warm else 0.0,
            )

        # Rule 3: below the priority threshold, solar goes to the battery first
        # and the process gets only what is left. Without this the process
        # always outbids the battery and the pack never refills, which is the
        # failure mode the greedy controller has by construction.
        pv_for_process = pv_available
        charge_request = 0.0
        if soc < self.soc_charge_priority:
            charge_request = min(
                self._battery.max_power_W,
                max(pv_available - self._process.min_power_W, 0.0),
            )
            pv_for_process = max(pv_available - charge_request, 0.0)
        elif soc < self.soc_charge_target:
            # top up with genuine surplus only
            charge_request = max(pv_available - self._rated_W, 0.0)

        offered_to_process = pv_for_process + spendable_W
        load = float(np.clip(offered_to_process / self._rated_W, 0.0, 1.0))
        if load < self._process.p.min_load_fraction:
            load = self._process.p.min_load_fraction

        return Request(
            load_fraction=load,
            battery_charge_W=charge_request,
            battery_discharge_W=spendable_W,
            enable=1.0,
        )

    def diagnostics(self) -> dict[str, float]:
        return {"rule_running": float(self._running)}
