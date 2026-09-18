"""Rule-based baseline: a credible PLC-style supervisory controller.

The strawman worth beating: roughly what a competent controls engineer would
ship without an optimiser. Per-subsystem interlocks, buffer-level bands,
start/stop hysteresis, an evening reserve. It knows the plant's local physics;
what it cannot do is *anticipate*.

**Sabatier** -- keep it lit. Nearly free to run (8 kW), expensive to relight
(31 kWh plus catalyst wear), so it runs whenever both buffers hold enough and
its feed follows the scarcer one.

**Contactor** -- cheapest flow that meets demand, not full flow. Fan power goes
as airflow cubed. Stop when the sorbent is nearly saturated.

**Calciner** -- the flexible load, and the one handled worst. Runs on strong
solar or a comfortable battery, given feedstock and CO2 headroom; temperature
hysteresis stops it chasing clouds. It cannot hold the kiln warm because
tomorrow is sunny.

**Electrolyser** -- surplus after the others, biased toward its part-load
efficiency peak. Stops when the H2 tank is full.

**Battery** -- charges from surplus, holds a reserve that grows as the sun
drops, from solar elevation alone so this stays an honest no-forecast baseline.

Every rule is local and instantaneous: none can trade today against tomorrow.
That is the space the economic planner occupies.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sfp.control.base import ControlContext, Controller
from sfp.sim.bus import Request


class RuleBasedController(Controller):
    """Per-subsystem interlocks, buffer bands and hysteresis. No foresight."""

    name = "rule-based"
    description = "PLC-style: buffer bands, hysteresis, evening reserve, no forecast"

    def __init__(
        self,
        soc_start: float = 0.30,
        soc_stop: float = 0.15,
        soc_calciner_on: float = 0.55,
        soc_calciner_off: float = 0.30,
        soc_charge_target: float = 0.90,
        evening_reserve: float = 0.35,
        sorbent_stop_loading: float = 0.90,
        sorbent_resume_loading: float = 0.75,
        co2_stop_fill: float = 0.92,
        h2_stop_fill: float = 0.95,
        min_dwell_s: float = 1800.0,
    ) -> None:
        self.soc_start = soc_start
        self.soc_stop = soc_stop
        self.soc_calciner_on = soc_calciner_on
        self.soc_calciner_off = soc_calciner_off
        self.soc_charge_target = soc_charge_target
        self.evening_reserve = evening_reserve
        self.sorbent_stop_loading = sorbent_stop_loading
        self.sorbent_resume_loading = sorbent_resume_loading
        self.co2_stop_fill = co2_stop_fill
        self.h2_stop_fill = h2_stop_fill
        self.min_dwell_s = min_dwell_s

        self._running: dict[str, bool] = {}
        self._last_change: dict[str, float] = {}

    def reset(self, context: ControlContext) -> None:
        super().reset(context)
        plant = context.plant
        self._plant = plant
        self._battery = plant["battery"]
        self._contactor = plant["contactor"]
        self._calciner = plant["calciner"]
        self._electrolyser = plant["electrolyser"]
        self._sabatier = plant["sabatier"]
        self._dt = context.dt_s

        # Capture target: the reactor's stoichiometric CO2 demand at full feed.
        # Matching demand rather than maximising capture is the fan-power rule.
        self._co2_demand_mol_s = float(self._sabatier.p.co2_feed_max_mol_s)

        self._running = {"calciner": False, "sabatier": False, "electrolyser": False}
        self._last_change = dict.fromkeys(self._running, -1e9)

    # --- helpers ----------------------------------------------------------
    def _hysteresis(self, key: str, t: float, want: bool) -> bool:
        """Apply a minimum dwell time to any on/off transition."""
        if want == self._running[key]:
            return want
        if t - self._last_change[key] < self.min_dwell_s:
            return self._running[key]
        self._running[key] = want
        self._last_change[key] = t
        return want

    def _reserve_soc(self, measurement: Mapping[str, Any]) -> float:
        """Charge held back for tomorrow morning, rising as the sun drops."""
        cos_zenith = float(measurement.get("cos_zenith", 0.0))
        daylight = np.clip(cos_zenith / 0.35, 0.0, 1.0)
        return self.evening_reserve * (1.0 - daylight)

    def _hours_until_dawn(self, t: float, measurement: Mapping[str, Any]) -> float:
        """Rough hours of darkness remaining. Clock and geometry only, no forecast.

        During daylight this returns the hours until the *following* dawn, so the
        reactor's rationing already accounts for the night ahead rather than
        discovering it at sunset.
        """
        seconds_into_day = t % 86400.0
        if float(measurement.get("cos_zenith", 0.0)) > 0.0:
            # daylight: assume a 12 h night follows the remainder of the day
            return 12.0 + max(0.0, (86400.0 - seconds_into_day) / 3600.0) * 0.5
        return 8.0

    def _parasitic_floor_W(self, state, enables) -> float:
        """Power needed to keep the currently enabled subsystems merely alive."""
        total = 0.0
        for key in ("contactor", "calciner", "electrolyser", "sabatier"):
            if enables.get(key, 0.0) < 0.5:
                continue
            total += float(self._plant[key].parasitic_power_W(state[key], 1.0, {}))
        return total

    # --- control ----------------------------------------------------------
    def act(
        self,
        t: float,
        state: Mapping[str, np.ndarray],
        measurement: Mapping[str, Any],
        forecast: Any = None,
    ) -> Request:
        pv = float(measurement.get("pv_available_W", 0.0))
        soc = float(state["battery"][0])
        reserve = self._reserve_soc(measurement)

        loading = float(measurement.get("solids_loading", 0.0))
        caco3 = float(measurement.get("solids_n_caco3_mol", 0.0))
        co2_fill = float(measurement.get("gas_co2_fill", 0.0))
        h2_fill = float(measurement.get("gas_h2_fill", 0.0))
        co2_avail = float(measurement.get("gas_co2_available_mol", 0.0))
        h2_avail = float(measurement.get("gas_h2_available_mol", 0.0))
        kiln_T = float(state["calciner"][0])

        setpoints: dict[str, float] = {}
        enables: dict[str, float] = {}

        # --- Sabatier: throttle, do not stop -------------------------------
        # Feed follows the scarcer reactant: 4:1 stoichiometry means the short
        # one throttles the reactor however much of the other is banked. The
        # reactor stays lit to a very low buffer level and simply runs slower --
        # turning down is nearly free, turning off costs a 31 kWh relight and a
        # catalyst thermal cycle.
        feed_capacity = min(co2_avail, h2_avail / 4.0)
        want_sabatier = feed_capacity > 5.0
        run_sabatier = self._hysteresis("sabatier", t, want_sabatier)
        enables["sabatier"] = 1.0 if run_sabatier else 0.0
        if run_sabatier:
            # Ration the buffers over the hours until dawn. At full feed the
            # reactor eats 1.0 mol/s of H2 against the electrolyser's 0.82, so
            # an unrationed reactor empties the tanks by dusk -- the opposite of
            # what the buffers are for. Hours-to-dawn needs only a clock.
            hours_left = self._hours_until_dawn(t, measurement)
            rationed = feed_capacity / max(hours_left * 3600.0, 1800.0)

            # In daylight the reactor may also consume at the rate hydrogen is
            # arriving: that throughput never touches the reserve. Rationing
            # alone would leave the plant under-running all afternoon.
            live_h2 = float(measurement.get("r_electrolysis_h2_mol_s", 0.0))
            flow_through = live_h2 / 4.0
            sustainable = max(rationed, flow_through)

            fraction = sustainable / self._sabatier.p.co2_feed_max_mol_s
            setpoints["sabatier"] = float(np.clip(fraction, 0.10, 1.0))
        else:
            setpoints["sabatier"] = 0.0

        # Stoichiometric balance of the two upstream chains: whichever side is
        # behind gets priority, or the plant stalls with one buffer full, the
        # other empty and the reactor idle.
        h2_equivalent = h2_avail / 4.0
        h2_poor = h2_equivalent < 0.8 * max(co2_avail, 1.0)
        co2_poor = co2_avail < 0.8 * max(h2_equivalent, 1.0)

        # --- Contactor: cheapest flow that meets demand --------------------
        if loading >= self.sorbent_stop_loading:
            enables["contactor"] = 0.0
            setpoints["contactor"] = 0.0
        else:
            target = self._co2_demand_mol_s
            flow = self._contactor.flow_for_capture_rate(target, measurement)
            enables["contactor"] = 1.0
            setpoints["contactor"] = float(np.clip(flow, 0.0, 1.0))

        # --- Calciner: the flexible load -----------------------------------
        solar_strong = pv > 0.45 * self._calciner.heater_power_rated_W
        battery_comfortable = soc > self.soc_calciner_on
        has_feed = caco3 > 5.0 * self._calciner.p.feedstock_reference_mol
        has_room = co2_fill < self.co2_stop_fill

        want_calciner = (solar_strong or battery_comfortable) and has_feed and has_room
        if soc < self.soc_calciner_off and not solar_strong:
            want_calciner = False
        # if the plant is carbon-poor relative to its hydrogen, the kiln earns
        # its power even when solar is only moderate
        if co2_poor and has_feed and has_room and soc > self.soc_calciner_off:
            want_calciner = True
        run_calciner = self._hysteresis("calciner", t, want_calciner)

        enables["calciner"] = 1.0 if run_calciner else 0.0
        if run_calciner:
            # ease off once at temperature so the heater is not fighting the
            # over-temperature interlock
            if kiln_T > self._calciner.p.temperature_target_K:
                setpoints["calciner"] = 0.35
            else:
                setpoints["calciner"] = 1.0
        else:
            setpoints["calciner"] = 0.0

        # --- Electrolyser: surplus, biased to its efficiency peak ----------
        want_electrolyser = (
            h2_fill < self.h2_stop_fill and (pv > 0.0 or soc > max(self.soc_start, reserve))
        )
        run_electrolyser = self._hysteresis("electrolyser", t, want_electrolyser)
        enables["electrolyser"] = 1.0 if run_electrolyser else 0.0
        if run_electrolyser:
            spare = max(pv - self._calciner.power_for_setpoint(
                state["calciner"], setpoints["calciner"], enables["calciner"]
            ), 0.0)
            by_power = self._electrolyser.fraction_for_power(state["electrolyser"], spare)
            peak = self._electrolyser.best_efficiency_fraction()
            setpoints["electrolyser"] = float(np.clip(max(by_power, 0.0), 0.0, 1.0))

            # Efficiency only matters while energy is scarce: if the
            # alternative for a kilowatt-hour is curtailment, poor conversion
            # beats none. Imminent curtailment is visible without a forecast --
            # battery nearly full and surplus left after every other load.
            # Holding the peak regardless curtailed 6.8 % of the array.
            battery_nearly_full = soc > self.soc_charge_target - 0.05
            surplus_at_risk = spare > self._electrolyser.rated_power_W() and battery_nearly_full
            if surplus_at_risk or h2_poor:
                setpoints["electrolyser"] = 1.0
            elif pv > 1.5 * self._electrolyser.rated_power_W() and not battery_nearly_full:
                setpoints["electrolyser"] = float(peak)
        else:
            setpoints["electrolyser"] = 0.0

        # --- Battery -------------------------------------------------------
        charge_request = self._battery.max_power_W if soc < self.soc_charge_target else 0.0
        spendable = max(soc - max(self.soc_stop, reserve), 0.0)
        usable_J = float(self._battery.usable_energy_J(state["battery"][1]))
        discharge_request = min(
            self._battery.max_power_W,
            spendable * usable_J * self._battery.p.eta_discharge / 3600.0,
        )

        # The reserve rations *discretionary* load only. Starving the
        # parasitics of an enabled subsystem to protect the battery is a false
        # economy -- the reactor's 8 kW is what protects 31 kWh of relight.
        # Above the hard SoC floor, keep-alive always wins.
        if soc > self.soc_stop:
            discharge_request = max(
                discharge_request, self._parasitic_floor_W(state, enables) * 1.15
            )

        return Request(
            setpoints=setpoints,
            enables=enables,
            battery_charge_W=charge_request,
            battery_discharge_W=discharge_request,
        )

    def diagnostics(self) -> dict[str, float]:
        return {f"rule_running_{k}": float(v) for k, v in self._running.items()}
