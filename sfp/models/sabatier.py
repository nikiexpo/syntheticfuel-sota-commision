"""Sabatier methanation reactor: CO2 + 4 H2 -> CH4 + 2 H2O, dH = -165 kJ/mol.

Lumped from the three-zone axial model of Moioli, Gallandat & Zuttel (2019). A
scheduling model needs only the aggregate consequence of that axial profile: a
hotspot temperature confined to a window, and a conversion set by how close that
temperature lets the reactor approach equilibrium.

It is **nearly free to run once lit** -- strongly exothermic and self-sustaining,
so the draw is just the recycle blower and condenser, ~8 kW against the
electrolyser's 328 kW. That makes it the natural night-time load, drawing down
hydrogen and CO2 banked during the day.

Starting it costs ~31 kWh of electric preheat and ages the catalyst by one
thermal cycle. The default policy is light it once and keep it lit; the
planner's job is to notice when that is wrong.

Two competing temperature pressures:

    hotter -> faster kinetics, so a closer approach to equilibrium
    hotter -> a *lower* equilibrium conversion (the reaction is exothermic)
           -> and accelerating catalyst sintering above ~400 degC

giving an interior optimum around 300-350 degC, where constraint and economics
happen to agree.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sfp.models import mathx as mx
from sfp.models.base import Signal, Subsystem, VectorSpec
from sfp.units import DH_SABATIER, M_CH4, R_GAS


class SabatierReactor(Subsystem):
    """Fixed-bed methanation with hotspot temperature and catalyst ageing."""

    name = "sabatier"

    @property
    def states(self) -> VectorSpec:
        return VectorSpec(
            (
                Signal("temperature_K", "K", "hotspot temperature",
                       lower=250.0, upper=self.p.temperature_max_K),
                Signal("catalyst_activity", "-", "fractional catalyst activity",
                       lower=0.0, upper=1.0),
                Signal("ch4_kg", "kg", "cumulative methane produced", lower=0.0),
            )
        )

    # --- coupling ---------------------------------------------------------
    @property
    def requires(self) -> tuple[str, ...]:
        return ("gas_co2_available_mol", "gas_h2_available_mol", "water_available_kg")

    @property
    def provides(self) -> tuple[str, ...]:
        return ("r_sabatier_co2_mol_s", "r_sabatier_h2_mol_s")

    @property
    def inputs(self) -> VectorSpec:
        return VectorSpec(
            (
                Signal("feed_fraction", "-", "CO2 feed as a fraction of maximum",
                       lower=0.0, upper=1.0),
                Signal("enable", "-", "1 = lit and held warm, 0 = shut down and cooling",
                       lower=0.0, upper=1.0),
            )
        )

    # --- gates ------------------------------------------------------------
    def _enable_gate(self, u):
        return mx.smooth_step(mx.clip(u[1], 0.0, 1.0) - 0.5, width=0.05)

    def _feed_fraction(self, u):
        return mx.clip(u[0], 0.0, 1.0) * self._enable_gate(u)

    # --- chemistry --------------------------------------------------------
    def equilibrium_conversion(self, temperature_K):
        """Logistic fit to the equilibrium CO2 conversion curve.

        Falls with temperature because the reaction is exothermic -- this is the
        term that stops the controller simply running the reactor as hot as the
        catalyst allows.
        """
        z = (temperature_K - self.p.equilibrium_T50_K) / self.p.equilibrium_width_K
        return 1.0 / (1.0 + mx.exp(z))

    def damkohler(self, temperature_K, activity, feed_fraction):
        """Da = k(T) * activity * residence time.

        Residence time is inversely proportional to feed rate, so Da rises as the
        feed is turned down. The 0.02 floor keeps it finite at zero feed, where
        the conversion is irrelevant because nothing is flowing.
        """
        arrhenius = mx.exp(
            -self.p.kinetic_activation_J_mol
            / R_GAS
            * (1.0 / mx.fmax(temperature_K, 200.0) - 1.0 / self.p.kinetic_reference_T_K)
        )
        return (
            self.p.kinetic_damkohler_ref
            * arrhenius
            * mx.fmax(activity, 0.0)
            / mx.fmax(feed_fraction, 0.02)
        )

    def conversion(self, x, u):
        """Actual CO2 conversion: equilibrium ceiling times kinetic approach."""
        feed = self._feed_fraction(u)
        x_eq = self.equilibrium_conversion(x[0])
        approach = 1.0 - mx.exp(-self.damkohler(x[0], x[1], feed))
        return x_eq * approach

    def feed_availability(self, w: Mapping[str, Any]):
        """Fraction of the commanded feed the buffers can actually supply.

        The 4:1 stoichiometry is a hard coupling: whichever of CO2 or H2 runs
        short throttles the reactor, no matter how much of the other is banked.
        This is the constraint that forces the two upstream chains to be
        co-scheduled rather than optimised separately.
        """
        co2 = w.get("gas_co2_available_mol", 0.0)
        h2 = w.get("gas_h2_available_mol", 0.0)
        water_ok = mx.smooth_step(w.get("water_available_kg", 1.0) - 1.0, width=1.0)

        # taper over the last few moles rather than cutting off abruptly
        co2_ok = mx.smooth_clip(co2 / 50.0, 0.0, 1.0, eps=1e-3)
        h2_ok = mx.smooth_clip(h2 / 200.0, 0.0, 1.0, eps=1e-3)
        return mx.fmin(co2_ok, h2_ok) * water_ok

    def co2_rate_mol_s(self, x, u, w: Mapping[str, Any]):
        """CO2 actually converted, mol/s."""
        feed = self._feed_fraction(u) * self.p.co2_feed_max_mol_s
        return feed * self.conversion(x, u) * self.feed_availability(w)

    # --- thermal ----------------------------------------------------------
    def reaction_heat_W(self, rate_mol_s):
        """Heat released, W (positive)."""
        return rate_mol_s * (-DH_SABATIER)

    def preheater_power_W(self, x, u):
        """Local light-off controller: electric heat until the reactor is hot."""
        enabled = self._enable_gate(u)
        error = self.p.temperature_target_K - x[0]
        demand = self.p.preheat_gain_W_K * mx.fmax(error, 0.0)
        return mx.fmin(demand, self.p.preheater_power_kw * 1e3) * enabled

    def cooling_power_W(self, x):
        """Local heat-removal loop protecting the catalyst."""
        error = x[0] - self.p.cooling_setpoint_K
        demand = self.p.cooling_gain_W_K * mx.fmax(error, 0.0)
        return mx.fmin(demand, self.p.cooling_power_max_W)

    def rhs(self, t, x, u, w: Mapping[str, Any]):
        rate = self.co2_rate_mol_s(x, u, w)
        feed_total = self._feed_fraction(u) * self.p.co2_feed_max_mol_s * 5.0  # CO2 + 4 H2

        heat_in = self.reaction_heat_W(rate) + self.preheater_power_W(x, u)
        cooling = self.cooling_power_W(x)
        temp_amb_K = w.get("temp_air", 20.0) + 273.15
        ambient = self.p.ambient_UA_W_K * (x[0] - temp_amb_K)
        sensible = (
            feed_total
            * self.p.feed_heat_capacity_J_mol_K
            * mx.fmax(x[0] - self.p.feed_temperature_K, 0.0)
        )

        d_temperature = (heat_in - cooling - ambient - sensible) / self.p.thermal_capacity_J_K

        # thermal sintering: exponential in temperature above the reference
        deactivation = (
            self.p.catalyst_deactivation_1_s
            * mx.exp(
                (x[0] - self.p.catalyst_deactivation_T_ref_K)
                / self.p.catalyst_deactivation_T_scale_K
            )
            * mx.fmax(x[1], 0.0)
        )
        d_activity = -deactivation

        d_ch4 = rate * M_CH4

        return mx.vertcat(d_temperature, d_activity, d_ch4)

    def outputs(self, t, x, u, w: Mapping[str, Any]) -> dict[str, Any]:
        enabled = self._enable_gate(u)
        rate = self.co2_rate_mol_s(x, u, w)
        preheat = self.preheater_power_W(x, u)
        aux = self.p.auxiliary_power_kw * 1e3 * enabled

        return {
            "power_electrical_W": preheat + aux,
            # coupling signals: consumed by the gas buffer and the water tank
            "r_sabatier_co2_mol_s": rate,
            "r_sabatier_h2_mol_s": 4.0 * rate,
            "sabatier_temperature_K": x[0],
            "sabatier_temperature_C": x[0] - 273.15,
            "sabatier_catalyst_activity": x[1],
            "sabatier_conversion": self.conversion(x, u),
            "sabatier_equilibrium_conversion": self.equilibrium_conversion(x[0]),
            "sabatier_feed_fraction": self._feed_fraction(u),
            "sabatier_feed_availability": self.feed_availability(w),
            "sabatier_preheat_W": preheat,
            "sabatier_aux_W": aux,
            "sabatier_reaction_heat_W": self.reaction_heat_W(rate),
            "sabatier_cooling_W": self.cooling_power_W(x),
            "sabatier_enabled": enabled,
            "sabatier_is_lit": mx.smooth_step(x[0] - self.p.temperature_ignition_K, width=5.0),
            "ch4_rate_kg_s": rate * M_CH4,
            "ch4_total_kg": x[2],
        }

    def initial_state(self) -> np.ndarray:
        return np.array(
            [
                self.p.temperature_initial_K,
                self.p.catalyst_activity_initial,
                0.0,
            ],
            dtype=float,
        )

    # --- helpers for the dispatcher and the planner -----------------------
    def power_for_feed(self, x, feed_fraction: float, enable: float = 1.0) -> float:
        return float(
            self.outputs(0.0, x, np.array([feed_fraction, enable], dtype=float), {})[
                "power_electrical_W"
            ]
        )

    def standby_power_W(self, x, enable: float = 1.0) -> float:
        """Draw with the feed shut but the reactor held warm."""
        return self.power_for_feed(x, 0.0, enable)

    def light_off_energy_J(self, from_temperature_K: float | None = None) -> float:
        """Electric energy to bring the reactor from cold to its target, J."""
        start = self.p.temperature_initial_K if from_temperature_K is None else from_temperature_K
        return self.p.thermal_capacity_J_K * max(self.p.temperature_target_K - start, 0.0)

    def best_temperature_K(self, activity: float = 1.0, feed_fraction: float = 1.0) -> float:
        """Temperature maximising conversion -- the balance of kinetics and equilibrium."""
        temperatures = np.linspace(450.0, self.p.temperature_max_K, 400)
        best, best_x = temperatures[0], -np.inf
        for temperature in temperatures:
            state = np.array([temperature, activity, 0.0])
            value = float(self.conversion(state, np.array([feed_fraction, 1.0])))
            if value > best_x:
                best, best_x = float(temperature), value
        return best
