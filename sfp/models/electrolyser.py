"""PEM water electrolyser: H2O -> H2 + 1/2 O2.

Structure after Gorgun (2006), with a lumped stack thermal state added -- the
paper resolves mole balances, membrane transport and cell voltage but not stack
temperature, and temperature matters for scheduling because it moves the ohmic
overpotential and therefore the efficiency the planner sees.

**Total efficiency peaks well below rated load.** Two effects fight:

    stack efficiency   falls with current   (activation + ohmic overpotentials)
    auxiliary share    falls with current   (a fixed ~15 kW spread over more H2)

so the total has an interior maximum. At the reference sizing:

    load    stack eta   total eta (LHV)
     10 %     0.75         0.48
     25 %     0.73         0.60
     50 %     0.69         0.62      <- peak
    100 %     0.64         0.61

A power-follow controller pushes to 100 % whenever the sun is strong and gives
up a couple of points for nothing, because that marginal kilowatt was worth more
in the kiln. This emerges from the polarisation curve and a fixed auxiliary
load; it is not an imposed penalty.

Activation overpotential uses the **inverse-hyperbolic-sine form** of
Butler-Volmer rather than the Tafel logarithm:

    eta_act = (RT / alpha F) * asinh( i / 2 i_0 )

Tafel diverges to -infinity as i -> 0, which wrecks an NLP that evaluates the
stack at zero load. asinh is exact at all currents, smooth through the origin,
and agrees with Tafel wherever Tafel is valid.

Thermal management is a **local regulatory loop**, not a supervisory decision:
the stack regulates its own temperature, as real stacks do.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sfp.models import mathx as mx
from sfp.models.base import Signal, Subsystem, VectorSpec
from sfp.units import F_FARADAY, LHV_H2_MASS, M_H2, M_H2O, R_GAS


class Electrolyser(Subsystem):
    """PEM stack with polarisation curve, thermal state and voltage degradation."""

    name = "electrolyser"

    @property
    def states(self) -> VectorSpec:
        return VectorSpec(
            (
                Signal("temperature_K", "K", "lumped stack temperature",
                       lower=250.0, upper=self.p.temperature_max_K),
                Signal("v_degradation", "V", "accumulated cell-voltage rise",
                       lower=0.0, upper=1.0),
            )
        )

    # --- coupling ---------------------------------------------------------
    @property
    def provides(self) -> tuple[str, ...]:
        return ("r_electrolysis_h2_mol_s", "r_electrolysis_water_mol_s")

    @property
    def inputs(self) -> VectorSpec:
        return VectorSpec(
            (
                Signal("current_fraction", "-", "current density as a fraction of rated",
                       lower=0.0, upper=1.0),
                Signal("enable", "-", "1 = energised, 0 = shut down", lower=0.0, upper=1.0),
            )
        )

    # --- ratings ----------------------------------------------------------
    @property
    def min_setpoint(self) -> float:
        """Hydrogen crossover safety limit -- a hard floor, never relaxed."""
        return float(self.p.current_density_min_fraction)

    @property
    def auxiliary_power_W(self) -> float:
        return self.p.auxiliary_power_kw * 1e3

    @property
    def rated_current_A(self) -> float:
        return self.p.current_density_max_A_cm2 * self.p.active_area_cm2

    def _enable_gate(self, u):
        return mx.smooth_step(mx.clip(u[1], 0.0, 1.0) - 0.5, width=0.05)

    def _current_fraction(self, u):
        """Commanded current, gated off below the crossover-safety minimum."""
        commanded = mx.clip(u[0], 0.0, 1.0)
        floor = self.p.current_density_min_fraction
        gate = mx.smooth_step(commanded - 0.6 * floor, width=0.05 * floor)
        return commanded * gate * self._enable_gate(u)

    def current_density(self, u):
        return self._current_fraction(u) * self.p.current_density_max_A_cm2

    # --- polarisation curve ----------------------------------------------
    def reversible_voltage(self, temperature_K):
        return self.p.reversible_voltage_V + self.p.reversible_temp_coefficient_V_K * (
            temperature_K - 298.15
        )

    def activation_overpotential(self, current_density, temperature_K):
        """Butler-Volmer in asinh form -- exact and smooth down to zero current."""
        prefactor = (R_GAS * temperature_K) / (
            self.p.charge_transfer_coefficient * F_FARADAY
        )
        ratio = current_density / (2.0 * self.p.exchange_current_density_A_cm2)
        return prefactor * mx.log(ratio + mx.sqrt(ratio * ratio + 1.0))

    def area_resistance(self, temperature_K):
        """Membrane area-specific resistance, falling with temperature."""
        delta = temperature_K - self.p.temperature_setpoint_K
        return mx.fmax(
            self.p.area_resistance_ohm_cm2 * (1.0 + self.p.resistance_temp_coefficient * delta),
            0.02,
        )

    def cell_voltage(self, current_density, temperature_K, v_degradation):
        return (
            self.reversible_voltage(temperature_K)
            + self.activation_overpotential(current_density, temperature_K)
            + current_density * self.area_resistance(temperature_K)
            + v_degradation
        )

    # --- production and power --------------------------------------------
    def hydrogen_rate_mol_s(self, x, u):
        """Faraday's law: 2 electrons per H2."""
        current = self.current_density(u) * self.p.active_area_cm2
        return self.p.faraday_efficiency * self.p.n_cells * current / (2.0 * F_FARADAY)

    def stack_power_W(self, x, u):
        i = self.current_density(u)
        current = i * self.p.active_area_cm2
        v_cell = self.cell_voltage(i, x[0], x[1])
        return self.p.n_cells * v_cell * current

    def total_power_W(self, x, u):
        enabled = self._enable_gate(u)
        heat = self.heat_generated_W(x, u)
        cooling_parasitic = self.p.cooling_parasitic_fraction * mx.fmax(heat, 0.0)
        return self.stack_power_W(x, u) + self.auxiliary_power_W * enabled + cooling_parasitic

    def heat_generated_W(self, x, u):
        """Ohmic and activation heat: N*I*(V_cell - V_thermoneutral).

        Written as a voltage difference rather than (electrical in - chemical
        out) so we are not differencing two large, nearly equal numbers.
        """
        i = self.current_density(u)
        current = i * self.p.active_area_cm2
        v_cell = self.cell_voltage(i, x[0], x[1])
        return self.p.n_cells * current * (v_cell - self.p.thermoneutral_voltage_V)

    def cooling_power_W(self, x):
        """Proportional thermal-management loop -- a local regulatory controller."""
        error = x[0] - self.p.temperature_setpoint_K
        demand = self.p.cooling_gain_W_K * mx.fmax(error, 0.0)
        return mx.fmin(demand, self.p.cooling_power_max_W)

    def efficiency_lhv(self, x, u):
        """Total system efficiency on an LHV basis, including auxiliaries."""
        power = self.total_power_W(x, u)
        chemical = self.hydrogen_rate_mol_s(x, u) * M_H2 * LHV_H2_MASS
        return chemical / mx.fmax(power, 1.0)

    # --- dynamics ---------------------------------------------------------
    def rhs(self, t, x, u, w: Mapping[str, Any]):
        heat = self.heat_generated_W(x, u)
        cooling = self.cooling_power_W(x)
        temp_amb_K = w.get("temp_air", 20.0) + 273.15
        ambient = self.p.ambient_UA_W_K * (x[0] - temp_amb_K)

        d_temperature = (heat - cooling - ambient) / self.p.thermal_capacity_J_K

        # degradation accrues with time under load; the start-up penalty is
        # applied by the fault/commitment logic rather than integrated here
        load = self._current_fraction(u)
        d_degradation = self.p.degradation_V_per_s * load

        return mx.vertcat(d_temperature, d_degradation)

    def outputs(self, t, x, u, w: Mapping[str, Any]) -> dict[str, Any]:
        i = self.current_density(u)
        rate = self.hydrogen_rate_mol_s(x, u)
        enabled = self._enable_gate(u)

        return {
            "power_electrical_W": self.total_power_W(x, u),
            # coupling signals
            "r_electrolysis_h2_mol_s": rate,
            "r_electrolysis_water_mol_s": rate,  # 1 mol H2O consumed per mol H2
            "electrolyser_current_density_A_cm2": i,
            "electrolyser_cell_voltage_V": self.cell_voltage(i, x[0], x[1]),
            "electrolyser_stack_power_W": self.stack_power_W(x, u),
            "electrolyser_aux_power_W": self.auxiliary_power_W * enabled,
            "electrolyser_h2_rate_kg_s": rate * M_H2,
            "electrolyser_temperature_K": x[0],
            "electrolyser_temperature_C": x[0] - 273.15,
            "electrolyser_v_degradation": x[1],
            "electrolyser_efficiency_lhv": self.efficiency_lhv(x, u),
            "electrolyser_heat_W": self.heat_generated_W(x, u),
            "electrolyser_load_fraction": self._current_fraction(u),
            "electrolyser_enabled": enabled,
        }

    def initial_state(self) -> np.ndarray:
        return np.array(
            [self.p.temperature_initial_K, self.p.voltage_degradation_initial_V], dtype=float
        )

    # --- helpers for the dispatcher and the planner -----------------------
    def rated_power_W(self, x: np.ndarray | None = None) -> float:
        """Total draw at full current and nominal temperature."""
        state = np.array([self.p.temperature_setpoint_K, 0.0]) if x is None else x
        return float(self.total_power_W(state, np.array([1.0, 1.0])))

    def min_power_W(self, x: np.ndarray | None = None) -> float:
        """Total draw at the crossover-limited minimum load."""
        state = np.array([self.p.temperature_setpoint_K, 0.0]) if x is None else x
        return float(
            self.total_power_W(state, np.array([self.p.current_density_min_fraction, 1.0]))
        )

    def power_for_fraction(self, x, current_fraction: float, enable: float = 1.0) -> float:
        return float(self.total_power_W(x, np.array([current_fraction, enable], dtype=float)))

    def fraction_for_power(self, x, power_W: float, enable: float = 1.0) -> float:
        """Invert the polarisation curve numerically.

        P(i) is strictly increasing in current, so bisection is safe and needs no
        derivative. Returns 0 when the target cannot clear the minimum load.
        """
        if enable < 0.5:
            return 0.0
        floor = self.p.current_density_min_fraction
        if power_W < self.power_for_fraction(x, floor, enable) - 1e-9:
            return 0.0
        if power_W >= self.power_for_fraction(x, 1.0, enable):
            return 1.0
        low, high = floor, 1.0
        for _ in range(40):
            mid = 0.5 * (low + high)
            if self.power_for_fraction(x, mid, enable) < power_W:
                low = mid
            else:
                high = mid
        return float(0.5 * (low + high))

    def best_efficiency_fraction(self, x: np.ndarray | None = None) -> float:
        """Load fraction maximising total LHV efficiency -- the part-load peak.

        Cached for the nominal state, because it is a property of the
        polarisation curve and the auxiliary load rather than of the moment. A
        controller that queries it every step was otherwise running a 200-point
        scan 288 times per simulated day, which was the single largest
        algorithmic cost in the closed loop.

        Passing an explicit `x` bypasses the cache, since the peak does move
        slightly with stack temperature and degradation.
        """
        if x is None:
            cached = getattr(self, "_best_fraction_cache", None)
            if cached is not None:
                return cached
        state = np.array([self.p.temperature_setpoint_K, 0.0]) if x is None else x
        fractions = np.linspace(self.p.current_density_min_fraction, 1.0, 200)
        best, best_eta = float(fractions[0]), -np.inf
        for frac in fractions:
            eta = float(self.efficiency_lhv(state, np.array([frac, 1.0])))
            if eta > best_eta:
                best, best_eta = float(frac), eta
        if x is None:
            self._best_fraction_cache = best
        return best
