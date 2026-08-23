"""Electrically heated calciner: CaCO3 -> CaO + CO2 at ~900 degC.

The plant's largest flexible load, and the source of its hardest scheduling
decision.

Why this subsystem drives the architecture
------------------------------------------
1. **A sharp thermodynamic threshold.** The reaction proceeds only while the
   equilibrium CO2 pressure exceeds the kiln's operating pressure. Baker's
   correlation gives

       p_eq(T) = 4.137e7 * exp(-20474/T)   [atm]

   which is 1 atm at 1170 K -- the textbook 897 degC. Running the kiln at 0.3 atm
   (steam sweep or vacuum) drops the threshold to about 1092 K (819 degC). Below
   that, heating the kiln produces *nothing*: there is no partial credit.

2. **Enormous thermal inertia.** ~3 MJ/K of refractory means a cold start costs
   772 kWh and takes 5.15 h at rated power, with the first CO2 appearing only
   after 4.9 h. You cannot chase a cloud with this machine.

Together those make "run the kiln today?" a unit-commitment problem, not a
setpoint choice -- and the overnight question turns out to be a dead heat
(all figures verified against the model, not estimated):

    hold at 900 degC for 12 h   180 kWh of standing loss
    let it drift and reheat     178 kWh  (it falls only to 697 degC; C/UA = 49 h)

Two kWh apart. There is no rule of thumb that resolves that, and it is exactly
why a planner earns its keep here: which one wins depends on ambient temperature,
on wind, on how long the gap really is, and on whether there will be sun to
reheat with -- none of which a local rule can know.

The kiln is deliberately oversized (150 kW against a ~70 kW steady demand). That
headroom is the charging power of the chemical battery: it is how midday surplus
gets banked as CO2 instead of being curtailed.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sfp.models import mathx as mx
from sfp.models.base import Signal, Subsystem, VectorSpec
from sfp.units import DH_CALCINATION, M_CACO3, P_ATM, R_GAS


class Calciner(Subsystem):
    """Lumped-thermal-mass rotary kiln with equilibrium-limited calcination."""

    name = "calciner"

    @property
    def states(self) -> VectorSpec:
        return VectorSpec(
            (
                Signal("temperature_K", "K", "lumped kiln temperature",
                       lower=200.0, upper=self.p.temperature_max_K),
            )
        )

    # --- coupling ---------------------------------------------------------
    @property
    def requires(self) -> tuple[str, ...]:
        return ("solids_n_caco3_mol",)

    @property
    def provides(self) -> tuple[str, ...]:
        return ("r_calcination_mol_s",)

    @property
    def inputs(self) -> VectorSpec:
        return VectorSpec(
            (
                Signal("heater_fraction", "-", "electric heater duty",
                       lower=0.0, upper=1.0),
                Signal("enable", "-", "1 = energised, 0 = dark and cooling",
                       lower=0.0, upper=1.0),
            )
        )

    # --- ratings ----------------------------------------------------------
    @property
    def heater_power_rated_W(self) -> float:
        return self.p.heater_power_rated_kw * 1e3

    @property
    def standby_power_W(self) -> float:
        return self.p.standby_power_kw * 1e3

    def _enable_gate(self, u):
        return mx.smooth_step(mx.clip(u[1], 0.0, 1.0) - 0.5, width=0.05)

    def overtemperature_interlock(self, temperature_K):
        """Hard-wired heater cutout as the refractory limit is approached.

        This is a layer-4 interlock, not a control action: every real kiln has
        one, and no optimiser is allowed to override it. It matters here because
        the energy balance genuinely runs away without it -- at full heater duty
        the throughput limit caps the endothermic heat sink at ~0.45 mol/s, and
        the steady-state temperature would settle near 1357 K, well past the
        refractory limit. The NMPC will carry T <= T_max as a hard constraint;
        this is what happens if it fails anyway.
        """
        return mx.smooth_step(self.p.temperature_max_K - temperature_K, width=5.0)

    def _heater_fraction(self, u, temperature_K=None):
        commanded = mx.clip(u[0], 0.0, 1.0) * self._enable_gate(u)
        if temperature_K is None:
            return commanded
        return commanded * self.overtemperature_interlock(temperature_K)

    # --- thermodynamics ---------------------------------------------------
    def equilibrium_pressure_atm(self, temperature_K):
        """Baker (1962) CaCO3 decomposition equilibrium pressure, atm."""
        t = mx.fmax(temperature_K, 200.0)
        return self.p.baker_prefactor_atm * mx.exp(-self.p.baker_exponent_K / t)

    def threshold_temperature_K(self) -> float:
        """Temperature at which p_eq equals the kiln's operating pressure.

        Below this the kiln consumes power and produces nothing.
        """
        return float(
            self.p.baker_exponent_K
            / np.log(self.p.baker_prefactor_atm / self.p.operating_pressure_atm)
        )

    def driving_force(self, temperature_K):
        """Thermodynamic driving force (1 - p_op/p_eq), clipped to [0, 1].

        Smooth so an NLP can approach the threshold from either side; the
        underlying physics is genuinely a hard switch.
        """
        p_eq = self.equilibrium_pressure_atm(temperature_K)
        ratio = self.p.operating_pressure_atm / mx.fmax(p_eq, 1e-12)
        return mx.smooth_clip(1.0 - ratio, 0.0, 1.0, eps=1e-3)

    def kinetic_factor(self, temperature_K):
        """Arrhenius rate factor, normalised so it saturates near the target."""
        t = mx.fmax(temperature_K, 200.0)
        return self.p.kinetic_prefactor_1_s * mx.exp(
            -self.p.kinetic_activation_J_mol / (R_GAS * t)
        )

    def feedstock_factor(self, n_caco3):
        """Rate taper as the kiln runs out of CaCO3 to calcine."""
        return mx.smooth_clip(n_caco3 / mx.fmax(self.p.feedstock_reference_mol, 1.0),
                              0.0, 1.0, eps=1e-3)

    def calcination_rate_mol_s(self, x, u, w: Mapping[str, Any]):
        """CaCO3 decomposition rate, mol/s.

        Product of three independent limits: thermodynamics (is it hot enough?),
        kinetics (how fast at this temperature?), and feedstock (is there any
        CaCO3 left?). Capped by the mechanical throughput of the feeder.
        """
        temperature = x[0]
        enabled = self._enable_gate(u)
        rate = (
            self.p.calcination_rate_max_mol_s
            * self.driving_force(temperature)
            * mx.smooth_clip(self.kinetic_factor(temperature), 0.0, 1.0, eps=1e-3)
            * self.feedstock_factor(w.get("solids_n_caco3_mol", 0.0))
            * enabled
        )
        return mx.fmax(rate, 0.0)

    # --- energy balance ---------------------------------------------------
    def heat_loss_W(self, x, w: Mapping[str, Any]):
        temperature = x[0]
        temp_amb_K = w.get("temp_air", 20.0) + 273.15
        ua = self.p.heat_loss_UA_W_K + self.p.wind_loss_coefficient * mx.fmax(
            w.get("wind_speed", 1.0), 0.0
        )
        return ua * (temperature - temp_amb_K)

    def sensible_heat_W(self, rate_mol_s, x):
        """Power spent heating incoming solids from feed to kiln temperature."""
        mass_flow = rate_mol_s * M_CACO3
        return mass_flow * self.p.solids_heat_capacity_J_kg_K * mx.fmax(
            x[0] - self.p.solids_feed_temperature_K, 0.0
        )

    def rhs(self, t, x, u, w: Mapping[str, Any]):
        heater_W = (
            self._heater_fraction(u, x[0]) * self.heater_power_rated_W * self.p.heater_efficiency
        )
        rate = self.calcination_rate_mol_s(x, u, w)

        reaction_W = rate * DH_CALCINATION  # endothermic, positive = heat absorbed
        loss_W = self.heat_loss_W(x, w)
        sensible_W = self.sensible_heat_W(rate, x)

        d_temperature = (heater_W - loss_W - reaction_W - sensible_W) / self.p.thermal_capacity_J_K
        return mx.vertcat(d_temperature)

    def outputs(self, t, x, u, w: Mapping[str, Any]) -> dict[str, Any]:
        enabled = self._enable_gate(u)
        heater_frac = self._heater_fraction(u, x[0])
        heater_W = heater_frac * self.heater_power_rated_W
        rate = self.calcination_rate_mol_s(x, u, w)

        return {
            "power_electrical_W": heater_W + self.standby_power_W * enabled,
            # coupling signals: consumed by SolidsInventory and the CO2 buffer
            "r_calcination_mol_s": rate,
            "calciner_temperature_K": x[0],
            "calciner_temperature_C": x[0] - 273.15,
            "calciner_heater_W": heater_W,
            "calciner_heater_fraction": heater_frac,
            "calciner_enabled": enabled,
            "calciner_equilibrium_atm": self.equilibrium_pressure_atm(x[0]),
            "calciner_driving_force": self.driving_force(x[0]),
            "calciner_heat_loss_W": self.heat_loss_W(x, w),
            "calciner_reaction_W": rate * DH_CALCINATION,
            "calciner_co2_rate_kg_s": rate * 44.0095e-3,
            "calciner_is_productive": mx.smooth_step(rate - 1e-4, width=1e-4),
        }

    def initial_state(self) -> np.ndarray:
        return np.array([self.p.temperature_initial_K], dtype=float)

    # --- helpers for the dispatcher and the planner -----------------------
    def power_for_fraction(self, heater_fraction: float, enable: float = 1.0) -> float:
        gate = 1.0 if enable >= 0.5 else 0.0
        return float(
            heater_fraction * gate * self.heater_power_rated_W + self.standby_power_W * gate
        )

    def fraction_for_power(self, power_W: float, enable: float = 1.0) -> float:
        if enable < 0.5:
            return 0.0
        usable = max(power_W - self.standby_power_W, 0.0)
        return float(np.clip(usable / self.heater_power_rated_W, 0.0, 1.0))

    def cold_start_energy_J(self, from_temperature_K: float | None = None) -> float:
        """Energy to bring the kiln to its operating temperature, J."""
        start = self.p.temperature_initial_K if from_temperature_K is None else from_temperature_K
        delta = max(self.p.temperature_target_K - start, 0.0)
        return self.p.thermal_capacity_J_K * delta / max(self.p.heater_efficiency, 1e-3)

    def standing_loss_W(self, temperature_K: float | None = None, temp_air_C: float = 20.0) -> float:
        """Steady heat loss at temperature, W -- the cost of holding the kiln warm."""
        temperature = self.p.temperature_target_K if temperature_K is None else temperature_K
        return float(self.p.heat_loss_UA_W_K * (temperature - (temp_air_C + 273.15)))

    def thermal_time_constant_s(self) -> float:
        """C/UA -- how long the kiln takes to forget it was hot."""
        return float(self.p.thermal_capacity_J_K / self.p.heat_loss_UA_W_K)
