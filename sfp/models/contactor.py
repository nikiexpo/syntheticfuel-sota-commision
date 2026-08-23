"""Air contactor: CO2 out of ambient air and into the solid inventory.

    CaO + CO2 -> CaCO3        exothermic, ambient temperature, fans only

Purely algebraic -- it owns no states. What it captures goes into
`SolidsInventory`, and the sorbent state it tapers against lives there too, so
both arrive through the coupling channel `w`.

The one piece of physics that matters for control
-------------------------------------------------
Capture and fan power scale differently with airflow:

    CO2 offered   ~  V                       (more air, more CO2)
    capture eta   =  1 - exp(-NTU),  NTU ~ 1/V   (less residence time)
    fan power     ~  V^3                     (affinity laws)

At the reference sizing, at 20 degC and 60 % RH (verified against the model):

    flow  60 % of max  ->  0.233 mol/s captured for  4.7 kW   (128 kWh/tCO2)
    flow 100 % of max  ->  0.279 mol/s captured for 20.4 kW   (462 kWh/tCO2)

**Twenty percent more CO2 for four and a third times the fan power.** A
power-follow controller that dumps surplus solar into the fans burns most of it
for almost nothing, and no amount of cheap electricity makes that a good trade,
because the fan energy could have gone into the calciner instead. This is a
genuine, emergent trade-off rather than an imposed penalty, and it is one of the
clearest places where the planning layer should beat the baselines.

Note what the right question is. Maximising captured CO2 *per watt* is
degenerate -- it drives flow to zero, where the single-pass fraction tends to 1
and the power tends to 0, and the reactor starves. The useful problem is
constrained: meet the CO2 demand for the least fan energy. See
`flow_for_capture_rate`.

Ambient dependence
------------------
Rate rises mildly with temperature (Arrhenius, modest activation energy) and
with humidity (surface water films catalyse carbonation). The humidity term is
one of the few places where *where you build the plant* changes process
performance rather than merely solar yield -- a dry inland site captures less per
unit of fan energy than a coastal one at the same irradiance. The structure
follows the ambient-dependence treatment in the open-source DAC model of
Shakouri Kalfati & Abdulla (2025), though that paper models solid-sorbent TVSA on
zeolites rather than calcium looping.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sfp.models import mathx as mx
from sfp.models.base import Signal, Subsystem, VectorSpec
from sfp.units import M_CO2, R_GAS, T0_C


class AirContactor(Subsystem):
    """Ambient-temperature CO2 capture onto circulating CaO."""

    name = "contactor"

    @property
    def states(self) -> VectorSpec:
        return VectorSpec(())

    # --- coupling ---------------------------------------------------------
    @property
    def requires(self) -> tuple[str, ...]:
        return ("solids_loading",)

    @property
    def provides(self) -> tuple[str, ...]:
        return ("r_carbonation_mol_s",)

    @property
    def inputs(self) -> VectorSpec:
        return VectorSpec(
            (
                Signal("air_flow_fraction", "-", "fan speed as a fraction of maximum flow",
                       lower=0.0, upper=1.0),
                Signal("enable", "-", "1 = energised, 0 = dark", lower=0.0, upper=1.0),
            )
        )

    # --- ratings ----------------------------------------------------------
    @property
    def fan_power_rated_W(self) -> float:
        return self.p.fan_power_rated_kw * 1e3

    @property
    def fan_idle_W(self) -> float:
        return self.p.fan_idle_kw * 1e3

    @property
    def ntu_design(self) -> float:
        """Number of transfer units implied by the design capture fraction."""
        return float(-np.log(max(1.0 - self.p.capture_fraction_design, 1e-6)))

    def _enable_gate(self, u):
        return mx.smooth_step(mx.clip(u[1], 0.0, 1.0) - 0.5, width=0.05)

    def _flow_fraction(self, u):
        return mx.clip(u[0], 0.0, 1.0) * self._enable_gate(u)

    # --- physics ----------------------------------------------------------
    def air_flow_m3_s(self, u):
        return self._flow_fraction(u) * self.p.air_flow_max_m3_s

    def capture_fraction(self, flow_m3_s):
        """Single-pass CO2 capture efficiency, falling with airflow.

        NTU is inversely proportional to flow (fixed bed, so residence time
        ~ 1/V). The 1e-3 floor keeps the expression finite at zero flow, where
        the capture fraction is irrelevant because no air is moving.
        """
        ntu = self.ntu_design * self.p.air_flow_design_m3_s / mx.fmax(flow_m3_s, 1e-3)
        return 1.0 - mx.exp(-ntu)

    def molar_air_density(self, temp_air_C, pressure_Pa):
        """Ideal-gas molar density of ambient air, mol/m^3."""
        return pressure_Pa / (R_GAS * (temp_air_C + T0_C))

    def temperature_factor(self, temp_air_C):
        """Arrhenius rate correction about the reference temperature."""
        t_k = temp_air_C + T0_C
        t_ref = self.p.temperature_reference_K
        return mx.exp(
            -self.p.activation_energy_J_mol / R_GAS * (1.0 / mx.fmax(t_k, 200.0) - 1.0 / t_ref)
        )

    def humidity_factor(self, relative_humidity_pct):
        """Saturating moisture promotion, normalised to 1 at the reference RH."""
        rh = mx.clip(relative_humidity_pct / 100.0, 0.0, 1.0)
        k = self.p.humidity_half_saturation
        ref = self.p.humidity_reference
        return (rh / (rh + k)) / (ref / (ref + k))

    def loading_factor(self, loading):
        """Rate taper as the sorbent fills up.

        Flat until `loading_taper`, then falling smoothly to zero at full
        saturation, so the contactor stops on its own rather than pushing CO2
        into a bed that cannot hold it.
        """
        taper = self.p.loading_taper
        headroom = mx.smooth_clip((1.0 - loading) / mx.fmax(1.0 - taper, 1e-3), 0.0, 1.0, eps=1e-3)
        return headroom

    def capture_rate_mol_s(self, u, w: Mapping[str, Any]):
        """CO2 capture rate, mol/s."""
        flow = self.air_flow_m3_s(u)
        density = self.molar_air_density(
            w.get("temp_air", 20.0), w.get("pressure", 101325.0)
        )
        co2_offered = flow * density * self.p.co2_mole_fraction

        eta = self.capture_fraction(flow)
        f_t = self.temperature_factor(w.get("temp_air", 20.0))
        f_rh = self.humidity_factor(w.get("relative_humidity", 60.0))
        f_load = self.loading_factor(w.get("solids_loading", 0.0))

        return co2_offered * eta * f_t * f_rh * f_load

    def fan_power_W(self, u):
        """Fan shaft power, W. Cubic in flow by the affinity laws."""
        frac = self._flow_fraction(u)
        enabled = self._enable_gate(u)
        return self.fan_power_rated_W * frac**self.p.fan_exponent + self.fan_idle_W * enabled

    # --- interface --------------------------------------------------------
    def rhs(self, t, x, u, w: Mapping[str, Any]):
        return mx.zeros_like_state(x, 0)

    def outputs(self, t, x, u, w: Mapping[str, Any]) -> dict[str, Any]:
        flow = self.air_flow_m3_s(u)
        rate = self.capture_rate_mol_s(u, w)
        power = self.fan_power_W(u)

        return {
            "power_electrical_W": power,
            # coupling signal consumed by SolidsInventory.rhs
            "r_carbonation_mol_s": rate,
            "contactor_air_flow_m3_s": flow,
            "contactor_capture_fraction": self.capture_fraction(flow),
            "contactor_co2_rate_kg_s": rate * M_CO2,
            "contactor_fan_power_W": power,
            "contactor_flow_fraction": self._flow_fraction(u),
            "contactor_enabled": self._enable_gate(u),
            # the number the controller should actually care about
            "contactor_specific_energy_kWh_per_tCO2": power
            / mx.fmax(rate * M_CO2, 1e-9)
            / 3.6e6
            * 1e3,
        }

    def initial_state(self) -> np.ndarray:
        return np.zeros(0)

    # --- helpers for the dispatcher and the planner -----------------------
    def power_for_flow(self, flow_fraction: float, enable: float = 1.0) -> float:
        return float(self.fan_power_W(np.array([flow_fraction, enable], dtype=float)))

    def flow_for_power(self, power_W: float, enable: float = 1.0) -> float:
        """Invert the cubic fan law: the flow fraction drawing roughly `power_W`."""
        if enable < 0.5:
            return 0.0
        usable = max(power_W - self.fan_idle_W, 0.0)
        return float(np.clip((usable / self.fan_power_rated_W) ** (1.0 / self.p.fan_exponent), 0.0, 1.0))

    def flow_for_capture_rate(
        self, target_mol_s: float, w: Mapping[str, Any] | None = None
    ) -> float:
        """Cheapest flow fraction that meets a target capture rate.

        This -- not "maximise CO2 per watt" -- is the question worth asking.
        Capture per unit of fan energy rises without bound as flow falls (the
        single-pass fraction tends to 1 while power tends to zero), so an
        unconstrained efficiency optimum is degenerate and simply says "run the
        fans as slowly as possible", which starves the reactor. The real problem
        is constrained: meet the CO2 demand for the least fan energy, and since
        capture increases monotonically with flow, that means the *smallest* flow
        that still meets the target.

        Returns 1.0 if even full flow cannot reach the target.
        """
        w = w or {"temp_air": 20.0, "relative_humidity": 60.0, "pressure": 101325.0}
        if target_mol_s <= 0.0:
            return 0.0
        if float(self.capture_rate_mol_s(np.array([1.0, 1.0]), w)) < target_mol_s:
            return 1.0
        low, high = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (low + high)
            if float(self.capture_rate_mol_s(np.array([mid, 1.0]), w)) < target_mol_s:
                low = mid
            else:
                high = mid
        return float(0.5 * (low + high))

    def marginal_capture_per_watt(
        self, flow_fraction: float, w: Mapping[str, Any] | None = None, delta: float = 1e-3
    ) -> float:
        """d(capture)/d(fan power) at a given flow, mol/J.

        The quantity the planner's price signal should be compared against: at a
        shadow price lambda, increasing flow is worth it only while this exceeds
        lambda's implied value per mole. Falls steeply with flow because power
        goes as V^3 while capture saturates.
        """
        w = w or {"temp_air": 20.0, "relative_humidity": 60.0, "pressure": 101325.0}
        lo = np.array([max(flow_fraction - delta, 0.0), 1.0])
        hi = np.array([min(flow_fraction + delta, 1.0), 1.0])
        d_rate = float(self.capture_rate_mol_s(hi, w)) - float(self.capture_rate_mol_s(lo, w))
        d_power = float(self.fan_power_W(hi)) - float(self.fan_power_W(lo))
        if abs(d_power) < 1e-12:
            return float("inf")
        return d_rate / d_power
