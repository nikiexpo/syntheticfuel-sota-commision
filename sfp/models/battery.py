"""Battery: an energy reservoir with a differentiable capacity-fade state.

Charge and discharge are kept as two separate non-negative inputs rather than
one signed power. That is deliberate: a signed power needs `abs()` for the
throughput and efficiency terms, and `abs()` puts a kink right at the operating
point the solver spends most of its time near. Two non-negative variables make
the whole model smooth, at the cost of admitting the unphysical
simultaneous-charge-and-discharge solution -- which never appears in practice
because both directions lose energy, so it is always dominated.

Fade is a state, not a post-processing step, so the planner can see that cycling
the battery today costs capacity tomorrow. Together with the sorbent-deactivation
state in the carbonator this is what stops the controller from behaving like the
greedy baseline: every buffer in this plant wears out, and the wear rates differ
by orders of magnitude.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sfp.models import mathx as mx
from sfp.models.base import Signal, Subsystem, VectorSpec
from sfp.units import J_PER_KWH, SECONDS_PER_DAY, SECONDS_PER_YEAR


class Battery(Subsystem):
    """Lithium-iron-phosphate pack with a bidirectional inverter."""

    name = "battery"

    @property
    def states(self) -> VectorSpec:
        return VectorSpec(
            (
                Signal("soc", "-", "state of charge", lower=0.0, upper=1.0),
                Signal("fade", "-", "fractional capacity loss", lower=0.0, upper=1.0),
                Signal("efc", "cycles", "cumulative equivalent full cycles", lower=0.0),
            )
        )

    @property
    def inputs(self) -> VectorSpec:
        p_max = self.p.power_kw * 1e3
        return VectorSpec(
            (
                Signal("p_charge_W", "W", "power drawn from the bus into the pack",
                       lower=0.0, upper=p_max),
                Signal("p_discharge_W", "W", "power delivered from the pack to the bus",
                       lower=0.0, upper=p_max),
            )
        )

    # --- ratings ----------------------------------------------------------
    @property
    def nominal_energy_J(self) -> float:
        return self.p.capacity_kwh * J_PER_KWH

    @property
    def max_power_W(self) -> float:
        return self.p.power_kw * 1e3

    def usable_energy_J(self, fade):
        """Present capacity after fade, J."""
        return self.nominal_energy_J * mx.fmax(1.0 - fade, 1e-3)

    def round_trip_efficiency(self) -> float:
        return self.p.eta_charge * self.p.eta_discharge

    def stored_energy_J(self, x):
        """Energy currently available above the empty reference, J."""
        return x[0] * self.usable_energy_J(x[1])

    # --- operating envelope, used by the controller ----------------------
    def max_charge_power_W(self, x, dt_s: float):
        """Largest charge power that will not overshoot soc_max within `dt_s`."""
        headroom_J = mx.fmax((self.p.soc_max - x[0]), 0.0) * self.usable_energy_J(x[1])
        return mx.fmin(self.max_power_W, headroom_J / (self.p.eta_charge * max(dt_s, 1e-6)))

    def max_discharge_power_W(self, x, dt_s: float):
        """Largest discharge power that will not undershoot soc_min within `dt_s`."""
        available_J = mx.fmax((x[0] - self.p.soc_min), 0.0) * self.usable_energy_J(x[1])
        return mx.fmin(self.max_power_W, available_J * self.p.eta_discharge / max(dt_s, 1e-6))

    # --- dynamics ---------------------------------------------------------
    def rhs(self, t, x, u, w: Mapping[str, Any]):
        soc, fade, _efc = x[0], x[1], x[2]
        p_charge = mx.fmax(u[0], 0.0)
        p_discharge = mx.fmax(u[1], 0.0)

        usable_J = self.usable_energy_J(fade)

        # net energy into the pack, after one-way losses in each direction
        net_W = self.p.eta_charge * p_charge - p_discharge / self.p.eta_discharge
        self_discharge = self.p.self_discharge_per_day / SECONDS_PER_DAY * soc

        d_soc = net_W / usable_J - self_discharge

        # one equivalent full cycle = a full charge plus a full discharge
        d_efc = (p_charge + p_discharge) / (2.0 * self.nominal_energy_J)

        cycle_fade = self.p.end_of_life_fade / self.p.cycle_life * d_efc
        calendar_fade = self.p.calendar_fade_per_year / SECONDS_PER_YEAR
        d_fade = cycle_fade + calendar_fade

        return mx.vertcat(d_soc, d_fade, d_efc)

    def outputs(self, t, x, u, w: Mapping[str, Any]) -> dict[str, Any]:
        soc, fade, efc = x[0], x[1], x[2]
        p_charge = mx.fmax(u[0], 0.0)
        p_discharge = mx.fmax(u[1], 0.0)

        losses = (1.0 - self.p.eta_charge) * p_charge + (
            1.0 / self.p.eta_discharge - 1.0
        ) * p_discharge

        return {
            # positive = consumed from the bus, so charging is positive
            "power_electrical_W": p_charge - p_discharge,
            "battery_soc": soc,
            "battery_fade": fade,
            "battery_efc": efc,
            "battery_stored_kWh": self.stored_energy_J(x) / J_PER_KWH,
            "battery_usable_kWh": self.usable_energy_J(fade) / J_PER_KWH,
            "battery_loss_W": losses,
            "battery_charge_W": p_charge,
            "battery_discharge_W": p_discharge,
        }

    def initial_state(self) -> np.ndarray:
        return np.array([self.p.soc_initial, 0.0, 0.0], dtype=float)

    # --- economics --------------------------------------------------------
    def capex_EUR(self) -> float:
        return self.p.capex_per_kwh * self.p.capacity_kwh

    def cost_per_efc_EUR(self) -> float:
        """Replacement cost attributable to one equivalent full cycle.

        The pack is written off over `cycle_life` cycles, so each cycle carries
        capex / cycle_life. This is the number the controller pays to move energy
        through the battery, and it is what makes chemical storage look cheap by
        comparison whenever the chemistry can absorb the energy instead.
        """
        return self.capex_EUR() / self.p.cycle_life

    def cost_per_kWh_throughput_EUR(self) -> float:
        """Marginal degradation cost of battery throughput, EUR/kWh."""
        return self.cost_per_efc_EUR() / (2.0 * self.p.capacity_kwh)

    def degradation_cost_EUR(self, efc_used: float) -> float:
        return self.cost_per_efc_EUR() * efc_used

    def remaining_life_fraction(self, fade) -> float:
        """1.0 at beginning of life, 0.0 at the end-of-life fade threshold."""
        return float(np.clip(1.0 - fade / self.p.end_of_life_fade, 0.0, 1.0))
