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

        # Cycle fade, weighted by how deeply the pack is being worked.
        #
        # `cycle_life` is rated at `dod_reference`, and real cells lose life
        # faster than proportionally when cycled deeper: the usual fit is
        # `N(DoD) = N_ref (DoD_ref/DoD)^k` with k above one, so the fade per unit
        # of throughput carries a factor `(DoD/DoD_ref)^(k-1)`.
        #
        # Without it, throughput is all that matters and a large pack swinging
        # gently is charged exactly as much per kWh as a small one cycling to its
        # limits twice a day. That flatters small packs, and it flatters them
        # precisely in a sizing sweep, which is where the error does most damage.
        #
        # **This is a local proxy for a path-dependent quantity.** Depth of
        # discharge is properly a property of a *cycle*, recovered by rainflow
        # counting over a trajectory; a differential model cannot see a cycle. The
        # instantaneous excursion from mid-band is used instead, which gets the
        # direction and rough magnitude right and should not be read as more than
        # that. It is floored so that shallow cycling is cheaper but never free.
        mid = 0.5 * (self.p.soc_max + self.p.soc_min)
        depth = 2.0 * mx.fabs(soc - mid) / self.p.dod_reference
        stress = self._dod_normalisation * mx.power(
            mx.smooth_max(depth, self.p.dod_stress_floor, eps=1e-2),
            self.p.dod_exponent - 1.0)

        cycle_fade = self.p.end_of_life_fade / self.p.cycle_life * stress * d_efc
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

    @property
    def _dod_normalisation(self) -> float:
        """Scale that makes a rated-depth cycle cost exactly one rated cycle.

        The stress term is evaluated at each instant, but `cycle_life` is a
        rating for a whole *cycle*. A cycle at the reference depth sweeps the
        normalised depth from 0 up to 1 and back, so the mean of `depth^(k-1)`
        along it is `1/k` -- not 1. Left unnormalised the model returns 17,450
        cycles where the datasheet says 6,000, which is nearly threefold in the
        battery's favour and would have quietly undone the whole point of adding
        the depth term.

        Closed form, including the floor: for `d` uniform on [0,1],

            E[max(d, f)^(k-1)] = f^k + (1 - f^k)/k
        """
        k = float(self.p.dod_exponent)
        f = float(self.p.dod_stress_floor)
        return 1.0 / (f ** k + (1.0 - f ** k) / k)

    def cost_per_kWh_delivered_EUR(self) -> float:
        """Degradation cost per kWh the pack actually delivers to the bus.

            C_deg = CAPEX / (Capacity * DoD * CycleLife * RTE)

        Three things the denominator accounts for that a bare `capex/cycle_life`
        does not:

        **DoD.** `cycle_life` is a rating *at a stated depth* -- 6000 cycles at
        80 % for these cells, per the datasheet note in `battery.yaml`. A "cycle"
        therefore moves 0.8 of nameplate, not all of it.

        **RTE.** Round-trip losses mean the energy that reaches the bus is
        `eta_charge * eta_discharge` of what went in. At 0.96 each way that is
        0.9216, so about 8 % of every cycle is heat.

        Together these were understating wear by a factor of
        `1/(0.8 * 0.9216) = 1.357`. The controller was being told the battery was
        a third cheaper to cycle than it is, which biases every storage decision
        and, in the sizing sweep, every conclusion about how big a pack to buy.
        """
        p = self.p
        usable = p.capacity_kwh * p.dod_reference * p.cycle_life
        return self.capex_EUR() / (usable * self.round_trip_efficiency())

    def cost_per_efc_EUR(self) -> float:
        """Replacement cost attributable to one equivalent full cycle.

        One EFC is `2 * capacity` of throughput -- charge in plus discharge out --
        so this is `C_deg` scaled to that. It still rises linearly with pack size,
        because a bigger pack costs more to replace and moves more energy per
        cycle; what is *independent* of size is the cost per kWh moved, which is
        correct: the same chemistry wears at the same rate per unit of energy.
        """
        return self.cost_per_kWh_delivered_EUR() * self.p.capacity_kwh

    def cost_per_kWh_throughput_EUR(self) -> float:
        """Marginal degradation cost of battery throughput, EUR/kWh."""
        return self.cost_per_efc_EUR() / (2.0 * self.p.capacity_kwh)

    # --- how long the pack actually lasts --------------------------------
    def fade_rates_per_year(self, efc_per_year: float,
                            mean_stress: float = 1.0) -> tuple[float, float]:
        """`(calendar, cycle)` fade per year at a given duty."""
        calendar = float(self.p.calendar_fade_per_year)
        cycle = (float(self.p.end_of_life_fade) / float(self.p.cycle_life)
                 * float(mean_stress) * float(efc_per_year))
        return calendar, cycle

    def life_years(self, efc_per_year: float, mean_stress: float = 1.0) -> float:
        """Years until the pack reaches its end-of-life fade at this duty."""
        calendar, cycle = self.fade_rates_per_year(efc_per_year, mean_stress)
        total = calendar + cycle
        return float(self.p.end_of_life_fade) / total if total > 0 else float("inf")

    def uncharged_replacement_PV_EUR(
        self, *, project_years: float, discount_rate: float,
        efc_per_year: float, mean_stress: float = 1.0,
    ) -> float:
        """Present value of pack replacements that nothing else pays for.

        The capital annuity amortises the battery over the *project's* life. The
        pack does not last that long: calendar fade alone (0.015/yr against a
        0.20 end-of-life threshold) gives 13.3 years, and cycling shortens it
        further -- at two equivalent full cycles a day a 1500 kWh pack lasts
        about five years, so the project needs four packs, not one.

        **Only the calendar-attributable share is returned, and that is the whole
        subtlety.** The operating objective already charges
        `cost_per_kWh_delivered` on every kWh moved, and over one pack's rated
        cycle life those charges total exactly one pack. So cycling *is* funded,
        through operating profit. What is not funded is the part of each
        replacement caused by time rather than use:

            share_uncharged = calendar / (calendar + cycle)

        Charging the whole replacement here as well would count the cycling twice
        and make the battery look worse than it is -- the opposite error, and
        just as wrong.
        """
        calendar, cycle = self.fade_rates_per_year(efc_per_year, mean_stress)
        total = calendar + cycle
        if total <= 0:
            return 0.0
        life = float(self.p.end_of_life_fade) / total
        share = calendar / total
        cost = self.capex_EUR() * share

        present = 0.0
        t = life
        while t < project_years:
            present += cost / (1.0 + discount_rate) ** t
            t += life
        return present

    def mean_stress_over(self, soc_trajectory, weights=None) -> float:
        """Duty-weighted mean of the depth-of-discharge stress factor.

        `weights` should be the throughput in each interval, so that stress is
        averaged over *energy moved* rather than over wall-clock time: an hour
        sitting idle at a deep state of charge damages nothing.
        """
        soc = np.asarray(soc_trajectory, dtype=float)
        mid = 0.5 * (self.p.soc_max + self.p.soc_min)
        depth = 2.0 * np.abs(soc - mid) / self.p.dod_reference
        stress = self._dod_normalisation * np.maximum(
            depth, self.p.dod_stress_floor) ** (self.p.dod_exponent - 1.0)
        if weights is None:
            return float(np.mean(stress))
        w = np.asarray(weights, dtype=float)
        n = min(len(w), len(stress))
        return float(np.average(stress[:n], weights=w[:n])) if w[:n].sum() > 0 \
            else float(np.mean(stress))

    def degradation_cost_EUR(self, efc_used: float) -> float:
        return self.cost_per_efc_EUR() * efc_used

    def remaining_life_fraction(self, fade) -> float:
        """1.0 at beginning of life, 0.0 at the end-of-life fade threshold."""
        return float(np.clip(1.0 - fade / self.p.end_of_life_fade, 0.0, 1.0))
