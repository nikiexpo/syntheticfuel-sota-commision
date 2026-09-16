"""Levelised cost of methane -- the objective everything is judged against.

The project README noted that the challenge brief never defines what "operate
efficiently" means, and that this is a real problem: maximising methane output,
maximising round-trip energy efficiency and maximising plant utilisation are
three different policies that disagree about what to do on a cloudy Tuesday.

This module picks one and commits to it:

    LCOM = (annualised capex + annual opex) / annual methane production   [EUR/kg]

It does three jobs at once. It makes "efficient" precise. It ranks control
strategies on a single axis. And it is exactly the number the siting tool has to
report, so the digital twin and the siting study share an objective rather than
each inventing their own.

The controller does not optimise LCOM directly -- capex is sunk by then. It
optimises the *marginal* terms, which is the same thing for scheduling purposes:

    revenue         methane produced x price
    less  wear      battery throughput, process starts, (from M1) sorbent cycles
    less  consumables water

`marginal_objective_EUR` implements that, and is the term the M4 planner
maximises. Curtailment is deliberately not in it -- see the note in the YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sfp.params import ParamSet, load_params
from sfp.units import LHV_CH4_MASS, M_H2O


@dataclass
class CapexBreakdown:
    """Installed cost by component, EUR."""

    pv: float = 0.0
    battery: float = 0.0
    process: float = 0.0

    @property
    def total(self) -> float:
        return self.pv + self.battery + self.process

    def as_dict(self) -> dict[str, float]:
        return {"pv": self.pv, "battery": self.battery, "process": self.process, "total": self.total}


class Economics:
    """Cost model shared by the controller objective and the report."""

    def __init__(self, params: ParamSet | None = None) -> None:
        self.p = params or load_params("economics")

    # --- capital ----------------------------------------------------------
    def capex(self, pv, battery, process) -> CapexBreakdown:
        return CapexBreakdown(
            pv=self.p.pv_capex_per_kwp * pv.p.capacity_kwp,
            battery=battery.capex_EUR(),
            process=self.p.process_capex_per_kw * process.p.rated_power_kw,
        )

    def capex_full(self, plant) -> CapexBreakdown:
        """Installed cost of a fully assembled plant.

        The process chain is costed on its aggregate electrical rating, which is
        crude but consistent: a per-subsystem cost model would imply a precision
        the underlying figures do not have. The electrolyser dominates in reality
        and it dominates here, since it is most of the rated load.
        """
        rated_kw = 0.0
        for key, sub in plant:
            if key in ("pv", "battery") or sub.n_inputs < 2:
                continue
            rated_kw += float(sub.power_for_setpoint(sub.initial_state(), 1.0, 1.0, {})) / 1e3
        return CapexBreakdown(
            pv=self.p.pv_capex_per_kwp * plant["pv"].p.capacity_kwp,
            battery=plant["battery"].capex_EUR(),
            process=self.p.process_capex_per_kw * rated_kw,
        )

    def capital_recovery_factor(self) -> float:
        """Annuity factor turning a lump-sum capex into an equivalent annual cost."""
        r = self.p.discount_rate
        n = self.p.project_lifetime_years
        if r <= 0.0:
            return 1.0 / n
        return r * (1.0 + r) ** n / ((1.0 + r) ** n - 1.0)

    def annualised_capex(self, capex: CapexBreakdown) -> float:
        return capex.total * self.capital_recovery_factor()

    def annual_fixed_opex(self, capex: CapexBreakdown) -> float:
        return capex.total * self.p.fixed_opex_fraction

    # --- marginal costs the controller actually sees ---------------------
    def water_cost_EUR(self, water_kg: float) -> float:
        return self.p.water_cost_per_m3 * water_kg / 1000.0

    def startup_cost_EUR(self, starts: float) -> float:
        return self.p.startup_cost_EUR * starts

    def revenue_EUR(self, ch4_kg: float) -> float:
        return self.p.methane_price_per_kg * ch4_kg

    def marginal_objective_EUR(
        self,
        *,
        ch4_kg: float,
        battery_efc: float = 0.0,
        battery_cost_per_efc: float = 0.0,
        starts: float = 0.0,
        water_kg: float = 0.0,
        co2_captured_kg: float = 0.0,
    ) -> float:
        """Operating profit over one interval, EUR. This is what the planner maximises.

        Positive is good. Every term here is something the controller can change
        *today*; sunk capital deliberately does not appear.
        """
        return (
            self.revenue_EUR(ch4_kg)
            + self.p.co2_credit_per_kg * co2_captured_kg
            - battery_efc * battery_cost_per_efc
            - self.startup_cost_EUR(starts)
            - self.water_cost_EUR(water_kg)
        )

    # --- levelised cost ---------------------------------------------------
    def lcom(
        self,
        capex: CapexBreakdown,
        annual_ch4_kg: float,
        annual_water_kg: float = 0.0,
        annual_battery_replacement_EUR: float = 0.0,
    ) -> float:
        """Levelised cost of methane, EUR/kg. `inf` if the plant makes nothing."""
        if annual_ch4_kg <= 0.0:
            return float("inf")
        annual_cost = (
            self.annualised_capex(capex)
            + self.annual_fixed_opex(capex)
            + self.water_cost_EUR(annual_water_kg)
            + annual_battery_replacement_EUR
        )
        return annual_cost / annual_ch4_kg

    def lcom_per_mwh(self, lcom_per_kg: float) -> float:
        """LCOM expressed per MWh of methane LHV, for comparison with energy prices."""
        mwh_per_kg = LHV_CH4_MASS / 3.6e9
        return lcom_per_kg / mwh_per_kg

    def payback_years(self, capex: CapexBreakdown, annual_ch4_kg: float, annual_opex: float | None = None) -> float:
        """Simple (undiscounted) payback. `inf` if the plant never covers its opex."""
        opex = self.annual_fixed_opex(capex) if annual_opex is None else annual_opex
        annual_margin = self.revenue_EUR(annual_ch4_kg) - opex
        if annual_margin <= 0.0:
            return float("inf")
        return capex.total / annual_margin

    def discounted_payback_years(
        self,
        capex_EUR: float,
        annual_cash_EUR: float,
        *,
        replacement_EUR: float = 0.0,
        replacement_interval_years: float = float("inf"),
        horizon_years: float | None = None,
        steps_per_year: int = 12,
    ) -> float:
        """Years until discounted cash flow first repays the investment.

        Preferred over a break-even *price* wherever price is a swept axis: the
        break-even price solves for the very quantity the sweep is varying, and
        collapses it. Payback is defined at every point of the sweep and answers
        "how long", not just "does it".

        It is also more honest than an annuitised profit rate. Net EUR/day at a
        capital recovery factor already assumes the project's full life at the
        discount rate, so a positive figure means only "repays within 25 years at
        7 %" -- a binary, dressed as a continuous number.

        **`annual_cash_EUR` must exclude any accrued wear charge.** Operating
        profit as the controller computes it subtracts `cost_per_kWh_delivered`
        on every kWh moved, which is an accrual for a replacement that has not
        happened yet. A cash-flow model pays for replacements when they occur,
        as `replacement_EUR` lumps -- so the accrual has to be added back first
        or the battery is charged twice.

        Returns `inf` if the investment is not repaid within `horizon_years`
        (the project lifetime by default), which includes every case where the
        cash flow is negative.
        """
        horizon = float(self.p.project_lifetime_years) if horizon_years is None \
            else float(horizon_years)
        if annual_cash_EUR <= 0.0:
            return float("inf")

        r = float(self.p.discount_rate)
        dt = 1.0 / steps_per_year
        cumulative = -float(capex_EUR)
        next_replacement = replacement_interval_years

        t = 0.0
        while t < horizon:
            t += dt
            discount = (1.0 + r) ** -t
            cumulative += annual_cash_EUR * dt * discount
            if replacement_EUR > 0.0 and t >= next_replacement:
                cumulative -= replacement_EUR * discount
                next_replacement += replacement_interval_years
            if cumulative >= 0.0:
                return t
        return float("inf")

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def stoichiometric_water_kg(ch4_kg: float) -> float:
        """Net make-up water per kg of methane, after condensate recycle.

        CO2 + 4 H2 -> CH4 + 2 H2O, and the 4 H2 cost 4 H2O at the electrolyser.
        Recovering the 2 H2O of product leaves a net 2 H2O per CH4, i.e.
        2 * 18.015 / 16.043 = 2.246 kg water per kg methane.
        """
        return ch4_kg * 2.0 * M_H2O / 16.0425e-3

    def annualise(self, value: float, duration_s: float) -> float:
        """Scale a run total to a year. Crude and clearly labelled as such.

        A 10-day summer window annualised by 365/10 overstates a European
        plant's yield badly, so the reporting layer annualises from a full TMY
        run wherever the number is used for LCOM. This helper exists so that
        scaling is always explicit at the call site rather than buried.
        """
        if duration_s <= 0.0:
            return 0.0
        return value * 31_536_000.0 / duration_s
