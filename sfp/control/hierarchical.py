"""The full hierarchical controller: planner over NMPC, coupled by a price.

This is the thing the project is actually about. Two optimisers on different
clocks, and exactly one number passing between them:

    [L2] economic planner    3 days, graded grid, re-solved every 3 h
              |
              |  lambda(t)        the shadow price of electricity, EUR/kWh
              |  x^target         where the buffers should be
              |  pi_i             what a banked mole is worth
              v
    [L3] inner NMPC          1 hour at 5 min, re-solved every control step
              |
              |  setpoints
              v
    [L4] DC bus + interlocks

The planner never sees a five-minute decision and the NMPC never sees a
three-day horizon. That is the point: the multiscale problem is dissolved by the
price rather than by forcing one optimiser to span both scales.

What happens when a layer fails
-------------------------------
Deliberate, and different for each.

**No plan yet, or the planner failed**: the NMPC runs on the last price it had.
If there has never been one, the controller holds everything off rather than
inventing a schedule -- a silent fallback here would look like a working
controller and would quietly become the thing being measured.

**The NMPC failed**: fall back to the planner's own schedule for that interval.
It is coarser and it is stale, but it is a feasible plan from a converged solve,
which is a great deal better than the last setpoint held indefinitely.

Both paths are counted and reported, because a controller that is silently
running on its fallback is not the controller anyone thinks is being measured.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sfp.control.base import ControlContext, Controller
from sfp.control.nmpc import InnerNMPC
from sfp.control.planner import EconomicPlanner
from sfp.control.planner_model import zi
from sfp.sim.bus import Request

#: Reduced-model states that carry value, mapped to the full state name the
#: NMPC's terminal term uses. These are the inventories the planner has an
#: opinion about; the fast states it eliminated are not its business.
#:
#: Every storable quantity appears exactly once, including the battery. The
#: chemical buffers are priced at the methane they can become; the battery is
#: priced from lambda -- see `_targets_from` for why it must be a *forward*
#: lambda rather than the current one.
TARGET_STATES: dict[str, str] = {
    "n_caco3": "solids.n_caco3",
    "n_h2": "gas.n_h2",
    "n_co2": "gas.n_co2",
    "soc": "battery.soc",
}

#: How far ahead to look for the price that makes stored energy worth having.
#: Long enough to reach the evening from a midday solve.
STORAGE_LOOKAHEAD_S: float = 12 * 3600.0


class HierarchicalController(Controller):
    """Economic planner over an inner NMPC, coordinated by lambda."""

    name = "hierarchical"
    description = (
        "Hierarchical economic MPC: a 3-day planner publishes the shadow price "
        "of electricity; a 1-hour NMPC on the full 16-state model buys energy "
        "at that price."
    )

    def __init__(
        self,
        *,
        planner: EconomicPlanner | None = None,
        nmpc_horizon_s: float = 3600.0,
        nmpc_dt_s: float = 300.0,
        nmpc_max_iter: int = 300,
        name: str | None = None,
    ) -> None:
        self.planner = planner or EconomicPlanner()
        self.nmpc_horizon_s = float(nmpc_horizon_s)
        self.nmpc_dt_s = float(nmpc_dt_s)
        self.nmpc_max_iter = int(nmpc_max_iter)
        if name:
            self.name = name

        self.nmpc: InnerNMPC | None = None
        self._last_u: np.ndarray | None = None
        self._diagnostics: dict[str, float] = {}
        self._nmpc_failures = 0
        self._nmpc_solves = 0

    # --- lifecycle --------------------------------------------------------
    def reset(self, context: ControlContext) -> None:
        super().reset(context)
        self.planner.reset(context)
        self.nmpc = InnerNMPC(
            context.plant,
            self.planner.economics,
            horizon_s=self.nmpc_horizon_s,
            dt_s=self.nmpc_dt_s,
            max_iter=self.nmpc_max_iter,
        )
        self._last_u = None
        self._diagnostics = {}
        self._nmpc_failures = 0
        self._nmpc_solves = 0

    # --- the control law --------------------------------------------------
    def act(self, t, state, measurement, forecast=None) -> Request:
        # 1. keep the plan fresh. The planner decides its own cadence.
        plan_request = self.planner.act(t, state, measurement, forecast)
        plan = self.planner.plan
        if plan is None:
            self._diagnostics.update(nmpc_used=0.0, nmpc_failed=0.0)
            return plan_request

        # 2. hand the price and the targets down
        x0 = self.context.plant.join(state)
        prices = self._price_window(plan, t)
        weather = self._weather_window(t, forecast)
        targets, terminal_prices = self._targets_from(plan, t)

        solution = self.nmpc.solve(
            t, x0, prices, weather,
            targets=targets, terminal_prices=terminal_prices,
            last_u=self._last_u,
        )
        self._nmpc_solves += 1

        if solution.stats is None or not solution.stats.success:
            self._nmpc_failures += 1
            self._diagnostics.update(
                nmpc_used=0.0, nmpc_failed=1.0,
                nmpc_failures=float(self._nmpc_failures),
                plan_lambda_EUR_per_kWh=float(prices[0]),
            )
            # the planner's own schedule is coarse and stale but converged
            return plan_request

        self._last_u = solution.controls[0] if solution.controls is not None else None
        self._diagnostics.update(
            nmpc_used=1.0,
            nmpc_failed=0.0,
            nmpc_failures=float(self._nmpc_failures),
            nmpc_iterations=float(solution.stats.iterations),
            nmpc_solve_time_s=float(solution.stats.wall_time_s),
            nmpc_objective_EUR=solution.objective_EUR,
            plan_lambda_EUR_per_kWh=solution.price_EUR_per_kWh,
        )

        return Request(
            setpoints=solution.setpoints,
            enables=solution.enables,
            battery_charge_W=solution.battery_charge_W,
            battery_discharge_W=solution.battery_discharge_W,
            curtail_fraction=solution.curtail_fraction,
        )

    # --- the hand-off -----------------------------------------------------
    def _price_window(self, plan, t: float) -> np.ndarray:
        """Lambda at each NMPC interval, read off the plan.

        Sampled from the plan rather than interpolated: lambda is a dual and is
        genuinely piecewise-constant on the planner's grid, so smoothing it would
        invent a price the planner never computed.
        """
        n = self.nmpc.n_steps
        return np.array(
            [plan.price_at(t + k * self.nmpc_dt_s) for k in range(n)], dtype=float
        )

    def _weather_window(self, t: float, forecast) -> list[dict[str, Any]]:
        series = forecast if forecast is not None else self.context.forecast
        dense = series.densify(self.nmpc_dt_s, None)
        offsets = dense["time_s"].to_numpy()
        start = max(int(np.searchsorted(offsets, t, side="right")) - 1, 0)
        rows = []
        for k in range(self.nmpc.n_steps):
            i = min(start + k, len(dense) - 1)
            rows.append(dense.iloc[i].to_dict())
        return rows

    def _targets_from(self, plan, t: float):
        """Inventory targets and their marginal values, from the plan.

        The targets are where the plan expects the buffers to be at the end of
        the NMPC's horizon. The prices are the planner's own marginal valuations
        -- the same numbers its terminal term used -- so the two layers agree
        about what a banked mole is worth and the NMPC has no incentive to spend
        inventory the planner was saving.
        """
        horizon_end = t + self.nmpc_horizon_s
        z_target = plan.target_at(horizon_end)
        price = (self.planner.economics.p.methane_price_per_kg
                 * 16.0425e-3 * self.planner.terminal_value_fraction)

        targets, prices = {}, {}
        for reduced, full in TARGET_STATES.items():
            targets[full] = float(z_target[zi(reduced)])
        # stoichiometry: 4 H2 -> 1 CH4, 1 CO2 -> 1 CH4, 1 CaCO3 -> 1 CH4
        prices["gas.n_h2"] = price / 4.0
        prices["gas.n_co2"] = price
        prices["solids.n_caco3"] = price

        # The battery is priced through the same chain as everything else: the
        # methane its stored energy can eventually become, at the electrolyser's
        # specific energy. Pricing it from a forward lambda instead is arguably
        # more principled and was tried; at 0.054 EUR/kWh across a 1500 kWh pack
        # it puts ~81 EUR on the terminal term against stage costs of ~5 EUR, so
        # it dominated the objective by fifteen to one, drove the solution into a
        # corner, and stopped the NMPC converging at all.
        kwh_per_soc = self.context.plant["battery"].nominal_energy_J / 3.6e6
        prices["battery.soc"] = price * kwh_per_soc / 56.4 / 2.016e-3 / 4.0
        return targets, prices

    # --- reporting --------------------------------------------------------
    def diagnostics(self) -> dict[str, float]:
        out = dict(self.planner.diagnostics())
        out.update(self._diagnostics)
        return out
