"""The shipped controller: economic dispatch MILP over an NMPC safety filter.

    [L2] economic dispatch MILP   240 h hourly, re-solved every 3 h
              |                   ALL the economics live here
              |  bands            what each machine may do, as a bound
              |  u_plan           the action it proposes
              v
    [L3] inner NMPC               5 min at 1 min, full 16-state model
              |                   NO economics: edit u_plan just enough that the
              |  setpoints        real dynamics are satisfied
              v
    [L4] DC bus + interlocks

**The outer layer makes every economic decision and no dynamic one; the inner
layer makes every dynamic decision and no economic one.** Not one euro appears
below L2. The inner objective is ||u_0 - u_plan||^2 plus exact-penalty slacks,
subject to the real sixteen-state dynamics, the state boxes, rate limits and the
dispatch bands. Full formulation in `docs/DISPATCH_NMPC.md`.

Given that the LP already produced a feasible schedule, that buys sub-minute
resolution inside an hourly plan, feasibility against dynamics the LP linearised
away (the kiln's kinetics, the reactor's light-off, the buffer tapers), and a
*shed prediction* -- the inner layer saying, before the fact, that the plan has
committed more load than the bus can carry.

The battery and the curtailment fraction are deliberately neither banded nor
referenced away from the plan's own values: they are the degrees of freedom that
close the bus in real time against weather forecast an hour earlier.

Commitment is handed down as a **bound, not a flag**. As a flag it is a free
variable on a near-discontinuous gate, and an enable left at its initial guess
draws idle load around the clock -- measured, 26.4 kW from four machines the
plan had shut down, 317 kWh over a twelve-hour night.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sfp.control.base import ControlContext, Controller
from sfp.control.dispatch import EconomicDispatch
from sfp.control.dispatch_model import zi
from sfp.control.nmpc import InnerNMPC
from sfp.sim.bus import Request

#: Buffer states the dispatch layer has an opinion about, mapped to the full
#: state name the NMPC's terminal term uses.
TARGET_STATES: dict[str, str] = {
    "n_caco3": "solids.n_caco3",
    "n_h2": "gas.n_h2",
    "n_co2": "gas.n_co2",
    "soc": "battery.soc",
}


class DispatchNMPCController(Controller):
    """The LP dispatch layer over the inner NMPC."""

    name = "dispatch-nmpc"
    description = (
        "Economic dispatch LP (240 h, hourly) publishing dispatch bands to a "
        "1-hour NMPC on the full 16-state plant model."
    )

    def __init__(
        self,
        *,
        dispatch: EconomicDispatch | None = None,
        #: Five minutes. In filter mode the horizon is purely a feasibility
        #: certificate, so it does not need to be long. Measured against 1 h on
        #: identical busy states: half the iterations (24.8 vs 51.4), 2.4x
        #: faster, 360 columns against 423, same action to within 1 %.
        nmpc_horizon_s: float = 300.0,
        #: One minute, matching the simulator's integration step, so the
        #: dynamics constraint and the plant agree by construction.
        nmpc_dt_s: float = 60.0,
        #: Pinned at 600 s rather than following `nmpc_dt_s`, which would solve
        #: on every control step instead of every other one.
        #:
        #: **Caveat:** the k = 0 rate-limit row then spans 600 s while the k > 0
        #: rows span 60 s, and `rate_limit` is one per-interval number. It is
        #: set for k = 0, where it binds against the applied control, so the
        #: tail is ten times too permissive -- a weaker feasibility certificate,
        #: but the same action. The clean fix is slew per second, scaled per row.
        resolve_interval_s: float | None = 600.0,
        nmpc_max_iter: int = 500,
        #: `"filter"` is the shipped formulation (`docs/DISPATCH_NMPC.md`):
        #: minimal edit to the plan's action, unbounded slack on every
        #: conflicting inequality, no economics.
        objective: str = "filter",
        name: str | None = None,
    ) -> None:
        self.objective = objective
        self.dispatch = dispatch or EconomicDispatch()
        self.nmpc_horizon_s = float(nmpc_horizon_s)
        self.nmpc_dt_s = float(nmpc_dt_s)
        self.resolve_interval_s = (self.nmpc_dt_s if resolve_interval_s is None
                                   else float(resolve_interval_s))
        self.nmpc_max_iter = int(nmpc_max_iter)
        if name:
            self.name = name

        self.nmpc: InnerNMPC | None = None
        self._last_u: np.ndarray | None = None
        self._diagnostics: dict[str, float] = {}
        self._nmpc_failures = 0
        self._nmpc_solves = 0
        self._held: Request | None = None
        self._next_solve_s = -np.inf
        self._failure_reasons: dict[str, int] = {}

    # --- lifecycle --------------------------------------------------------
    def reset(self, context: ControlContext) -> None:
        super().reset(context)
        self.dispatch.reset(context)
        self.nmpc = InnerNMPC(
            context.plant,
            self.dispatch.economics,
            horizon_s=self.nmpc_horizon_s,
            dt_s=self.nmpc_dt_s,
            max_iter=self.nmpc_max_iter,
            objective=self.objective,
        )
        self._last_u = None
        self._diagnostics = {}
        self._nmpc_failures = 0
        self._nmpc_solves = 0
        self._held = None
        self._next_solve_s = -np.inf
        self._failure_reasons = {}

    # --- the control law --------------------------------------------------
    def act(self, t, state, measurement, forecast=None) -> Request:
        plan_request = self.dispatch.act(t, state, measurement, forecast)
        plan = self.dispatch.plan
        if plan is None:
            self._diagnostics.update(nmpc_used=0.0, nmpc_failed=0.0)
            return plan_request

        # Hold the last solution across the control steps inside one interval:
        # re-solving would pose the same problem against a state that has
        # barely moved.
        if t < self._next_solve_s and self._held is not None:
            self._diagnostics.update(nmpc_used=1.0, nmpc_failed=0.0, nmpc_held=1.0)
            return self._held
        self._next_solve_s = t + self.resolve_interval_s

        x0 = self.context.plant.join(state)
        prices = self._price_window(plan, t)
        weather = self._weather_window(t, forecast)
        targets = self._targets_from(plan, t)
        bands = plan.bands_at(t + self.nmpc_horizon_s * 0.5)

        solution = self.nmpc.solve(
            t, x0, prices, weather,
            targets=targets,
            bands=bands, last_u=self._last_u,
            reference_u=(self._reference_action(plan, t)
                         if self.objective == "filter" else None),
        )
        self._nmpc_solves += 1

        if solution.stats is None or not solution.stats.success:
            self._nmpc_failures += 1
            # Record *why*, not just that: a flag-only failure path is how 83 %
            # of solves once failed silently behind a plausible-looking fallback.
            status = "none" if solution.stats is None else str(solution.stats.status)
            self._failure_reasons[status] = self._failure_reasons.get(status, 0) + 1
            self._diagnostics.update(
                nmpc_used=0.0, nmpc_failed=1.0,
                nmpc_failures=float(self._nmpc_failures),
                nmpc_iterations=float(
                    getattr(solution.stats, "iterations", float("nan"))
                    if solution.stats else float("nan")),
                plan_lambda_EUR_per_kWh=float(prices[0]),
            )
            # The dispatch layer's own schedule: coarse and an hour stale, but a
            # feasible plan from a converged solve.
            self._held = plan_request
            return plan_request

        self._last_u = solution.controls[0] if solution.controls is not None else None
        self._diagnostics.update(
            nmpc_used=1.0,
            nmpc_failed=0.0,
            nmpc_held=0.0,
            nmpc_failures=float(self._nmpc_failures),
            nmpc_iterations=float(solution.stats.iterations),
            nmpc_solve_time_s=float(solution.stats.wall_time_s),
            nmpc_objective_EUR=solution.objective_EUR,
            nmpc_predicted_shed_kWh=solution.predicted_shed_kWh,
            nmpc_edit=solution.edit,
            **{f"bind_{k}": v for k, v in solution.binding.items()},
            **{f"cf_{k}": v for k, v in solution.counterfactual.items()},
            plan_lambda_EUR_per_kWh=solution.price_EUR_per_kWh,
        )

        self._held = Request(
            setpoints=solution.setpoints,
            enables=solution.enables,
            battery_charge_W=solution.battery_charge_W,
            battery_discharge_W=solution.battery_discharge_W,
            curtail_fraction=solution.curtail_fraction,
        )
        return self._held

    # --- the hand-off -----------------------------------------------------
    def _reference_action(self, plan, t: float) -> np.ndarray:
        """The complete action the dispatch layer proposes, in plant units.

        A safety filter edits *all* of it -- the four process setpoints with
        their enables, both battery channels and the curtailment fraction --
        because a channel left out of the reference is one the inner layer is
        re-deciding rather than filtering.
        """
        nmpc = self.nmpc
        u = np.zeros(nmpc._n_u, dtype=float)
        k = plan.index_at(t)
        bands = plan.bands_at(t)

        offset = 0
        for key in nmpc._input_keys:
            size = nmpc._input_sizes[key]
            if key == "pv":
                u[offset] = (float(plan.curtail[k]) if len(plan.curtail) else 0.0)
            elif key == "battery":
                if len(plan.battery_W):
                    u[offset] = float(plan.battery_W[k, 0])
                    u[offset + 1] = float(plan.battery_W[k, 1])
            else:
                band = bands.get(key)
                if band is not None and band.committed:
                    u[offset] = float(band.setpoint_max)
                    if size > 1:
                        u[offset + 1] = 1.0
            offset += size
        return np.clip(u, nmpc._u_lo, nmpc._u_hi)

    def _price_window(self, plan, t: float) -> np.ndarray:
        """Lambda at each NMPC interval.

        Reporting and the failure path only -- the inner stage cost does not
        price energy. Sampled rather than interpolated, because a dual is
        piecewise-constant on the dispatch layer's grid.
        """
        return np.array(
            [plan.price_at(t + k * self.nmpc_dt_s) for k in range(self.nmpc.n_steps)],
            dtype=float,
        )

    def _weather_window(self, t: float, forecast) -> list[dict[str, Any]]:
        series = forecast if forecast is not None else self.context.forecast
        dense = series.densify(self.nmpc_dt_s, None)
        offsets = dense["time_s"].to_numpy()
        start = max(int(np.searchsorted(offsets, t, side="right")) - 1, 0)
        return [dense.iloc[min(start + k, len(dense) - 1)].to_dict()
                for k in range(self.nmpc.n_steps)]

    def _targets_from(self, plan, t: float):
        """Where the plan expects the buffers at the end of the inner horizon.

        Targets only, never prices: what a banked mole is worth was settled by
        the dispatch LP when it chose the trajectory, and re-deriving it here
        would put an economic decision in the layer that must make none. Unused
        in filter mode, which carries no terminal term.
        """
        z_target = plan.target_at(t + self.nmpc_horizon_s)
        return {full: float(z_target[zi(reduced)])
                for reduced, full in TARGET_STATES.items()}

    # --- reporting --------------------------------------------------------
    def diagnostics(self) -> dict[str, float]:
        out = dict(self.dispatch.diagnostics())
        out.update(self._diagnostics)
        return out
