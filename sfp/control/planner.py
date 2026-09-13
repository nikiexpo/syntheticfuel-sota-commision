"""The economic planner (layer 2) -- the outer problem of the hierarchy.

Decides, over a three-day horizon on a graded time grid, how much energy goes
into each chain and how much inventory to bank, re-solving every three hours on a
receding horizon. It hands three things downward:

    inventory targets      where the buffers should be at the next tick
    commitment schedule    which subsystems should be running
    lambda                 the shadow price of electricity, EUR/kWh

The third is the important one, and it is the reason the hierarchy works. Lambda
is the dual of the energy-balance constraint: the marginal value, in euro, of one
more kilowatt-hour on the bus at that hour. Handed down, it lets the inner NMPC
solve a *local* problem -- make methane, buy energy at lambda -- without ever
seeing the planner's time grid or its ten-day horizon. That is Lagrangian
decomposition rather than a heuristic hand-off, and it degrades gracefully: a
stale plan still yields a sensible price, where a stale setpoint trajectory does
not.

At M3 the planner drives the plant directly, so the numbers it produces are
visible immediately and its price signal can be sanity-checked against the
physics before anything depends on it. The NMPC slots underneath at M4.

Structure of the NLP
--------------------
Multiple shooting on the reduced seven-state model: the state at every interval
boundary is a decision variable and the dynamics are equality constraints. That
is more variables than single shooting but far better conditioned over a long
horizon, and it makes every buffer bound a simple box rather than a deeply nested
expression.

    variables    z_0..z_N   (7 each)     states at each interval boundary
                 u_0..u_N-1 (14 each)    setpoints, enables, battery,
                                         curtailment, vents, water make-up
                 starts     (N-1)        reactor light-off epigraph
    constraints  dynamics                z_{k+1} = RK4(z_k, u_k, dt_k)
                 bus_balance             demand == supply          <- lambda
                 sorbent_capacity        n_CaCO3 <= n_tot * X(N)
                 commitment              setpoint <= enable
                 min_load                electrolyser >= i_min * enable
                 start_counter           s_k >= e_k - e_{k-1}
                 initial_state           z_0 == current estimate

Everything is solved in non-dimensional variables: each state is divided by a
characteristic magnitude, each control by its own upper bound, and every
constraint row by a characteristic scale. See `_build`.

The grid is graded rather than uniform -- hourly through the first day, then
three- and six-hourly. Only the near term is ever implemented, because the plan
is rebuilt every three hours; the far end exists to value the terminal
inventories. `bookkeeping/04_PLANNER_TRACTABILITY.md` records what that bought
and what it cost.

Curtailment is a free variable rather than a residue here, which is the one place
the planner's world differs structurally from the bus's. It has to be: the
planner must be *able* to choose to spill, since deciding what not to absorb is a
real scheduling decision. The bus then computes curtailment as a residue when the
plan meets reality.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping

import casadi as ca
import numpy as np

from sfp.control.base import ControlContext, Controller
from sfp.control.planner_model import COMMITTED, N_U, N_Z, PlannerModel, ui, zi
from sfp.economics import Economics
from sfp.sim.bus import Request
from sfp.solvers import NLPBuilder, SolverBackend, get_backend
from sfp.units import M_CH4, M_H2O

#: Names the planner uses for the blocks it later reads back by name.
BUS_BALANCE = "bus_balance"


@dataclass
class Plan:
    """A solved schedule on the planner's (non-uniform) grid."""

    t0_s: float
    dt_s: np.ndarray             # (N,) interval lengths, seconds
    states: np.ndarray           # (N+1, N_Z)
    controls: np.ndarray         # (N, N_U)
    lambda_EUR_per_kWh: np.ndarray   # (N,)
    objective_EUR: float
    solve_stats: Any = None
    horizon_s: float = 0.0

    def __post_init__(self) -> None:
        self.dt_s = np.atleast_1d(np.asarray(self.dt_s, dtype=float))
        #: left edge of each interval, relative to t0
        self.edges_s = np.concatenate(([0.0], np.cumsum(self.dt_s)))

    def index_at(self, t_s: float) -> int:
        """Index of the planning interval containing absolute time `t_s`.

        A search rather than a division, because the grid is non-uniform.
        """
        k = int(np.searchsorted(self.edges_s, t_s - self.t0_s, side="right")) - 1
        return int(np.clip(k, 0, len(self.controls) - 1))

    def control_at(self, t_s: float) -> np.ndarray:
        return self.controls[self.index_at(t_s)]

    def price_at(self, t_s: float) -> float:
        return float(self.lambda_EUR_per_kWh[self.index_at(t_s)])

    def target_at(self, t_s: float) -> np.ndarray:
        """The state the plan expects at the *end* of the interval containing t."""
        return self.states[min(self.index_at(t_s) + 1, len(self.states) - 1)]

    @property
    def age_s(self) -> float:
        return self.horizon_s


class EconomicPlanner(Controller):
    """Receding-horizon economic MPC over the reduced model."""

    name = "planner"
    description = (
        "Economic MPC: 3-day horizon on a graded grid, re-solved every 3 h, on a "
        "seven-state reduced model. Publishes the shadow price of electricity."
    )

    #: Grid spacing, as (hours covered, interval length in hours). Near-term
    #: decisions are the ones actually implemented, so they get the resolution;
    #: the far end exists to value the terminal inventories correctly and can be
    #: coarse. Three days at this grading is 24 + 16 = 40 intervals against 72
    #: uniform ones, and the NLP cost is steeply superlinear in the step count.
    DEFAULT_GRADING: tuple[tuple[float, float], ...] = (
        (24.0, 1.0),    # first day, hourly
        (48.0, 3.0),    # days 2-3, three-hourly
        (96.0, 6.0),    # beyond, six-hourly
    )

    def __init__(
        self,
        *,
        #: Three days, not the seven the formulation asks for.
        #:
        #: 168 h was built, measured and does not converge: 56 intervals, 1238
        #: variables, 3000 iterations and 452 s ending at a constraint violation
        #: of 2e-3, whether warm-started from 72 h or not. 72 h converges
        #: reliably in ~1600 iterations and ~155 s cold, and much faster warm.
        #:
        #: Raising this is safe -- the planner shortens its own target to the
        #: longest horizon that converged and warns once -- but the default is
        #: set to what works so that no run pays 452 s for a failed stage.
        #: `bookkeeping/04_PLANNER_TRACTABILITY.md` has the measurements.
        horizon_hours: int = 72,
        replan_interval_s: float = 3 * 3600.0,
        terminal_value_fraction: float = 0.5,
        sabatier_start_cost_EUR: float = 2.0,
        backend: SolverBackend | str = "ipopt",
        max_iter: int = 3000,
        tol: float = 1e-4,
        commit_threshold: float = 0.5,
        grading: tuple[tuple[float, float], ...] | None = None,
        homotopy_hours: tuple[int, ...] = (24,),
        name: str | None = None,
    ) -> None:
        self.horizon_hours = int(horizon_hours)
        self.grading = grading if grading is not None else self.DEFAULT_GRADING
        self.homotopy_hours = tuple(int(h) for h in homotopy_hours)
        self.replan_interval_s = float(replan_interval_s)
        self.terminal_value_fraction = float(terminal_value_fraction)
        self.sabatier_start_cost_EUR = float(sabatier_start_cost_EUR)
        self.commit_threshold = float(commit_threshold)
        self.max_iter = int(max_iter)
        # A planning tolerance, not a physics tolerance. IPOPT's default 1e-6
        # spends hundreds of iterations polishing a schedule whose inputs are a
        # weather forecast; the model error dwarfs the solver error by orders of
        # magnitude long before that.
        #
        # At the full horizon this is what decides whether there is a plan at
        # all. Measured at 168 h, the solver reached a constraint violation of
        # 6e-9 -- primal-feasible by any standard anyone cares about -- and then
        # spent its whole iteration budget failing to tighten the *dual* to 1e-5.
        # The schedule was usable; only the convergence test disagreed.
        self.tol = float(tol)
        self._backend_spec = backend
        if name:
            self.name = name

        self.model: PlannerModel | None = None
        self.plan: Plan | None = None
        self._last_solution_x: np.ndarray | None = None
        self._next_replan_s = -np.inf
        self._diagnostics: dict[str, float] = {}
        self._failures = 0
        self._solves = 0

    # --- lifecycle --------------------------------------------------------
    def reset(self, context: ControlContext) -> None:
        super().reset(context)
        self.model = PlannerModel(context.plant)
        self.economics = context.economics or Economics()
        self.backend = (
            self._backend_spec
            if isinstance(self._backend_spec, SolverBackend)
            else get_backend(self._backend_spec, max_iter=self.max_iter,
                             tol=self.tol, acceptable_tol=self.tol * 100.0)
        )
        self.plan = None
        self._last_solution_x = None
        self._target_hours = self.horizon_hours
        self._reported_shortfalls: set[int] = set()
        self._next_replan_s = -np.inf
        self._diagnostics = {}
        self._failures = 0
        self._solves = 0
        self._nlp = None
        self._nlp_signature: tuple | None = None

    # --- the control law --------------------------------------------------
    def act(self, t, state, measurement, forecast=None) -> Request:
        self._last_t = t
        if t >= self._next_replan_s:
            self._replan(t, state, forecast)
            self._next_replan_s = t + self.replan_interval_s

        if self.plan is None:
            # No usable plan has ever been produced. Rather than invent one,
            # hold everything off and let the diagnostics show it -- a silent
            # fallback here would look like a working controller.
            return Request.all_off()

        return self._request_from(self.plan, t, measurement)

    def _request_from(self, plan: Plan, t: float, measurement: Mapping[str, Any]) -> Request:
        u = plan.control_at(t)
        setpoints, enables = {}, {}
        for key in COMMITTED:
            e = float(u[ui(f"{key}_on")])
            committed = e >= self.commit_threshold
            enables[key] = 1.0 if committed else 0.0
            # The setpoint is the rate; the enable only priced the parasitic
            # load (see PlannerModel.rates). So rounding the commitment up needs
            # no renormalisation -- the setpoint already means what it says.
            raw = float(u[ui(_SETPOINT_OF[key])])
            setpoints[key] = float(np.clip(raw, 0.0, 1.0)) if committed else 0.0

        return Request(
            setpoints=setpoints,
            enables=enables,
            battery_charge_W=float(max(u[ui("battery_charge_W")], 0.0)),
            battery_discharge_W=float(max(u[ui("battery_discharge_W")], 0.0)),
            curtail_fraction=float(np.clip(u[ui("curtail")], 0.0, 1.0)),
        )

    # --- planning ---------------------------------------------------------
    def _replan(self, t: float, state, forecast) -> None:
        started = time.perf_counter()
        model = self.model
        z0 = model.initial_state(state)
        hourly = self._forecast_rows(t, forecast)
        available_hours = len(hourly["pv_available_W"])
        if available_hours == 0:
            return

        target = min(self._target_hours, available_hours)
        stage_log: list[str] = []

        def attempt(warm):
            """Solve the horizon, warm-starting through a homotopy if cold.

            The kiln ignition transient makes this genuinely non-convex, and a
            cold solve straight at the full horizon is a lot to ask of any guess:
            it either takes thousands of iterations or converges to a plant that
            never calcines. Solving a short horizon first and warm-starting the
            next from it walks the solver into the right basin cheaply. The
            homotopy is skipped when a previous plan is available, which is the
            usual case on a receding horizon.
            """
            solution = grid = solved_grid = None
            stages = [] if warm is not None else [
                h for h in self.homotopy_hours if h < target
            ]
            for hours in [*stages, target]:
                grid = self._grid(hours)
                weather = self._aggregate(hourly, grid)
                nlp = self._build(len(grid), z0, weather, grid)

                x0 = None
                if warm is not None:
                    z_w, u_w = self._resample(warm[0], warm[1], warm[2], grid)
                    x0 = self._splice(self._pack(grid, z_w, u_w), nlp, grid,
                                      warm[3])

                stage = self.backend.solve(nlp, x0=x0)
                stage_log.append(
                    f"{hours}h/{len(grid)}i {stage.stats.status}"
                    f" {stage.stats.iterations}it {stage.stats.wall_time_s:.0f}s"
                )
                if not stage.stats.success:
                    break
                solution, solved_grid = stage, grid
                # a freshly solved stage covers its whole horizon
                warm = (grid, *self._unpack(stage.x, len(grid)), float(np.sum(grid)))
            return solution, solved_grid, grid

        warm = self._shifted_warm_start(t)      # (grid, states, controls) or None
        solution, solved_grid, grid = attempt(warm)

        # A warm start is normally the cheapest route and occasionally the worst
        # one: the previous plan can sit in a basin the new weather has made
        # infeasible. Falling back to a cold solve costs one extra homotopy but
        # recovers, where keeping the stale plan would degrade for the rest of
        # the run and report a failure at every tick.
        if solution is None and warm is not None:
            stage_log.append("cold retry")
            solution, solved_grid, grid = attempt(None)

        self._solves += 1
        self._diagnostics["plan_stages"] = " | ".join(stage_log)
        # A later stage may have failed after an earlier one succeeded. Keep the
        # best *converged* horizon rather than the longest attempted one -- a
        # three-day plan that solved is worth more than a seven-day one that did
        # not, and the grid must match the solution it came from.
        grid = solved_grid if solution is not None else grid
        n = len(grid)

        # If the full horizon did not converge but a shorter one did, shorten the
        # target permanently rather than re-attempting the same failure every
        # three hours. Without this the homotopy -- which only runs on a cold
        # start -- would be skipped on every later replan, the single full-horizon
        # solve would fail again, and the planner would run the rest of the
        # simulation on an ever-staler plan while reporting a failure each time.
        if solution is not None:
            achieved = int(round(float(np.sum(grid)) / 3600.0))
            if achieved < target and achieved > 0:
                self._target_hours = achieved
                if achieved not in self._reported_shortfalls:
                    self._reported_shortfalls.add(achieved)
                    warnings.warn(
                        f"planner horizon reduced to {achieved} h: the "
                        f"{target} h problem did not converge. Plans remain "
                        f"valid, but multi-day trades beyond {achieved} h are "
                        f"outside what this planner can see.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        if solution is None or not solution.success:
            self._failures += 1
            # Keep the previous plan. It is stale, but a stale price is still a
            # sensible price -- which is exactly the property price coordination
            # was chosen for. The failure is recorded, never swallowed.
            self._diagnostics.update(
                plan_solve_failed=1.0,
                plan_solve_status=float("nan"),
                plan_failures=float(self._failures),
            )
            return

        # decision variables are non-dimensional; restore physical units
        states = solution.value("z").reshape(n + 1, N_Z) * model.state_scale()
        controls = solution.value("u").reshape(n, N_U) * model.control_scale()

        # Three conversions, all easy to get wrong and all pinned by tests.
        #
        # 1. `lam` on a problem built with `maximise` is already the marginal
        #    value of relaxing the constraint -- no sign flip (see Solution.dual).
        # 2. The balance row was divided by `power_scale`, so its dual is EUR per
        #    scaled unit. Dividing by the same factor returns EUR per watt.
        # 3. One extra watt held for the whole interval is dt/3.6e6 kWh, and on a
        #    graded grid `dt` differs per interval -- dividing by a single scalar
        #    would report the six-hourly tail as six times cheaper than it is.
        lam_per_W = solution.dual(BUS_BALANCE) / model.power_scale()
        lam_kWh = lam_per_W * 3.6e6 / grid

        self.plan = Plan(
            t0_s=t,
            dt_s=grid,
            states=states,
            controls=controls,
            lambda_EUR_per_kWh=lam_kWh,
            objective_EUR=-solution.f,
            solve_stats=solution.stats,
            horizon_s=float(np.sum(grid)),
        )
        self._last_solution_x = solution.x
        self._last_grid = grid
        self._diagnostics.update(
            plan_objective_EUR=-solution.f,
            plan_solve_time_s=time.perf_counter() - started,
            plan_iterations=float(solution.stats.iterations),
            plan_kkt_residual=solution.stats.kkt_residual,
            plan_solve_failed=0.0,
            plan_failures=float(self._failures),
            plan_intervals=float(n),
            plan_lambda_mean=float(np.mean(lam_kWh)),
            plan_lambda_max=float(np.max(lam_kWh)),
        )

    # --- horizon data -----------------------------------------------------
    dt_plan_s: float = 3600.0

    def _grid(self, horizon_hours: float) -> np.ndarray:
        """Interval lengths in seconds for a graded grid over `horizon_hours`."""
        intervals: list[float] = []
        remaining = float(horizon_hours)
        for span_h, step_h in self.grading:
            take = min(span_h, remaining)
            while take > 1e-9:
                dt = min(step_h, take)
                intervals.append(dt)
                take -= dt
                remaining -= dt
            if remaining <= 1e-9:
                break
        # anything past the last band keeps the coarsest spacing
        coarsest = self.grading[-1][1]
        while remaining > 1e-9:
            dt = min(coarsest, remaining)
            intervals.append(dt)
            remaining -= dt
        return np.array(intervals, dtype=float) * 3600.0

    def _aggregate(self, hourly: dict[str, np.ndarray], grid: np.ndarray) -> dict:
        """Average the hourly forecast over each interval of the graded grid.

        Averaging rather than sampling matters for the PV row. A six-hour
        interval sampled at its left edge would take a single instantaneous
        irradiance as the whole block's availability -- at dawn that reads as
        near zero for six hours, and at noon as a full day of peak sun. The mean
        preserves the energy, which is the quantity the bus balance is about.
        """
        edges = np.concatenate(([0.0], np.cumsum(grid))) / 3600.0
        out: dict[str, np.ndarray] = {}
        n_hours = len(hourly["pv_available_W"])
        for key, values in hourly.items():
            if key == "horizon_s":
                continue
            block = np.empty(len(grid))
            for k in range(len(grid)):
                lo = int(np.floor(edges[k]))
                hi = max(int(np.ceil(edges[k + 1])), lo + 1)
                block[k] = float(np.mean(values[lo:min(hi, n_hours)]))
            out[key] = block
        return out

    def _forecast_rows(self, t: float, forecast) -> dict[str, np.ndarray]:
        """Hourly exogenous inputs over the horizon, from the forecast.

        The forecast is a `WeatherSeries` covering the whole run, so the horizon
        is truncated at its end rather than extrapolated -- a planner that
        invented weather past the end of its data would quietly report
        confidence it does not have.
        """
        series = forecast if forecast is not None else self.context.forecast
        dense = series.densify(self.dt_plan_s, None)
        offsets = dense["time_s"].to_numpy()
        start = int(np.searchsorted(offsets, t, side="right")) - 1
        start = max(start, 0)
        stop = min(start + self.horizon_hours, len(dense))
        window = dense.iloc[start:stop]

        pv = self.model.pv
        available = np.array(
            [float(pv.available_power(row)) for _, row in window.iterrows()], dtype=float
        )
        return {
            "pv_available_W": available,
            "temp_air": window["temp_air"].to_numpy(dtype=float),
            "relative_humidity": window["relative_humidity"].to_numpy(dtype=float),
            "wind_speed": window["wind_speed"].to_numpy(dtype=float),
            "pressure": window["pressure"].to_numpy(dtype=float),
        }

    # --- NLP construction -------------------------------------------------
    def _build(self, n: int, z0: np.ndarray, weather: dict[str, np.ndarray],
               grid: np.ndarray):
        """Assemble the planning NLP, in non-dimensional variables.

        Every decision variable is divided by a characteristic magnitude and
        every constraint row is scaled to order one. That is not cosmetic: in
        physical units this problem mixes a state of order 0.5 with one of order
        30,000 and a constraint residual in watts, and IPOPT's convergence test
        is a single norm over all of them. Unscaled, it hits the iteration limit
        with a constraint violation of order 1000; scaled, it converges.

        `grid` gives each interval's length, so the horizon can be graded --
        hourly where decisions are implemented, six-hourly out at the end where
        the plan only has to value the terminal inventories.

        Rebuilt on every replan. The alternative -- one parameterised NLP whose
        data is swapped -- needs a fixed horizon length, and the horizon
        genuinely shortens as the run approaches the end of the forecast.
        """
        model = self.model
        limits = model.limits(z0, float(np.sum(grid)))
        lo, hi = limits.lower(), limits.upper()
        # z0 can sit marginally outside a freshly-computed snug bound (the plant
        # integrates with its own clipper); widen rather than pose an infeasible
        # initial-state row.
        lo = np.minimum(lo, z0)
        hi = np.maximum(hi, z0)

        z_scale = model.state_scale()
        u_scale = model.control_scale()
        p_scale = model.power_scale()
        Sz, Su = ca.DM(z_scale), ca.DM(u_scale)

        u_lo, u_hi = self._control_bounds()
        guess_z, guess_u = self._rollout_guess(n, z0, weather, grid)

        # Per-interval control bounds, so the curtailment fraction can be pinned
        # where it means nothing. With no sun, `pv * (1 - gamma)` is zero for
        # every gamma: the variable has no effect on any constraint or on the
        # objective, so it is a flat direction the solver is free to wander
        # along. It also breaks the primal-side check that lambda is zero
        # wherever curtailment is interior -- at night gamma sits at some
        # arbitrary interior value while lambda is correctly high, which looks
        # exactly like a broken price and is not one.
        tiled_lo = np.tile(u_lo, (n, 1))
        tiled_hi = np.tile(u_hi, (n, 1))
        dark = np.asarray(weather["pv_available_W"], dtype=float) < 1.0
        tiled_lo[dark, ui("curtail")] = 0.0
        tiled_hi[dark, ui("curtail")] = 0.0
        guess_u[dark, ui("curtail")] = 0.0

        b = NLPBuilder(f"planner_{n}h")
        zv = b.variable(
            "z", (n + 1) * N_Z,
            lb=np.tile(lo / z_scale, n + 1), ub=np.tile(hi / z_scale, n + 1),
            x0=(guess_z / z_scale).ravel(),
        )
        uv = b.variable(
            "u", n * N_U,
            lb=(tiled_lo / u_scale).ravel(), ub=(tiled_hi / u_scale).ravel(),
            x0=(guess_u / u_scale).ravel(),
        )

        # physical quantities, recovered from the scaled decision variables
        Z = [zv[k * N_Z:(k + 1) * N_Z] * Sz for k in range(n + 1)]
        U = [uv[k * N_U:(k + 1) * N_U] * Su for k in range(n)]

        b.constraint("initial_state", (Z[0] - ca.DM(z0)) / Sz, equals=0.0)

        defects, balances, capacities, commitments, min_loads = [], [], [], [], []
        profit = 0.0
        for k in range(n):
            w = {
                "temp_air": float(weather["temp_air"][k]),
                "relative_humidity": float(weather["relative_humidity"][k]),
                "wind_speed": float(weather["wind_speed"][k]),
                "pressure": float(weather["pressure"][k]),
            }
            rates = model.rates(Z[k], U[k], w)
            dt = float(grid[k])

            defects.append((Z[k + 1] - model.step(Z[k], U[k], w, dt)) / Sz)

            # --- bus balance. Its dual is lambda, the price of electricity.
            #
            # Written `demand - supply`, and the direction is load-bearing.
            # Perturbing this row's right-hand side to c > 0 means demand may
            # exceed supply by c -- that is c watts arriving from nowhere, so
            # dJ/dc is exactly the marginal value of energy on the bus, positive
            # and in EUR per watt. Writing it the other way round (`supply -
            # demand`) gives the negative of that: the cost of being forced to
            # over-generate. Both are legitimate duals of the same constraint and
            # only one of them is a price, so the orientation is fixed here and
            # pinned by `test_lambda_is_positive_and_scales_with_scarcity`.
            pv = float(weather["pv_available_W"][k])
            supply = pv * (1.0 - U[k][ui("curtail")]) + U[k][ui("battery_discharge_W")]
            demand = model.total_load_W(Z[k], U[k], w, rates) + U[k][ui("battery_charge_W")]
            balances.append((demand - supply) / p_scale)

            # --- sorbent capacity shrinks as the loop cycles
            capacities.append(
                (Z[k + 1][zi("n_caco3")]
                 - model.capture_capacity_mol(Z[k + 1][zi("cycle_number")]))
                / model.n_total_mol
            )

            # --- commitment: a setpoint may not exceed its enable
            for key in COMMITTED:
                commitments.append(U[k][ui(_SETPOINT_OF[key])] - U[k][ui(f"{key}_on")])

            # --- the electrolyser's minimum load is a safety limit, not a
            # preference: below it hydrogen crosses into the oxygen stream.
            min_loads.append(
                U[k][ui("electrolyser_load")]
                - model.electrolyser_min_load * U[k][ui("electrolyser_on")]
            )

            profit = profit + model.stage_profit_EUR(Z[k], U[k], w, self.economics, dt, rates)

        # --- reactor light-off ------------------------------------------------
        # `max(e_k - e_{k-1}, 0)` written directly in the objective is a kink
        # sitting exactly where the solver spends most of its time, since the
        # reactor's commitment is unchanged at almost every hour. The standard
        # epigraph form is exact and smooth: a non-negative variable bounded
        # below by the increase, with a positive cost that drives it to that
        # bound. This is the `s_k >= e_k - e_{k-1}` of the written formulation.
        starts = b.variable("sabatier_starts", n - 1, lb=0.0, ub=1.0, x0=0.0)
        b.constraint(
            "start_counter",
            ca.vertcat(*[starts[k - 1] - (U[k][ui("sabatier_on")]
                                          - U[k - 1][ui("sabatier_on")])
                         for k in range(1, n)]),
            lb=0.0, ub=1e3,
        )
        profit = profit - self.sabatier_start_cost_EUR * ca.sum1(starts)

        b.constraint("dynamics", ca.vertcat(*defects), equals=0.0)
        b.constraint(BUS_BALANCE, ca.vertcat(*balances), equals=0.0)
        b.constraint("sorbent_capacity", ca.vertcat(*capacities), ub=0.0, lb=-1e3)
        b.constraint("commitment", ca.vertcat(*commitments), ub=0.0, lb=-1e3)
        b.constraint("min_load", ca.vertcat(*min_loads), lb=0.0, ub=1e3)

        b.maximise(profit + self._terminal_value(Z[n]))
        return b.build()

    def _control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Physical box bounds on one control vector.

        Owned by the model, so the bounds and the scaling that divides by them
        cannot drift apart.
        """
        return self.model.control_bounds()

    # --- initial guess ----------------------------------------------------
    def _rollout_guess(self, n, z0, weather, grid) -> tuple[np.ndarray, np.ndarray]:
        """A dynamically consistent starting point, from a simple forward pass.

        Two reasons this is not optional.

        **The kiln ignition gradient is flat.** Below about 1050 K the derivative
        of the calcination rate with respect to temperature is numerically zero
        -- the Baker equilibrium term is still saturated at its clip. Started
        from a cold kiln, a gradient-based solver has *no local information*
        saying that heating would eventually pay, so it leaves the kiln cold and
        converges to a plant that never calcines. Rolling forward under a rule
        that lights the kiln puts the optimiser in the basin where the gradient
        exists, and it can then decide whether to stay there.

        **Multiple shooting needs consistent states.** Tiling `z0` across the
        horizon makes every dynamics row a defect from the start; a rollout makes
        them all zero, so IPOPT begins from a feasible trajectory and spends its
        iterations on optimality rather than on finding the constraint manifold.

        The rule itself is deliberately crude. It is a starting point, not a
        controller: the optimiser is expected to beat it comprehensively, and
        `test_planner_beats_its_own_initial_guess` checks that it does.
        """
        model = self.model
        u_lo, u_hi = self._control_bounds()
        p_max = float(model.battery.max_power_W)
        limits = model.limits(z0, float(np.sum(grid)))
        lo, hi = limits.lower(), limits.upper()

        pv = weather["pv_available_W"]
        strong = float(np.percentile(pv[pv > 0], 40)) if np.any(pv > 0) else 0.0

        guess_z = np.zeros((n + 1, N_Z))
        guess_u = np.zeros((n, N_U))
        z = np.array(z0, dtype=float)
        guess_z[0] = z

        for k in range(n):
            w = {
                "temp_air": float(weather["temp_air"][k]),
                "relative_humidity": float(weather["relative_humidity"][k]),
                "wind_speed": float(weather["wind_speed"][k]),
                "pressure": float(weather["pressure"][k]),
            }
            u = np.zeros(N_U)
            dt = float(grid[k])
            sunny = pv[k] >= max(strong, 1.0)

            # reactor: lit throughout -- it is the cheapest load and the whole
            # point of the buffers is to keep it fed
            u[ui("sabatier_on")] = 1.0
            u[ui("sabatier_feed")] = 0.7
            # kiln: on whenever the sun is up, which is what lights it at all
            u[ui("calciner_on")] = 1.0 if sunny else 0.0
            u[ui("calciner_heat")] = 1.0 if sunny else 0.0
            u[ui("contactor_on")] = 1.0
            u[ui("contactor_flow")] = 0.35
            u[ui("electrolyser_on")] = 1.0 if sunny else 0.0
            u[ui("electrolyser_load")] = 0.6 if sunny else 0.0

            # top the water tank back up to where it started, so the rollout does
            # not walk it towards its floor and hand the solver a cornered guess
            rates = model.rates(z, u, w)
            net_water = float(rates["electrolysis"]) * M_H2O - (
                2.0 * model.plant["water"].p.condensate_recovery
                * float(rates["methanation"]) * M_H2O)
            u[ui("water_makeup_kg_s")] = max(net_water, 0.0)

            # close the balance with the battery, then curtail the remainder
            load = float(model.total_load_W(z, u, w))
            net = float(pv[k]) - load
            if net >= 0.0:
                charge = min(net, p_max)
                u[ui("battery_charge_W")] = charge
                if pv[k] > 1.0:
                    u[ui("curtail")] = min(max((net - charge) / pv[k], 0.0), 1.0)
            else:
                u[ui("battery_discharge_W")] = min(-net, p_max)

            u = np.clip(u, u_lo, u_hi)
            z_next = np.asarray(model.step(z, u, w, dt), dtype=float).ravel()

            # Keep the rollout inside the box so the guess never starts outside
            # its own bounds. The overflow must be read from the *unclipped*
            # trial step: clipping first would silently destroy the surplus and
            # the vent -- which exists precisely to remove it -- would size to
            # zero.
            over_h2 = max(z_next[zi("n_h2")] - hi[zi("n_h2")], 0.0)
            over_co2 = max(z_next[zi("n_co2")] - hi[zi("n_co2")], 0.0)
            if over_h2 > 0.0 or over_co2 > 0.0:
                u[ui("h2_vent_mol_s")] = min(over_h2 / dt, u_hi[ui("h2_vent_mol_s")])
                u[ui("co2_vent_mol_s")] = min(over_co2 / dt, u_hi[ui("co2_vent_mol_s")])
                z_next = np.asarray(model.step(z, u, w, dt), dtype=float).ravel()

            z = np.clip(z_next, lo, hi)
            guess_u[k] = u
            guess_z[k + 1] = z

        return guess_z, guess_u

    def _terminal_value(self, z_end):
        """Value of what is left in the buffers at the end of the horizon.

        Without this the planner empties every buffer on the last day, which is
        optimal for the horizon and wrong for the plant. Inventories are priced
        at the methane they could become, discounted by
        `terminal_value_fraction` so that banking is never more attractive than
        producing. Battery charge is priced through the same chain, at the
        electrolyser's specific energy.

        The tex proposes taking these prices from the previous solve's duals,
        which is better and is a natural upgrade at M6; a fixed fraction is used
        here because a first implementation should not depend on its own output.
        """
        model = self.model
        price = self.economics.p.methane_price_per_kg * M_CH4  # EUR per mol CH4
        f = self.terminal_value_fraction

        # stoichiometry: 4 H2 -> 1 CH4, 1 CO2 -> 1 CH4, 1 CaCO3 -> 1 CO2 -> 1 CH4
        value = (
            f * price * z_end[zi("n_h2")] / 4.0
            + f * price * z_end[zi("n_co2")]
            + f * price * z_end[zi("n_caco3")]
        )
        # battery: kWh -> kg H2 -> mol CH4, at the electrolyser's 56.4 kWh/kg
        kwh = z_end[zi("soc")] * model.battery.nominal_energy_J / 3.6e6
        value = value + f * price * kwh / 56.4 / (2.016e-3) / 4.0
        return value

    def _shifted_warm_start(self, t: float):
        """The previous plan, shifted forward so it lines up with the new `t0`.

        Returns `(grid, states, controls)` in physical units, ready to be
        re-gridded by `_resample`.

        This used to return the previous solution *unshifted*, which was worse
        than useless: the plan is a time series, the window has moved on by the
        replan interval, and handing IPOPT a trajectory misaligned by three hours
        starts it with the night schedule sitting where the afternoon belongs. It
        also silently overrode the rollout guess, so every solve after the first
        began from something worse than a cold start would have given.

        The shift is a re-origin, not an index offset: the previous plan is
        resampled from `t` onwards onto its own grid shape. Because the grading
        is graded rather than uniform, interval `k` of the new plan is not
        interval `k+1` of the old one, and only a time-based lookup gets that
        right.
        """
        plan = self.plan
        if plan is None:
            return None

        grid = plan.dt_s
        edges = np.concatenate(([0.0], np.cumsum(grid)))
        mid = t + 0.5 * (edges[:-1] + edges[1:])

        controls = np.stack([plan.control_at(m) for m in mid])
        states = np.vstack([
            plan.states[plan.index_at(t)],
            np.stack([plan.states[min(plan.index_at(m) + 1, len(plan.states) - 1)]
                      for m in mid]),
        ])
        # How far this warm start is *actually* informative. The old plan ran to
        # `t0 + horizon`, and the new one starts at `t`, so it covers only
        # `horizon - (t - t0)`. Beyond that `control_at` clamps to the final
        # interval and repeats it. Reporting the old grid's full length here --
        # which is what it did first -- told `_splice` the tail was good data and
        # left the last replan interval as a frozen extrapolation.
        covered = max(plan.horizon_s - (t - plan.t0_s), 0.0)
        return grid, states, controls, covered

    def _pack(self, grid: np.ndarray, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Build a scaled decision vector from a physical trajectory."""
        return np.concatenate([
            (states / self.model.state_scale()).ravel(),
            (controls / self.model.control_scale()).ravel(),
            np.zeros(max(len(grid) - 1, 0)),   # start counters
        ])

    def _unpack(self, x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of `_pack`: physical states and controls from a solution."""
        n_z = (n + 1) * N_Z
        states = x[:n_z].reshape(n + 1, N_Z) * self.model.state_scale()
        controls = x[n_z:n_z + n * N_U].reshape(n, N_U) * self.model.control_scale()
        return states, controls

    def _splice(self, x0: np.ndarray, nlp, grid: np.ndarray,
                src_horizon_s: float) -> np.ndarray:
        """Use the warm start only where it has data; the rollout beyond.

        A homotopy stage extends the horizon, so the previous solution covers
        only the front of the new grid. `_resample` fills the tail by holding the
        source's last state and control constant, which is a trajectory where
        nothing changes while the dynamics insist that it must -- every interval
        out there starts as a large defect, and IPOPT went into restoration and
        declared the problem locally infeasible.

        `nlp.x0` already holds a dynamically consistent rollout over the *whole*
        new horizon, so the tail is taken from there. The result is the previous
        solution where it is informative and the rollout where it is not.
        """
        edges = np.concatenate(([0.0], np.cumsum(grid)))
        n = len(grid)
        out = np.array(x0, dtype=float, copy=True)
        for k in range(n):
            if edges[k] < src_horizon_s - 1e-6:
                continue
            u_lo = (n + 1) * N_Z + k * N_U
            out[u_lo:u_lo + N_U] = nlp.x0[u_lo:u_lo + N_U]
            z_lo = (k + 1) * N_Z
            out[z_lo:z_lo + N_Z] = nlp.x0[z_lo:z_lo + N_Z]
        return out

    @staticmethod
    def _resample(src_grid, states, controls, dst_grid):
        """Move a trajectory from one grid onto another by nearest-interval lookup.

        Homotopy stages and successive replans have different interval counts and
        different spacings, so a warm start has to be *re-gridded*, not resized.
        An earlier version truncated or padded the raw decision vector, which
        looks harmless and is not: the vector is `[z | u | starts]`, so changing
        the interval count moves the block boundaries and a truncation splices
        the tail of the state block into the head of the control block. The
        result is a warm start that is not a trajectory at all, and IPOPT spends
        its whole budget recovering from it.
        """
        src_edges = np.concatenate(([0.0], np.cumsum(src_grid)))
        dst_edges = np.concatenate(([0.0], np.cumsum(dst_grid)))
        mid = 0.5 * (dst_edges[:-1] + dst_edges[1:])

        idx = np.clip(np.searchsorted(src_edges, mid, side="right") - 1,
                      0, len(controls) - 1)
        new_u = controls[idx]
        new_z = np.vstack([states[0], states[np.clip(idx + 1, 0, len(states) - 1)]])
        return new_z, new_u

    # --- reporting --------------------------------------------------------
    def diagnostics(self) -> dict[str, float]:
        out = dict(self._diagnostics)
        if self.plan is not None:
            out["plan_lambda_EUR_per_kWh"] = self.plan.price_at(self._last_t)
        return out

    #: absolute time of the most recent `act`, so `diagnostics` reports the
    #: price actually in force rather than the price at the start of the plan
    _last_t: float = 0.0


#: setpoint variable belonging to each committed subsystem
_SETPOINT_OF: dict[str, str] = {
    "contactor": "contactor_flow",
    "calciner": "calciner_heat",
    "electrolyser": "electrolyser_load",
    "sabatier": "sabatier_feed",
}
