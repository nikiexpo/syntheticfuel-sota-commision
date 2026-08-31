"""The economic planner (layer 2) -- the outer problem of the hierarchy.

Decides, over a ten-day horizon at hourly resolution, how much energy goes into
each chain and how much inventory to bank, re-solving every three hours on a
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
Multiple shooting on the reduced six-state model: the state at every hour is a
decision variable and the dynamics are equality constraints. That is more
variables than single shooting but far better conditioned over 240 steps, and it
makes every buffer bound a simple box rather than a deeply nested expression.

    variables    z_0..z_N  (6 each)      states at each hour boundary
                 u_0..u_N-1 (11 each)    setpoints, enables, battery, curtailment
    constraints  dynamics                z_{k+1} = RK4(z_k, u_k)
                 bus_balance             supply == demand          <- lambda
                 sorbent_capacity        n_CaCO3 <= n_tot * X(N)
                 commitment              setpoint <= enable
                 min_load                electrolyser >= i_min * enable
                 initial_state           z_0 == current estimate

Curtailment is a free variable rather than a residue here, which is the one place
the planner's world differs structurally from the bus's. It has to be: the
planner must be *able* to choose to spill, since deciding what not to absorb is a
real scheduling decision. The bus then computes curtailment as a residue when the
plan meets reality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import casadi as ca
import numpy as np

from sfp.control.base import ControlContext, Controller
from sfp.control.planner_model import COMMITTED, N_U, N_Z, PlannerModel, ui, zi
from sfp.economics import Economics
from sfp.sim.bus import Request
from sfp.solvers import NLPBuilder, SolverBackend, get_backend
from sfp.units import M_CH4

#: Names the planner uses for the blocks it later reads back by name.
BUS_BALANCE = "bus_balance"


@dataclass
class Plan:
    """A solved schedule, sampled on the planner's hourly grid."""

    t0_s: float
    dt_s: float
    states: np.ndarray           # (N+1, 6)
    controls: np.ndarray         # (N, 11)
    lambda_EUR_per_kWh: np.ndarray   # (N,)
    objective_EUR: float
    solve_stats: Any = None
    horizon_s: float = 0.0

    def index_at(self, t_s: float) -> int:
        """Index of the planning interval containing absolute time `t_s`."""
        k = int(np.floor((t_s - self.t0_s) / self.dt_s))
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
        "Economic MPC: 10-day horizon at 1 h, re-solved every 3 h, on a six-state "
        "reduced model. Publishes the shadow price of electricity."
    )

    def __init__(
        self,
        *,
        horizon_hours: int = 240,
        replan_interval_s: float = 3 * 3600.0,
        terminal_value_fraction: float = 0.5,
        sabatier_start_cost_EUR: float = 2.0,
        backend: SolverBackend | str = "ipopt",
        max_iter: int = 3000,
        tol: float = 1e-5,
        commit_threshold: float = 0.5,
        name: str | None = None,
    ) -> None:
        self.horizon_hours = int(horizon_hours)
        self.replan_interval_s = float(replan_interval_s)
        self.terminal_value_fraction = float(terminal_value_fraction)
        self.sabatier_start_cost_EUR = float(sabatier_start_cost_EUR)
        self.commit_threshold = float(commit_threshold)
        self.max_iter = int(max_iter)
        # A planning tolerance, not a physics tolerance. IPOPT's default 1e-6
        # spends hundreds of iterations polishing a schedule whose inputs are a
        # weather forecast; the model error dwarfs the solver error by orders of
        # magnitude long before that.
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
        weather = self._forecast_rows(t, forecast)
        n = len(weather["pv_available_W"])
        if n == 0:
            return

        nlp = self._build(n, z0, weather)
        solution = self.backend.solve(nlp, x0=self._warm_start(nlp, n))
        self._solves += 1

        if not solution.success:
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

        # Two conversions, both easy to get wrong and both pinned by tests.
        #
        # 1. `lam` on a problem built with `maximise` is already the marginal
        #    value of relaxing the constraint -- no sign flip (see Solution.dual).
        # 2. The balance row was divided by `power_scale`, so its dual is EUR per
        #    scaled unit. Dividing by the same factor returns EUR per watt; one
        #    extra watt held for the whole interval is dt/3.6e6 kWh.
        lam_per_W = solution.dual(BUS_BALANCE) / model.power_scale()
        lam_kWh = lam_per_W * 3.6e6 / self.dt_plan_s

        self.plan = Plan(
            t0_s=t,
            dt_s=self.dt_plan_s,
            states=states,
            controls=controls,
            lambda_EUR_per_kWh=lam_kWh,
            objective_EUR=-solution.f,
            solve_stats=solution.stats,
            horizon_s=n * self.dt_plan_s,
        )
        self._last_solution_x = solution.x
        self._diagnostics.update(
            plan_objective_EUR=-solution.f,
            plan_solve_time_s=time.perf_counter() - started,
            plan_iterations=float(solution.stats.iterations),
            plan_kkt_residual=solution.stats.kkt_residual,
            plan_solve_failed=0.0,
            plan_failures=float(self._failures),
            plan_lambda_mean=float(np.mean(lam_kWh)),
            plan_lambda_max=float(np.max(lam_kWh)),
        )

    # --- horizon data -----------------------------------------------------
    dt_plan_s: float = 3600.0

    def _forecast_rows(self, t: float, forecast) -> dict[str, np.ndarray]:
        """Hourly exogenous inputs over the horizon, from the forecast.

        The forecast is a `WeatherSeries` covering the whole run, so the horizon
        is truncated at its end rather than extrapolated -- a planner that
        invented weather past the end of its data would quietly report
        confidence it does not have.
        """
        series = forecast if forecast is not None else self.context.forecast
        horizon_s = self.horizon_hours * self.dt_plan_s
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
            "horizon_s": np.array([horizon_s]),
        }

    # --- NLP construction -------------------------------------------------
    def _build(self, n: int, z0: np.ndarray, weather: dict[str, np.ndarray]):
        """Assemble the planning NLP, in non-dimensional variables.

        Every decision variable is divided by a characteristic magnitude and
        every constraint row is scaled to order one. That is not cosmetic: in
        physical units this problem mixes a state of order 0.5 with one of order
        30,000 and a constraint residual in watts, and IPOPT's convergence test
        is a single norm over all of them. Unscaled, it hits the iteration limit
        with a constraint violation of order 1000; scaled, it converges.

        Rebuilt on every replan. The alternative -- one parameterised NLP whose
        data is swapped -- needs a fixed horizon length, and the horizon
        genuinely shortens as the run approaches the end of the forecast.
        """
        model = self.model
        limits = model.limits()
        lo, hi = limits.lower(), limits.upper()
        dt = self.dt_plan_s

        z_scale = model.state_scale()
        u_scale = model.control_scale()
        p_scale = model.power_scale()
        Sz, Su = ca.DM(z_scale), ca.DM(u_scale)

        u_lo, u_hi = self._control_bounds()
        guess_z, guess_u = self._rollout_guess(n, z0, weather)

        b = NLPBuilder(f"planner_{n}h")
        zv = b.variable(
            "z", (n + 1) * N_Z,
            lb=np.tile(lo / z_scale, n + 1), ub=np.tile(hi / z_scale, n + 1),
            x0=(guess_z / z_scale).ravel(),
        )
        uv = b.variable(
            "u", n * N_U,
            lb=np.tile(u_lo / u_scale, n), ub=np.tile(u_hi / u_scale, n),
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
        """Physical box bounds on one control vector."""
        model = self.model
        lo = np.zeros(N_U)
        hi = np.ones(N_U)
        p_max = float(model.battery.max_power_W)
        hi[ui("battery_charge_W")] = p_max
        hi[ui("battery_discharge_W")] = p_max
        # generous vent caps: they exist for feasibility, not as a real duty
        hi[ui("h2_vent_mol_s")] = 5.0
        hi[ui("co2_vent_mol_s")] = 5.0
        return lo, hi

    # --- initial guess ----------------------------------------------------
    def _rollout_guess(self, n, z0, weather) -> tuple[np.ndarray, np.ndarray]:
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
        dt = self.dt_plan_s
        u_lo, u_hi = self._control_bounds()
        p_max = float(model.battery.max_power_W)

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

            # keep the rollout inside the box so the guess never starts outside
            # its own bounds; the vents are what make that physically meaningful
            lo, hi = model.limits().lower(), model.limits().upper()
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

    def _warm_start(self, nlp, n: int) -> np.ndarray | None:
        """Shift the previous solution forward by one replan interval."""
        if self._last_solution_x is None or self._last_solution_x.size != nlp.n_x:
            return None
        return self._last_solution_x

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
