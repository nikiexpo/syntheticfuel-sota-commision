"""The inner NMPC (layer 3) -- the local problem, solved at the planner's price.

This layer does **not** track setpoints. It solves a small economic problem over
a short horizon on the full sixteen-state model:

    max  sum_l dt [ p_CH4 * r_sab * M_CH4  -  lambda(t_l) * P_process ]
         + sum_i pi_i * (x_N,i - x_i^target)

subject to the real nonlinear dynamics, the DC bus balance, the hard temperature
limits and rate limits on every input.

Why price coordination rather than setpoint tracking
----------------------------------------------------
The plan is *always* wrong, because the forecast is always wrong. If the planner
says "kiln at 340 kW at 14:00" and a cloud bank arrives, tracking that setpoint is
actively harmful: the NMPC would drain the battery to hit a number computed under
conditions that no longer exist. Telling it instead that "electricity is worth
0.054 EUR/kWh right now, and you need this much CaCO3 banked by dawn" lets it
respond sensibly to a situation the planner never saw.

Three consequences, and the first is why the architecture is shaped this way:

1. **The multiscale problem dissolves.** The inner layer never needs the
   planner's time grid, its horizon, or its reduced state vector. It needs one
   number per instant and a handful of terminal prices.
2. **Graceful degradation.** A stale plan still yields a sensible price; a stale
   setpoint trajectory does not.
3. It is the Lagrangian decomposition of the full problem rather than a
   heuristic hand-off, so the two layers are solving one problem between them.

The model is the plant's own
----------------------------
`Plant.rhs` and `Plant.step(clip=False)` are evaluated symbolically here -- the
same coupled two-phase evaluation the simulator integrates, with the same
parameters. There is no second copy of the dynamics to drift out of step. (The
*truth* simulator's parameters are perturbed away from these at M5; that mismatch
is the point, and it arrives through the plant this controller is handed, not
through a re-typed model.)

Fast, because it has to be
--------------------------
This solves once per control interval, which over a ten-day run at five-minute
cadence is 2880 solves. A horizon of 1 h at 5 min is 12 intervals -- about 320
variables -- and it is warm-started from the previous solve shifted by one step,
which is the standard MPC trick and is worth an order of magnitude here. The
horizon is short deliberately: the planner already carries the long view, and
that division of labour is the entire reason for having two layers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import casadi as ca
import numpy as np

from sfp.sim.bus import CONTROLLABLE
from sfp.sim.plant import WEATHER_KEYS
from sfp.solvers import NLPBuilder, SolverBackend, get_backend
from sfp.units import M_CH4

#: Constraint block whose dual is the local price of bus power.
BUS_BALANCE = "bus_balance"

#: Subsystems whose setpoint the NMPC commands, and the index of that setpoint
#: within the subsystem's input vector. Every one of these is (setpoint, enable).
SETPOINT_INDEX = 0
ENABLE_INDEX = 1


@dataclass
class NMPCSolution:
    """One solved local problem."""

    setpoints: dict[str, float]
    enables: dict[str, float]
    battery_charge_W: float
    battery_discharge_W: float
    curtail_fraction: float
    objective_EUR: float
    price_EUR_per_kWh: float
    stats: Any = None
    states: np.ndarray | None = field(default=None, repr=False)
    controls: np.ndarray | None = field(default=None, repr=False)


class InnerNMPC:
    """Short-horizon economic NMPC on the full plant model.

    Not a `Controller`: it has no opinion about when to re-plan and no access to
    a forecast beyond what it is handed. `HierarchicalController` owns both this
    and the planner and wires the price between them.
    """

    def __init__(
        self,
        plant,
        economics,
        *,
        horizon_s: float = 3600.0,
        dt_s: float = 300.0,
        backend: SolverBackend | str = "ipopt",
        max_iter: int = 300,
        tol: float = 1e-4,
        rate_limit: float = 0.34,
        terminal_weight: float = 1.0,
    ) -> None:
        self.plant = plant
        self.economics = economics
        self.horizon_s = float(horizon_s)
        self.dt_s = float(dt_s)
        self.n_steps = max(1, int(round(self.horizon_s / self.dt_s)))
        self.rate_limit = float(rate_limit)
        self.terminal_weight = float(terminal_weight)
        self.backend = (
            backend if isinstance(backend, SolverBackend)
            else get_backend(backend, max_iter=max_iter, tol=tol,
                             acceptable_tol=tol * 100.0)
        )

        self._input_keys = [key for key, sub in plant if sub.n_inputs]
        self._input_sizes = {key: plant[key].n_inputs for key in self._input_keys}
        self._n_u = sum(self._input_sizes.values())
        self._warm: np.ndarray | None = None
        self._nlp = None

        self.n_x = plant.n_states
        self._x_lo, self._x_hi = self._state_bounds()
        self._u_lo, self._u_hi = self._input_bounds()
        self._x_scale = self._state_scale()
        self._u_scale = np.maximum(np.abs(self._u_hi), 1.0)
        self._p_scale = max(float(plant["pv"].rated_ac_W), 1e5)
        self._weather_keys = tuple(sorted(WEATHER_KEYS))
        self._battery = plant["battery"]
        self._step_fn = self._build_step_function()

    # --- the one-step map, built once -------------------------------------
    def _build_step_function(self) -> ca.Function:
        """Compile `(x, u, w) -> (x_next, bus_total, process_power, r_CH4)`.

        Built once and *called* at each interval, rather than letting
        `Plant.step` inline itself into the NLP twelve times over. The
        difference is not marginal: one RK4 step is four full two-phase
        evaluations of a nine-subsystem coupled plant, so inlining puts
        forty-eight of them in a single expression graph, and both construction
        and every subsequent derivative evaluation pay for all of it. As a
        `ca.Function` it is one node that CasADi differentiates once.

        Weather enters as a vector rather than a dict so that it can be an
        argument; the keys are the plant's own `WEATHER_KEYS`, in a fixed order.
        """
        x = ca.MX.sym("x", self.n_x)
        u = ca.MX.sym("u", self._n_u)
        wv = ca.MX.sym("w", len(self._weather_keys))
        w = {key: wv[i] for i, key in enumerate(self._weather_keys)}
        u_dict = self._split_u(u)

        x_next = self.plant.step(0.0, x, u_dict, w, self.dt_s, clip=False)
        outputs, _ = self.plant.evaluate(0.0, x, u_dict, w)

        total = 0.0
        for key in self.plant.subsystems:
            channel = f"power.{key}"
            if channel in outputs:
                total = total + outputs[channel]
        process = 0.0
        for key in CONTROLLABLE:
            channel = f"power.{key}"
            if channel in outputs:
                process = process + outputs[channel]
        r_sab = outputs.get("r_sabatier_co2_mol_s", 0.0)

        return ca.Function(
            "nmpc_step", [x, u, wv], [x_next, total, process, r_sab],
            ["x", "u", "w"], ["x_next", "bus_total", "process_W", "r_ch4"],
        )

    def _weather_vector(self, w: Mapping[str, Any]) -> np.ndarray:
        return np.array([float(w.get(key, 0.0)) for key in self._weather_keys],
                        dtype=float)

    def _state_scale(self) -> np.ndarray:
        """Characteristic magnitude of each of the sixteen states.

        The NMPC is solved non-dimensionally for the same reason the planner is:
        the raw state vector runs from 0.5 (state of charge) through 1173
        (kelvin) to 42,000 (moles of CaO), and IPOPT's convergence test is one
        norm over all of them. Unscaled, this problem hit its iteration limit
        on every solve.

        The scale is taken from whichever of the state's own upper bound or its
        initial value is larger, floored at one. That is crude next to a
        hand-tuned vector, but it is derived from the plant rather than typed in,
        so it cannot go stale when a subsystem is resized.
        """
        x0 = np.abs(np.asarray(self.plant.initial_state(), dtype=float))
        bound = np.where(np.isfinite(self._x_hi), np.abs(self._x_hi), 0.0)
        return np.maximum(np.maximum(bound, x0), 1.0)

    # --- bounds -----------------------------------------------------------
    def _state_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = [], []
        for _key, sub in self.plant:
            if not sub.n_states:
                continue
            a, b = sub.state_bounds()
            lo.append(np.asarray(a, dtype=float))
            hi.append(np.asarray(b, dtype=float))
        if not lo:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(lo), np.concatenate(hi)

    def _input_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = [], []
        for key in self._input_keys:
            a, b = self.plant[key].input_bounds()
            lo.append(np.asarray(a, dtype=float))
            hi.append(np.asarray(b, dtype=float))
        return np.concatenate(lo), np.concatenate(hi)

    def _split_u(self, u_flat):
        """Map one flat control vector onto the plant's per-subsystem dict."""
        out, offset = {}, 0
        for key, sub in self.plant:
            n = sub.n_inputs
            if n:
                out[key] = u_flat[offset:offset + n]
                offset += n
            else:
                out[key] = np.zeros(0)
        return out

    def _guess_u(self) -> np.ndarray:
        """A mid-range starting control vector.

        All-zeros means every subsystem off and no curtailment, which violates
        the bus balance by the full array output at midday and sits the
        commitment variables exactly on a bound. Starting at half-open with
        everything enabled is closer to any plausible answer and, more usefully,
        is interior -- the barrier does not have to push off a face before it can
        make progress.
        """
        u = np.zeros(self._n_u)
        for key in self._input_keys:
            if key in ("pv", "battery"):
                continue
            u[self._u_index(key, SETPOINT_INDEX)] = 0.5
            if self._input_sizes[key] > 1:
                u[self._u_index(key, ENABLE_INDEX)] = 1.0
        return u

    def _bind_enables(self, enables: Mapping[str, float] | None):
        """Pin each subsystem's enable to the plan's commitment.

        Returns control bounds with the enable entries collapsed to a point, and
        a matching initial guess. A subsystem the plan has shut down also has its
        setpoint pinned to zero -- otherwise the setpoint is a free variable that
        nothing in the objective depends on, which is a flat manifold for the
        solver to wander over.
        """
        lo = np.array(self._u_lo, dtype=float, copy=True)
        hi = np.array(self._u_hi, dtype=float, copy=True)
        guess = self._guess_u()
        if not enables:
            return lo, hi, guess

        for key in self._input_keys:
            if key in ("pv", "battery") or self._input_sizes[key] < 2:
                continue
            e = 1.0 if float(enables.get(key, 1.0)) >= 0.5 else 0.0
            i_e = self._u_index(key, ENABLE_INDEX)
            lo[i_e] = hi[i_e] = guess[i_e] = e
            if e == 0.0:
                i_s = self._u_index(key, SETPOINT_INDEX)
                lo[i_s] = hi[i_s] = guess[i_s] = 0.0
        return lo, hi, guess

    def _u_index(self, key: str, within: int) -> int:
        offset = 0
        for k in self._input_keys:
            if k == key:
                return offset + within
            offset += self._input_sizes[k]
        raise KeyError(key)

    # --- the problem ------------------------------------------------------
    def solve(
        self,
        t: float,
        x0: np.ndarray,
        prices: np.ndarray,
        weather: list[Mapping[str, Any]],
        *,
        targets: Mapping[str, float] | None = None,
        terminal_prices: Mapping[str, float] | None = None,
        last_u: np.ndarray | None = None,
        enables: Mapping[str, float] | None = None,
    ) -> NMPCSolution:
        """Solve the local problem over `n_steps` intervals from `t`.

        `prices` is lambda in EUR/kWh at each interval; `weather` one row per
        interval. `targets` and `terminal_prices` are keyed by full state name
        (``"gas.n_h2"``) and value the terminal state -- they are what carries
        the planner's long view into a one-hour problem.
        """
        if not targets or not terminal_prices:
            raise ValueError(
                "InnerNMPC needs terminal inventory prices. Since lambda prices "
                "only the battery, nothing in a one-hour horizon rewards making "
                "hydrogen or carbonate -- the methane they become is hours away "
                "-- so without a terminal value the process setpoints sit on a "
                "flat manifold, the solve does not converge, and any answer it "
                "does return is arbitrary. The planner supplies these."
            )

        n = self.n_steps
        dt = self.dt_s
        plant = self.plant
        names = plant.state_names()

        xs, us = self._x_scale, self._u_scale
        Sx, Su = ca.DM(xs), ca.DM(us)
        x0 = np.asarray(x0, dtype=float)
        # the plant's clipper can leave a state a hair outside its own box
        lo = np.minimum(self._x_lo, x0)
        hi = np.maximum(self._x_hi, x0)

        b = NLPBuilder("nmpc")
        xv = b.variable(
            "x", (n + 1) * self.n_x,
            lb=np.tile(lo / xs, n + 1), ub=np.tile(hi / xs, n + 1),
            x0=np.tile(x0 / xs, n + 1),
        )
        # Commitment is fixed, not optimised. The plant gates every subsystem
        # with `smooth_step(e - 0.5, width=0.05)`, which is a near-discontinuity:
        # as a free variable the enable has essentially zero gradient anywhere
        # except within +/-0.1 of the switching point, so the solver has nothing
        # to move it with and burns its whole budget. More to the point, this is
        # not the NMPC's decision -- the commitment schedule is one of the three
        # things the planner publishes, and the inner layer's job is continuous
        # modulation at the price it is given. Pinning the enables here removes
        # four near-discontinuous gates per interval *and* puts the decision in
        # the layer that has the horizon to make it.
        u_lo, u_hi, u_guess = self._bind_enables(enables)
        uv = b.variable(
            "u", n * self._n_u,
            lb=np.tile(u_lo / us, n), ub=np.tile(u_hi / us, n),
            x0=np.tile(u_guess / us, n),
        )

        # physical quantities, recovered from the scaled decision variables
        X = [xv[k * self.n_x:(k + 1) * self.n_x] * Sx for k in range(n + 1)]
        U = [uv[k * self._n_u:(k + 1) * self._n_u] * Su for k in range(n)]

        b.constraint("initial_state", (X[0] - ca.DM(x0)) / Sx, equals=0.0)

        defects, balances, rates = [], [], []
        profit = 0.0
        for k in range(n):
            w = self._weather_vector(weather[min(k, len(weather) - 1)])
            step = self._step_fn(x=X[k], u=U[k], w=ca.DM(w))

            # the plant's own coupled dynamics, with no clipping
            defects.append((X[k + 1] - step["x_next"]) / Sx)

            # --- bus balance: the plant's sign convention makes this a plain sum
            balances.append(step["bus_total"] / self._p_scale)

            # --- the local economic objective
            r_sab = step["r_ch4"]
            price = float(prices[min(k, len(prices) - 1)])

            # Battery wear is not optional here. Without it nothing in the
            # objective depends on battery throughput, so charging and
            # discharging at once is free, and the solver duly did exactly that
            # (635 kW in, 500 kW out, 58 kW of pure round-trip loss). The wear
            # term is what makes simultaneous charge/discharge strictly worse
            # than either alone, which is how this formulation avoids needing an
            # explicit complementarity constraint between them.
            i_c = self._u_index("battery", 0)
            charge, discharge = U[k][i_c], U[k][i_c + 1]
            efc = (charge + discharge) * dt / (2.0 * self._battery.nominal_energy_J)

            # No price on energy in the stage cost at all. Every intertemporal
            # value lives in the terminal term, which is the only place it can
            # be stated once.
            #
            # Two formulations were tried and both were wrong, in instructive
            # ways. Pricing total *process* power -- the written formulation --
            # double-counts against the enforced bus balance: substitute the
            # balance and it becomes a charge of lambda on every watt of PV
            # delivered, including watts that would otherwise be spilled, which
            # is a standing bias toward curtailment. Moving lambda onto the
            # battery instead fixes that but breaks something subtler: lambda is
            # the *marginal* value of energy, and it is zero at midday precisely
            # because the plant is already saturated and spilling. Pricing
            # storage at the current lambda therefore says charging is worthless
            # at noon -- exactly when the battery should be filling for the
            # evening. Storage is worth the price at the hour it will be *used*,
            # which a one-hour horizon cannot see and the terminal value can.
            profit = profit + dt * (
                self.economics.p.methane_price_per_kg * r_sab * M_CH4
            ) - efc * self._battery.cost_per_efc_EUR()

            # --- rate limits. Real actuators move at a finite speed, and an
            # unconstrained NMPC will happily chatter a kiln between 0 and 1 on
            # a five-minute grid because the model lets it.
            prev = ca.DM(np.asarray(last_u, dtype=float)) if k == 0 and last_u is not None \
                else (U[k - 1] if k > 0 else None)
            if prev is not None:
                rates.append((U[k] - prev) / Su)

        b.constraint("dynamics", ca.vertcat(*defects), equals=0.0)
        b.constraint(BUS_BALANCE, ca.vertcat(*balances), equals=0.0)
        if rates:
            b.constraint("rate_limit", ca.vertcat(*rates),
                         lb=-self.rate_limit, ub=self.rate_limit)

        b.maximise(profit + self._terminal_value(X[n], names, targets,
                                                 terminal_prices))
        nlp = b.build()

        warm = self._warm if (self._warm is not None
                              and self._warm.size == nlp.n_x) else None
        started = time.perf_counter()
        solution = self.backend.solve(nlp, x0=warm)
        elapsed = time.perf_counter() - started

        if not solution.success:
            return NMPCSolution(
                setpoints={}, enables={}, battery_charge_W=0.0,
                battery_discharge_W=0.0, curtail_fraction=0.0,
                objective_EUR=float("nan"), price_EUR_per_kWh=float(prices[0]),
                stats=solution.stats,
            )

        self._warm = solution.x
        # the decision vector is non-dimensional; restore physical units
        u_first = solution.value("u")[:self._n_u] * us
        states = solution.value("x").reshape(n + 1, self.n_x) * xs

        setpoints, enables = {}, {}
        for key in CONTROLLABLE:
            if key not in self._input_sizes:
                continue
            s = float(u_first[self._u_index(key, SETPOINT_INDEX)])
            e = float(u_first[self._u_index(key, ENABLE_INDEX)])
            enables[key] = 1.0 if e >= 0.5 else 0.0
            setpoints[key] = float(np.clip(s, 0.0, 1.0))

        battery = u_first[self._u_index("battery", 0):self._u_index("battery", 0) + 2]
        return NMPCSolution(
            setpoints=setpoints,
            enables=enables,
            battery_charge_W=float(max(battery[0], 0.0)),
            battery_discharge_W=float(max(battery[1], 0.0)),
            curtail_fraction=float(np.clip(u_first[self._u_index("pv", 0)], 0.0, 1.0)),
            objective_EUR=-solution.f,
            price_EUR_per_kWh=float(prices[0]),
            stats=solution.stats,
            states=states,
            controls=solution.value("u").reshape(n, self._n_u) * us,
        )

    def _terminal_value(self, x_end, names, targets, terminal_prices):
        """Penalise leaving the inventories away from where the plan wants them.

        Without a terminal term the NMPC empties every buffer inside its own
        hour, which is locally optimal and globally wrong -- exactly the failure
        the hierarchy exists to prevent.

        Quadratic in the deviation, not linear in the level. That is a deliberate
        change from the written formulation, and it was forced by measurement.
        A linear terminal value `pi * (x_N - x_target)` is only correct if `pi`
        is a true constant marginal value, and it is not: the worth of another
        kilowatt-hour in the battery falls as the battery fills, because there is
        less and less chance of using it. Priced linearly at the full
        methane-chain value it comes to roughly 50-80 EUR across the pack against
        stage costs of about 5 EUR, so the objective became "charge as hard as
        possible" by a factor of fifteen, the solution sat in a corner, and the
        NMPC stopped converging at all.

        Three variants were tried and all failed the same way -- pricing process
        power against the enforced balance, pricing the battery at the current
        lambda, and pricing it at a forward lambda. They differ only in which
        large linear coefficient they use. The original converged solely because
        a `lambda * P_process` term happened to push back against the storage
        incentive, which is two errors cancelling rather than a formulation.

        The quadratic form is standard economic-MPC practice: cheap near the
        target, expensive far from it, bounded gradient everywhere, and it
        respects the fact that the planner chose that target for a reason. The
        inner layer stays free to deviate where local conditions justify it,
        which is the whole point of not tracking a setpoint trajectory.
        """
        if not targets or not terminal_prices:
            return 0.0
        penalty = 0.0
        index = {name: i for i, name in enumerate(names)}
        for name, target in targets.items():
            if name not in index or name not in terminal_prices:
                continue
            i = index[name]
            scale = max(float(self._x_scale[i]), 1e-9)
            deviation = x_end[i] - float(target)
            penalty = penalty + (self.terminal_weight
                                 * float(terminal_prices[name])
                                 * deviation * deviation / scale)
        return -penalty
