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
    #: Load shed the inner layer expects over its horizon, kWh. Non-zero means
    #: the dispatch layer has committed more than the bus can carry, which is
    #: worth reporting rather than absorbing silently.
    predicted_shed_kWh: float = 0.0
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
        #: Price charged for predicting a load shed, EUR/kWh. Against a shadow
        #: price of order 0.05 this is a hundredfold penalty, so shedding is a
        #: genuine last resort rather than a cheap way out of a tight hour.
        #: Used by the economic objective only; the tracking one uses
        #: `shed_weight`, which carries no units.
        shed_penalty_EUR_per_kWh: float = 5.0,
        #: `"economic"` maximises operating profit -- the original formulation,
        #: kept because `HierarchicalController` is built around it.
        #:
        #: `"tracking"` is the proposed architecture: the inner layer makes no
        #: economic decision at all. Every euro lives in the dispatch LP, and
        #: this layer only has to realise the strategy it is handed on the real
        #: sixteen-state dynamics, at five-minute resolution, without violating
        #: anything. See `_tracking_objective`.
        objective: str = "economic",
        #: Tracking mode weights, all dimensionless. Setpoint deviation is
        #: O(1) because controls are scaled to their own bounds; inventory
        #: deviation is normalised by the state scale, so a 10 % drift on a
        #: buffer costs `0.01 * terminal_weight` against a full setpoint
        #: deviation's `setpoint_weight`. The defaults make a 10 % inventory
        #: drift equal to one saturated setpoint.
        setpoint_weight: float = 1.0,
        #: Tracking mode only, so the economic path that `HierarchicalController`
        #: uses keeps its own `terminal_weight` untouched. 100 is what the
        #: comment above actually implies: inventory deviation is normalised
        #: by the state scale, so a 10 % drift contributes 0.01 and needs a
        #: weight of 100 to match one saturated setpoint. Left at 1.0 it is a
        #: hundredfold too weak, and the layer will drain a buffer to its floor
        #: rather than give up a setpoint -- which is what it did.
        tracking_terminal_weight: float = 100.0,
        #: Weight on a soft band excursion. Between tracking and shed by two
        #: orders each way: a ramp that overshoots the plan's ceiling for an
        #: interval must dominate ordinary tracking error, and must never
        #: compete with keeping the bus supplied.
        band_weight: float = 1.0e2,
        shed_weight: float = 1.0e4,
    ) -> None:
        self.plant = plant
        self.economics = economics
        self.horizon_s = float(horizon_s)
        self.dt_s = float(dt_s)
        self.n_steps = max(1, int(round(self.horizon_s / self.dt_s)))
        self.rate_limit = float(rate_limit)
        self.terminal_weight = float(terminal_weight)
        self.shed_penalty_EUR_per_kWh = float(shed_penalty_EUR_per_kWh)
        if objective not in ("economic", "tracking"):
            raise ValueError(f"objective must be 'economic' or 'tracking', got {objective!r}")
        self.objective = objective
        self.setpoint_weight = float(setpoint_weight)
        self.band_weight = float(band_weight)
        self.tracking_terminal_weight = float(tracking_terminal_weight)
        self.shed_weight = float(shed_weight)
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

    def _bind_bands(self, bands: Mapping[str, Any]):
        """Apply the dispatch layer's bands. Returns `(lo, hi, guess, ceilings)`.

        Commitment is a *bound*, which is the point of banding rather than
        flagging: handed down as a flag it becomes a variable sitting on a
        near-discontinuous gate, handed down as a bound it costs no variable and
        no smoothing. Per subsystem:

            committed      enable pinned to 1, setpoint ceiling enforced softly
            not committed  enable and setpoint both pinned to 0

        **The ceiling is soft, and that is not a convenience.** A setpoint is
        slew-limited -- the plant models no actuator dynamics at all, so the
        NMPC's rate limit is the only thing standing in for a valve that cannot
        jump. When a band narrows faster than the actuator can follow, a hard
        ceiling and the rate limit have no feasible point between them: measured,
        115 of 216 solves returned `Infeasible_Problem_Detected`.

        The physically correct response to a ceiling dropping from 0.9 to 0.2 is
        a ramp -- 0.56, 0.22, 0.20 -- which necessarily spends two intervals
        above the band. A soft ceiling admits exactly that trajectory and prices
        the excursion, so the solver converges to the band as fast as the slew
        allows and no faster.

        The lower edge stays hard, because it is realised by the enable pin: a
        committed machine draws its idle load as a consequence of being
        energised, which is discrete and not slew-limited.

        `ceilings` is NaN wherever no ceiling applies.
        """
        lo = np.array(self._u_lo, dtype=float, copy=True)
        hi = np.array(self._u_hi, dtype=float, copy=True)
        guess = self._guess_u()
        ceilings = np.full(self._n_u, np.nan)

        for key in self._input_keys:
            if key in ("pv", "battery") or self._input_sizes[key] < 2:
                continue
            band = bands.get(key)
            if band is None:
                continue
            i_e = self._u_index(key, ENABLE_INDEX)
            i_s = self._u_index(key, SETPOINT_INDEX)
            if not band.committed:
                lo[i_e] = hi[i_e] = guess[i_e] = 0.0
                lo[i_s] = hi[i_s] = guess[i_s] = 0.0
                continue
            lo[i_e] = hi[i_e] = guess[i_e] = 1.0
            # The ceiling is NOT written into `hi` -- it is enforced softly, as a
            # penalised row, because a hard box on a slew-limited control is
            # infeasible the moment the band narrows faster than the actuator can
            # follow. `ceilings` carries it to the constraint builder.
            ceilings[i_s] = float(np.clip(band.setpoint_max, 0.0, hi[i_s]))
            guess[i_s] = float(np.clip(guess[i_s], lo[i_s], ceilings[i_s]))
        return lo, hi, guess, ceilings

    def _tracking_target(self, bands) -> tuple[list[int], np.ndarray]:
        """Which control columns to follow, and the value to follow, scaled.

        Only the four process setpoints. The battery and the curtailment
        fraction are deliberately *not* tracked: they are the balancing degrees
        of freedom that close the bus in real time against weather the dispatch
        layer forecast an hour ago, and pinning them to a plan would remove the
        one thing the inner layer is there to do. Enables are already pinned by
        the band, so tracking them would be redundant.

        A decommitted machine has its setpoint pinned to zero by the band, so it
        contributes nothing either way.
        """
        cols: list[int] = []
        values: list[float] = []
        if not bands:
            return cols, np.zeros(0)
        for key in self._input_keys:
            if key in ("pv", "battery") or self._input_sizes[key] < 2:
                continue
            band = bands.get(key)
            if band is None or not band.committed:
                continue
            i_s = self._u_index(key, SETPOINT_INDEX)
            cols.append(i_s)
            values.append(float(band.setpoint) / float(self._u_scale[i_s]))
        return cols, np.array(values, dtype=float)

    @staticmethod
    def _setpoint_deviation(u_block, track):
        """Sum of squared deviations from the plan, in scaled control units."""
        cols, values = track
        if not cols:
            return 0.0
        d = u_block[cols] - ca.DM(values)
        return ca.dot(d, d)

    def _rate_limited(self, u_lo: np.ndarray, u_hi: np.ndarray) -> list[int]:
        """Which control columns the rate limit may be applied to.

        Two exclusions, and the second one was a real defect.

        **Enables are never rate-limited.** Commitment is a discrete event -- a
        machine is either energised or it is not -- so bounding how fast an
        enable may change is meaningless in the first place.

        **A pinned control cannot move, so constraining its rate cannot help and
        can only hurt.** Once the dispatch layer pins an enable, the rate row
        against `last_u` demands |0 - 1| = 1.0 against a limit of 0.34. The
        feasible set is then *empty*, and IPOPT says so in seven iterations.

        That is what made 846 of 864 solves fail in the first banded run, and it
        failed in the worst possible way: every failure fell back silently to the
        dispatch layer's own request, so the run completed, reported no error,
        and produced results bit-identical to the layer below it. An inner layer
        that is doing nothing at all looks exactly like an inner layer that is
        adding nothing -- `nmpc_failed` was the only thing separating them.
        """
        movable = (u_hi - u_lo) > 1e-9
        for key in self._input_keys:
            if key not in ("pv", "battery") and self._input_sizes[key] >= 2:
                movable[self._u_index(key, ENABLE_INDEX)] = False
        return [int(i) for i in np.flatnonzero(movable)]

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
        bands: Mapping[str, Any] | None = None,
    ) -> NMPCSolution:
        """Solve the local problem over `n_steps` intervals from `t`.

        `prices` is lambda in EUR/kWh at each interval; `weather` one row per
        interval. `targets` and `terminal_prices` are keyed by full state name
        (``"gas.n_h2"``) and value the terminal state -- they are what carries
        the planner's long view into a one-hour problem.

        `bands` is the newer interface: one `Band` per subsystem, carrying a
        commitment and a setpoint ceiling. It supersedes `enables`, which passed
        only the commitment. When both are given the band wins.
        """
        if not targets or (self.objective == "economic" and not terminal_prices):
            raise ValueError(
                "InnerNMPC needs terminal inventory targets (and, in economic "
                "mode, their prices). Nothing in a one-hour horizon rewards "
                "making hydrogen or carbonate -- the methane they become is "
                "hours away -- so without a terminal term the process setpoints "
                "sit on a flat manifold, the solve does not converge, and any "
                "answer it does return is arbitrary. The layer above supplies "
                "these."
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
        if bands:
            u_lo, u_hi, u_guess, ceilings = self._bind_bands(bands)
        else:
            u_lo, u_hi, u_guess = self._bind_enables(enables)
            ceilings = np.full(self._n_u, np.nan)
        capped = [int(i) for i in np.flatnonzero(np.isfinite(ceilings))]
        movable = self._rate_limited(u_lo, u_hi)
        track = self._tracking_target(bands)

        b = NLPBuilder("nmpc")
        xv = b.variable(
            "x", (n + 1) * self.n_x,
            lb=np.tile(lo / xs, n + 1), ub=np.tile(hi / xs, n + 1),
            x0=np.tile(x0 / xs, n + 1),
        )
        # Predicted load shed, per interval, in units of the power scale.
        shed = b.variable("shed", n, lb=0.0, ub=10.0, x0=0.0)
        # Excursion above each soft band ceiling, per capped control per interval.
        over = (b.variable("band_over", n * len(capped), lb=0.0, ub=1.0, x0=0.0)
                if capped else None)

        # The rate-limit reference is where the actuator *actually is*, and it is
        # deliberately not projected into the current band.
        #
        # When a band narrows -- a ceiling dropping from 0.9 to 0.2 -- the old
        # control sits outside the new box, and a hard ceiling plus
        # `|u_0 - last_u| <= 0.34` has no feasible point: 115 of 216 solves
        # returned Infeasible_Problem_Detected this way.
        #
        # Projecting the reference to 0.2 removes the infeasibility by deleting
        # the constraint that caused it, and is wrong. The plant models no
        # actuator slew at all, so this rate limit *is* the only representation
        # of a valve that cannot jump; pretending the actuator was already at the
        # new ceiling licenses exactly the single-step jump the limit exists to
        # forbid. The physically correct trajectory is a ramp -- 0.56, 0.22, 0.20
        # -- which spends two intervals outside the band.
        #
        # So the band ceiling is soft instead (see `slack` below). The reference
        # stays where the actuator is.
        reference = None if last_u is None else np.asarray(last_u, dtype=float)
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

            # --- bus balance, with the shed the real bus can perform.
            #
            # `bus_total` is the sum of every power channel, so a positive value
            # is a deficit -- and a deficit is something the DC bus resolves by
            # shedding load. Modelling the balance as a hard equality therefore
            # made the NMPC *more rigid than the plant it controls*: wherever the
            # plant would simply shed, the NMPC had no feasible point at all.
            #
            # With commitment pinned by the dispatch layer the idle loads are
            # mandatory, so this bites exactly where it hurts -- a nearly flat
            # battery at night against a hard SoC floor. Measured: 35 % of solves
            # infeasible even after the rate-limit fix.
            #
            # The slack is non-negative (you cannot shed a surplus; curtailment
            # handles that) and priced far above any real energy value, so the
            # solver will exhaust every alternative first. Its real worth is as a
            # diagnostic: a positive shed is the inner layer predicting that the
            # dispatch layer has committed more load than the bus can carry.
            balances.append(step["bus_total"] / self._p_scale - shed[k])

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
            if self.objective == "tracking":
                # No euros. The dispatch LP already decided what to run and how
                # hard; this layer's only job is to realise that on dynamics the
                # LP could not see, and to say so when it cannot.
                profit = profit - self.setpoint_weight * self._setpoint_deviation(
                    uv[k * self._n_u:(k + 1) * self._n_u], track)
            else:
                profit = profit + dt * (
                    self.economics.p.methane_price_per_kg * r_sab * M_CH4
                ) - efc * self._battery.cost_per_efc_EUR()

            # --- rate limits. Real actuators move at a finite speed, and an
            # unconstrained NMPC will happily chatter a kiln between 0 and 1 on
            # a five-minute grid because the model lets it.
            prev = ca.DM(reference) if k == 0 and reference is not None \
                else (U[k - 1] if k > 0 else None)
            if prev is not None and len(movable):
                rates.append((U[k][movable] - prev[movable]) / Su[movable])

        # --- soft band ceilings: u <= ceiling + over, over >= 0
        if capped:
            rows = []
            for k in range(n):
                blk = uv[k * self._n_u:(k + 1) * self._n_u]
                for m, col in enumerate(capped):
                    rows.append(blk[col]
                                - float(ceilings[col] / self._u_scale[col])
                                - over[k * len(capped) + m])
            b.constraint("band_ceiling", ca.vertcat(*rows), lb=-1e3, ub=0.0)

        b.constraint("dynamics", ca.vertcat(*defects), equals=0.0)
        b.constraint(BUS_BALANCE, ca.vertcat(*balances), equals=0.0)
        if rates:
            b.constraint("rate_limit", ca.vertcat(*rates),
                         lb=-self.rate_limit, ub=self.rate_limit)

        # Band excursions are penalised well above setpoint tracking but well
        # below a shed: exceeding the plan's ceiling for an interval while the
        # actuator ramps is undesirable, and is not in the same class as failing
        # to supply the bus.
        if capped:
            profit = profit - self.band_weight * ca.sum1(over)

        shed_kWh = self._p_scale * dt / 3.6e6
        if self.objective == "tracking":
            # Dimensionless, and far above anything tracking or the terminal term
            # can offer, so a shed is never traded for a better-followed plan.
            profit = profit - self.shed_weight * ca.sum1(shed)
        else:
            # Priced per kWh actually shed, so the penalty scales with the
            # interval length rather than being an arbitrary per-row constant.
            profit = profit - self.shed_penalty_EUR_per_kWh * shed_kWh * ca.sum1(shed)

        b.maximise(profit + self._terminal_value(X[n], names, targets,
                                                 terminal_prices))
        nlp = b.build()

        # Project the stored trajectory into the *current* box before reusing it.
        #
        # The variable count does not change when a band moves, so the stored
        # solution always looks reusable -- but a band that has narrowed leaves
        # it sitting outside the new bounds, and IPOPT then spends its early
        # iterations pushing an infeasible point back onto the feasible set
        # rather than optimising. Band changes are rare while the plant is idle
        # and constant once it is cycling, which is exactly the pattern seen in
        # the solve times: under a second early in a run, several seconds once
        # every machine is committed.
        warm = None
        if self._warm is not None and self._warm.size == nlp.n_x:
            warm = np.clip(self._warm, nlp.lbx, nlp.ubx)
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
            predicted_shed_kWh=float(np.sum(solution.value("shed")) * shed_kWh),
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

        In **tracking** mode there are no prices at all. The deviation is
        normalised by the state's own characteristic magnitude and weighted by a
        dimensionless `terminal_weight`, so ending the horizon 10 % away from the
        planned inventory costs `0.01 * terminal_weight` against a fully
        saturated setpoint's `setpoint_weight`. What a banked mole is *worth*
        was settled by the dispatch LP; this layer only needs to know where the
        plan expects the buffers to be.
        """
        if not targets:
            return 0.0
        tracking = self.objective == "tracking"
        if not tracking and not terminal_prices:
            return 0.0
        penalty = 0.0
        index = {name: i for i, name in enumerate(names)}
        for name, target in targets.items():
            if name not in index:
                continue
            if not tracking and name not in terminal_prices:
                continue
            i = index[name]
            scale = max(float(self._x_scale[i]), 1e-9)
            deviation = (x_end[i] - float(target)) / scale
            weight = self.tracking_terminal_weight if tracking else (
                self.terminal_weight * float(terminal_prices[name]) * scale)
            penalty = penalty + weight * deviation * deviation
        return -penalty
