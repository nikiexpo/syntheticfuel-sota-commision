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

#: Lower bound on every penalised slack, instead of exactly zero.
#:
#: IPOPT relaxes each bound by `bound_relax_factor * max(1, |bound|)`, default
#: 1e-8, so a slack declared `lb = 0` may sit at -1e-8. That is numerically
#: nothing -- until it is multiplied by an exact-penalty weight of 1e6, at which
#: point it contributes -0.06 to an objective whose edit term is 0.003. Measured:
#: `shed` and `state_slack` together supplied -0.26 of a -0.254 objective, 86x
#: the term the filter is supposed to be minimising, so the solver's strongest
#: gradient pointed at mining bound tolerance rather than at editing the plan.
#:
#: Setting the bound to the relaxation width makes the *relaxed* bound exactly
#: zero: 1e-8 - 1e-8 * max(1, 1e-8) = 0. The slack can still reach zero, so no
#: constant offset is introduced, but it can no longer go negative and be paid
#: for it. Cheaper than `bound_relax_factor = 0`, which removes a relaxation the
#: barrier method wants and measured 39 % slower.
SLACK_FLOOR = 1e-8

#: State-name suffixes that get no soft box in filter mode.
#:
#: The soft state box is the largest block in the problem -- 174 of 360 columns
#: at the default horizon, plus a constraint row each -- and most of it protects
#: states that cannot reach a bound inside a five-minute horizon no matter what
#: the controller does. Two structural kinds:
#:
#: * **Monotone accumulators** (`efc`, `consumed_kg`, `ch4_kg`): they only ever
#:   increase, so the lower bound of zero they start above is unreachable, and
#:   they have no upper bound at all.
#: * **Degradation states** (`fade`, `cycle_number`, `v_degradation`,
#:   `catalyst_activity`): these move on timescales of weeks. Their bounds are
#:   real, but nothing a filter decides in five minutes can approach one.
#:
#: Excluded states keep their **hard** box, which is the safe direction: a slack
#: that can never be needed is removed, not a constraint. If a horizon long
#: enough to move these is ever wanted, this list is what has to change.
SLOW_STATES: frozenset[str] = frozenset({
    "fade", "efc", "cycle_number", "v_degradation", "consumed_kg", "ch4_kg",
    "catalyst_activity",
})


def _shift(values: np.ndarray, steps: int, stride: int | None) -> np.ndarray:
    """Advance a stacked per-interval trajectory by one interval.

    `values` is `steps` blocks of `stride` entries. The result drops the first
    block and repeats the last, which is the guess a receding horizon implies:
    what was predicted for interval k+1 is now the prediction for interval k,
    and the new final interval has no predecessor to copy but its own.

    Blocks with no per-interval structure (`stride is None`) are returned
    unchanged -- the slacks, whose previous values are still a reasonable
    starting point but have no meaningful ordering to advance.
    """
    if stride is None or steps < 2 or values.size != steps * stride:
        return values
    out = values.reshape(steps, stride)
    return np.vstack([out[1:], out[-1:]]).ravel()

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
    #: Filter mode only. `edit` is ||u_0 - u_plan||^2 in scaled control units --
    #: how far the filter moved the plan's proposed action. `binding` is the
    #: dual mass per inequality block, i.e. which constraints produced that
    #: edit; `counterfactual` is what the plan's own action would have violated
    #: had it been applied unchanged. See `_attribution`.
    edit: float = 0.0
    binding: dict[str, float] = field(default_factory=dict)
    counterfactual: dict[str, float] = field(default_factory=dict)
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
        max_iter: int = 500,
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
        #: --- filter mode ---------------------------------------------------
        #: Exact-penalty weights. Ordered by what a violation actually means:
        #: overshooting a plan band or a slew limit while ramping is a nuisance;
        #: leaving a state box is a safety matter; failing to supply the bus is
        #: the thing this layer exists to prevent. Each must be large enough
        #: that the violation is preferred only over infeasibility, and the edit
        #: term is O(1) by construction, so these are absolute.
        filter_rho_band: float = 1.0e3,
        filter_rho_rate: float = 1.0e3,
        filter_rho_state: float = 1.0e5,
        filter_rho_shed: float = 1.0e6,
        #: Tail regularisation only. Present so the horizon past u_0 is
        #: well-posed, and small enough that it cannot shape u_0 -- which is the
        #: only thing a filter is allowed to decide.
        filter_epsilon: float = 1.0e-3,
        #: How far outside its true box a state variable may range before the
        #: *variable* bound bites, as a fraction of the box span. The soft rows
        #: sit at the true bounds; this is only here to keep the NLP bounded.
        filter_state_margin: float = 0.25,
        #: Drop the soft box on states that cannot reach a bound within the
        #: horizon. See `SLOW_STATES`.
        filter_trim_slacks: bool = True,
        #: Advance the stored solution by one interval before reusing it.
        #: Off reproduces the previous behaviour, which reused the vector
        #: as-is -- every entry one interval stale. See `_warm_start`.
        warm_shift: bool = True,
    ) -> None:
        self.plant = plant
        self.economics = economics
        self.horizon_s = float(horizon_s)
        self.dt_s = float(dt_s)
        self.n_steps = max(1, int(round(self.horizon_s / self.dt_s)))
        self.rate_limit = float(rate_limit)
        self.terminal_weight = float(terminal_weight)
        self.shed_penalty_EUR_per_kWh = float(shed_penalty_EUR_per_kWh)
        if objective not in ("economic", "tracking", "filter"):
            raise ValueError("objective must be 'economic', 'tracking' or "
                             f"'filter', got {objective!r}")
        self.objective = objective
        self.setpoint_weight = float(setpoint_weight)
        self.band_weight = float(band_weight)
        self.tracking_terminal_weight = float(tracking_terminal_weight)
        self.shed_weight = float(shed_weight)
        self.filter_rho_band = float(filter_rho_band)
        self.filter_rho_rate = float(filter_rho_rate)
        self.filter_rho_state = float(filter_rho_state)
        self.filter_rho_shed = float(filter_rho_shed)
        self.filter_epsilon = float(filter_epsilon)
        self.filter_state_margin = float(filter_state_margin)
        self.filter_trim_slacks = bool(filter_trim_slacks)
        self.warm_shift = bool(warm_shift)
        self.backend = (
            backend if isinstance(backend, SolverBackend)
            else get_backend(backend, max_iter=max_iter, tol=tol,
                             acceptable_tol=tol * 100.0)
        )

        self._input_keys = [key for key, sub in plant if sub.n_inputs]
        self._input_sizes = {key: plant[key].n_inputs for key in self._input_keys}
        self._n_u = sum(self._input_sizes.values())
        self._warm: dict[str, np.ndarray] | None = None
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

    def _rate_limited(self, u_lo: np.ndarray, u_hi: np.ndarray,
                      process_only: bool = False) -> list[int]:
        """Which control columns the rate limit may be applied to.

        **`process_only` excludes the battery and the curtailment fraction**,
        and that exclusion is the point of the argument. Neither has a physical
        slew limit worth modelling -- a converter responds in milliseconds, and
        curtailment is a firing-angle change -- yet both were being limited to
        34 % of span per interval. Those are precisely the two channels whose
        purpose is to close the bus balance in real time, so limiting them
        manufactures the imbalance that the shed slack then has to absorb.

        The limit is meaningful for the four process setpoints, which stand in
        for valves and heaters the plant model does not represent at all.

        Two further exclusions, and the second was a real defect.

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
        offset = 0
        for key in self._input_keys:
            size = self._input_sizes[key]
            if key in ("pv", "battery"):
                if process_only:
                    movable[offset:offset + size] = False
            elif size >= 2:
                movable[self._u_index(key, ENABLE_INDEX)] = False
            offset += size
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
        reference_u: np.ndarray | None = None,
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
        filtering = self.objective == "filter"
        if filtering and reference_u is None:
            raise ValueError(
                "filter mode needs `reference_u`: the complete action the "
                "dispatch layer proposes, which is the thing being minimally "
                "edited. Without it there is nothing to stay close to and the "
                "filter degenerates into an unconstrained feasibility problem."
            )
        if not filtering and (
                not targets or (self.objective == "economic" and not terminal_prices)):
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
        movable = self._rate_limited(u_lo, u_hi, process_only=filtering)
        track = self._tracking_target(bands)

        b = NLPBuilder("nmpc")

        # In filter mode the state box is relaxed into the objective. A hard box
        # on top of hard dynamics and a hard initial state is infeasible whenever
        # the current state plus the model implies a bound crossing that no
        # admissible control can prevent -- a flat battery at night against a
        # hard SoC floor, with committed machines drawing mandatory idle load.
        # That is exactly when a safety filter is most needed, so it is exactly
        # when it must not fail.
        #
        # The *variable* bound is widened by a margin rather than removed, to
        # keep the NLP bounded; the true bounds are re-imposed below as
        # penalised rows.
        # Cumulative counters (methane made, water consumed, equivalent full
        # cycles) have no upper bound at all, so `hi` is +inf for them. Those
        # columns get no soft row -- a row against an infinite bound evaluates
        # to Inf and the solve dies with `Invalid_Number_Detected` before it
        # takes a single iteration.
        soft_lo = np.isfinite(lo)
        soft_hi = np.isfinite(hi)
        if filtering and self.filter_trim_slacks:
            slow = np.array([nm.split(".")[-1] in SLOW_STATES for nm in names])
            soft_lo &= ~slow
            soft_hi &= ~slow
        if filtering:
            span = np.where(soft_lo & soft_hi, np.maximum(hi - lo, 1e-9), 0.0)
            x_lb = np.where(soft_lo, lo - self.filter_state_margin * span, lo) / xs
            x_ub = np.where(soft_hi, hi + self.filter_state_margin * span, hi) / xs
        else:
            x_lb, x_ub = lo / xs, hi / xs
        xv = b.variable(
            "x", (n + 1) * self.n_x,
            lb=np.tile(x_lb, n + 1), ub=np.tile(x_ub, n + 1),
            x0=np.tile(x0 / xs, n + 1),
        )
        # Predicted load shed, per interval, in units of the power scale.
        #
        # **A slack with a finite upper bound does not soften a constraint** --
        # it relocates the infeasibility. The tracking formulation capped these
        # at 10 and 1 respectively, which is why `Infeasible_Problem_Detected`
        # survived the soft-constraint work. In filter mode they are unbounded.
        slack_ub = np.inf if filtering else 10.0
        shed = b.variable("shed", n, lb=SLACK_FLOOR, ub=slack_ub,
                          x0=SLACK_FLOOR)
        # Excursion above each soft band ceiling, per capped control per interval.
        over = (b.variable("band_over", n * len(capped), lb=SLACK_FLOOR,
                           ub=np.inf if filtering else 1.0, x0=SLACK_FLOOR)
                if capped else None)
        # Filter-mode only: state-box excursion, and slew-limit excursion.
        soft_cols = [(i, s) for i in range(self.n_x)
                     for s, ok in ((0, soft_lo[i]), (1, soft_hi[i])) if ok]
        xi = (b.variable("state_slack", (n + 1) * len(soft_cols),
                         lb=SLACK_FLOOR, ub=np.inf, x0=SLACK_FLOOR)
              if filtering and soft_cols else None)
        eta = (b.variable("rate_slack", n * len(movable), lb=SLACK_FLOOR,
                          ub=np.inf, x0=SLACK_FLOOR)
               if filtering and movable else None)

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
            if filtering:
                # Nothing per-stage. A filter's objective is the size of the
                # edit to u_0, not the quality of the trajectory: steps 1..N-1
                # exist only to certify that a feasible continuation exists.
                # Anything summed over the horizon here would make this a
                # tracking controller again.
                pass
            elif self.objective == "tracking":
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
                delta = (U[k][movable] - prev[movable]) / Su[movable]
                if filtering:
                    # Two one-sided rows against a shared slack, so the limit is
                    # soft in both directions:  -rho - eta <= d <= rho + eta.
                    e = eta[k * len(movable):(k + 1) * len(movable)]
                    rates.append(delta - self.rate_limit - e)      # <= 0
                    rates.append(-delta - self.rate_limit - e)     # <= 0
                else:
                    rates.append(delta)

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
            if filtering:
                b.constraint("rate_limit", ca.vertcat(*rates), lb=-1e6, ub=0.0)
            else:
                b.constraint("rate_limit", ca.vertcat(*rates),
                             lb=-self.rate_limit, ub=self.rate_limit)

        # --- soft state box:  x_lo - xi_minus <= x <= x_hi + xi_plus
        if filtering and xi is not None:
            rows = []
            lo_s, hi_s = lo / xs, hi / xs
            for k in range(n + 1):
                blk = xv[k * self.n_x:(k + 1) * self.n_x]
                base = k * len(soft_cols)
                for m, (i, side) in enumerate(soft_cols):
                    s = xi[base + m]
                    rows.append(blk[i] - float(hi_s[i]) - s if side
                                else float(lo_s[i]) - blk[i] - s)   # <= 0
            b.constraint("state_box", ca.vertcat(*rows), lb=-1e6, ub=0.0)

        # Band excursions are penalised well above setpoint tracking but well
        # below a shed: exceeding the plan's ceiling for an interval while the
        # actuator ramps is undesirable, and is not in the same class as failing
        # to supply the bus.
        if capped and not filtering:
            profit = profit - self.band_weight * ca.sum1(over)

        shed_kWh = self._p_scale * dt / 3.6e6
        if filtering:
            # --- the whole filter objective, assembled here.
            #
            #   min  ||u_0 - u_plan||^2_W  +  rho . (violations)  +  eps . (tail)
            #
            # The edit term is O(1): controls are scaled to their own bounds, so
            # a full-span edit on one column costs 1. That is what makes the
            # exact-penalty weights absolute rather than relative.
            ref = np.asarray(reference_u, dtype=float) / us
            d0 = uv[0:self._n_u] - ca.DM(ref)
            edit = ca.dot(d0, d0)

            # Tail regularisation. Not tracking -- it has no reference at all,
            # it only discourages the tail from chattering, which would leave
            # u_0 sitting at the end of a trajectory the solver never had to
            # make smooth. Small enough that it cannot shape u_0.
            tail = 0.0
            for k in range(1, n):
                dk = uv[k * self._n_u:(k + 1) * self._n_u] \
                     - uv[(k - 1) * self._n_u:k * self._n_u]
                tail = tail + ca.dot(dk, dk)

            penalty = self.filter_rho_shed * ca.sum1(shed)
            if xi is not None:
                penalty = penalty + self.filter_rho_state * ca.sum1(xi)
            if eta is not None:
                penalty = penalty + self.filter_rho_rate * ca.sum1(eta)
            if capped:
                penalty = penalty + self.filter_rho_band * ca.sum1(over)
            profit = -(edit + penalty + self.filter_epsilon * tail)
        elif self.objective == "tracking":
            # Dimensionless, and far above anything tracking or the terminal term
            # can offer, so a shed is never traded for a better-followed plan.
            profit = profit - self.shed_weight * ca.sum1(shed)
        else:
            # Priced per kWh actually shed, so the penalty scales with the
            # interval length rather than being an arbitrary per-row constant.
            profit = profit - self.shed_penalty_EUR_per_kWh * shed_kWh * ca.sum1(shed)

        # No terminal term in filter mode. A terminal inventory penalty is an
        # economic objective, and at the tracking weights it outweighed setpoint
        # following a hundredfold -- a second optimiser on a one-hour horizon,
        # overriding the 240-hour layer that had far better information. A
        # filter that penalises only u_0 has no incentive to drain a buffer,
        # because it is not optimising the tail at all.
        b.maximise(profit if filtering else
                   profit + self._terminal_value(X[n], names, targets,
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
        # How many intervals each variable block spans, so the warm start can
        # advance all of them together. Built here because this is where the
        # block sizes are known.
        warm = self._warm_start(nlp, {
            "x": n + 1, "state_slack": n + 1,
            "u": n, "shed": n, "band_over": n, "rate_slack": n,
        })
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

        self._warm = {name: np.asarray(solution.value(name), dtype=float).ravel()
                      for name in nlp.var_blocks}
        edit, binding, counterfactual = (
            self._attribution(solution, nlp, n, reference_u, soft_cols, capped,
                              movable, x0, weather, u_lo, ceilings)
            if filtering else (0.0, {}, {}))
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
            edit=edit, binding=binding, counterfactual=counterfactual,
            stats=solution.stats,
            states=states,
            controls=solution.value("u").reshape(n, self._n_u) * us,
        )

    def _warm_start(self, nlp, steps: Mapping[str, int]) -> np.ndarray | None:
        """The previous solution, advanced one interval, in this problem's layout.

        Two defects in the version this replaces, and the docstring at the top of
        this module described the fixed behaviour rather than the actual one --
        it claimed the solve was "warm-started from the previous solve shifted by
        one step, which is the standard MPC trick and is worth an order of
        magnitude here". Nothing shifted anything.

        **It was not shifted.** The stored vector was reused as-is, so at the
        next interval every entry was one step stale: the guess for `x_0` was the
        old `x_0`, a full interval behind the state actually measured. On a ramp
        that is a worse starting point than it looks, and it showed -- 127 mean
        iterations where a warm-started receding-horizon NMPC converges in tens.

        **It was discarded whenever the size changed.** The guard was
        `self._warm.size == nlp.n_x`, and the size moves with commitment, because
        `band_over` and `rate_slack` are sized by how many machines are committed
        and movable. So the warm start vanished at exactly the transitions where
        a good guess is worth most. Storing per *block* instead means a changed
        slack count costs only that block, and the states and controls -- the
        expensive part to get right -- carry over regardless.

        The shift itself is the standard one: with the horizon advanced by one
        interval, the old `x_1..x_N` become the new `x_0..x_{N-1}`, and the last
        entry is repeated because nothing is known about it yet.

        **Every block is shifted, not just the states and controls.** A first
        version advanced `x` and `u` and left the slacks alone, which measured
        44 % *slower* than not shifting at all: the slacks are per-interval too,
        so the result was a warm start whose interval-k slack belonged to the
        constraint at interval k-1. A uniformly stale guess is self-consistent;
        a half-shifted one is not, and IPOPT spent its early iterations undoing
        the mismatch. `steps` carries how many intervals each block spans, since
        that is what `solve` knows and this method cannot infer safely.
        """
        if not self._warm:
            return None
        guess = np.array(nlp.x0, dtype=float, copy=True)
        for name, block in nlp.var_blocks.items():
            prev = self._warm.get(name)
            if prev is None:
                continue
            if self.warm_shift and name in steps and steps[name] > 1:
                count = int(steps[name])
                if prev.size % count == 0:
                    prev = _shift(prev, count, prev.size // count)
            if prev.size == block.size:
                guess[block.slice] = prev
        return np.clip(guess, nlp.lbx, nlp.ubx)

    def _attribution(self, solution, nlp, n, reference_u, soft_cols, capped,
                     movable, x0, weather, u_lo, ceilings):
        """Why did the filter move the plan, and what was pressing on it?

        Two independent signals, because they answer the question differently
        and disagreeing would itself be informative.

        **Duals.** At the optimum `grad f = 2 (u_0 - u_plan)` is balanced by
        `sum_i lam_i grad g_i`, so the multipliers say which constraints
        produced the edit and in what proportion. Reported as dual mass,
        `sum |lam|`, per inequality block, with the state box broken down per
        state and side -- "the SoC floor" is a usable answer where "block
        state_box" is not.

        `dynamics` and `initial_state` are deliberately excluded. Their
        multipliers are never zero: they are adjoint variables propagating the
        effect of a constraint, not the constraint that caused it, and
        including them would swamp the attribution with something that is
        always large and never informative.

        **Counterfactual.** What the plan's own action would have violated, had
        it been applied unchanged -- obtained by rolling `u_plan` forward
        through the same nonlinear model with a zero-order hold. This is the
        legible half: "the plan would have taken SoC below its floor" reads as
        an explanation in a way that a multiplier does not.
        """
        out_edit = 0.0
        if reference_u is not None:
            d = (np.asarray(solution.value("u"), dtype=float)[:self._n_u]
                 - np.asarray(reference_u, dtype=float) / self._u_scale)
            out_edit = float(d @ d)

        binding: dict[str, float] = {}
        names = self.plant.state_names()
        for block in ("bus_balance", "band_ceiling", "rate_limit", "state_box"):
            if block not in nlp.con_blocks:
                continue
            lam = np.abs(np.asarray(solution.dual(block), dtype=float).ravel())
            binding[block] = float(lam.sum())
            if block == "state_box" and soft_cols and lam.size % len(soft_cols) == 0:
                per = lam.reshape(-1, len(soft_cols)).sum(axis=0)
                for m, (i, side) in enumerate(soft_cols):
                    if per[m] > 0.0:
                        tag = f"{names[i]}.{'hi' if side else 'lo'}"
                        binding[tag] = float(per[m])

        # --- counterfactual: apply the plan unchanged and see what breaks
        counter: dict[str, float] = {}
        if reference_u is not None:
            u_ref = np.asarray(reference_u, dtype=float)
            u_map, off = {}, 0
            for key in self._input_keys:
                sz = self._input_sizes[key]
                u_map[key] = u_ref[off:off + sz]
                off += sz
            for key, sub in self.plant:
                u_map.setdefault(key, np.zeros(sub.n_inputs))
            x = np.asarray(x0, dtype=float).copy()
            worst_lo = np.zeros(self.n_x)
            worst_hi = np.zeros(self.n_x)
            deficit_W = 0.0
            try:
                for k in range(n):
                    w = weather[min(k, len(weather) - 1)]
                    # The bus first, on the state the plan would actually be in.
                    # This is the constraint the plan usually breaks -- its own
                    # balance is an hourly average over a seven-state reduced
                    # model, and this one is instantaneous on all sixteen -- so
                    # a counterfactual that checked only state boxes came back
                    # empty while the duals pointed squarely at the bus.
                    step = self._step_fn(x=x, u=np.concatenate(
                        [u_map[key] for key in self._input_keys]),
                        w=self._weather_vector(w))
                    deficit_W = max(deficit_W, float(step["bus_total"]))
                    x = self.plant.step(0.0, x, u_map, w, self.dt_s, clip=False)
                    worst_lo = np.maximum(worst_lo, self._x_lo - x)
                    worst_hi = np.maximum(worst_hi, x - self._x_hi)
            except Exception:
                return out_edit, binding, counter
            if deficit_W > 1.0:
                counter["bus_deficit_W"] = float(deficit_W)
            for i in range(self.n_x):
                if np.isfinite(worst_lo[i]) and worst_lo[i] > 1e-9:
                    counter[f"{names[i]}.lo"] = float(worst_lo[i])
                if np.isfinite(worst_hi[i]) and worst_hi[i] > 1e-9:
                    counter[f"{names[i]}.hi"] = float(worst_hi[i])
        return out_edit, binding, counter

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
