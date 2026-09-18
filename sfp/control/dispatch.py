"""The economic dispatch layer (L2) -- a mixed-integer linear program.

Unit commitment over storage, in place of `EconomicPlanner`'s nonlinear program.
It decides *all* the economics and none of the dynamics; the inner NMPC does the
reverse.

What it publishes downward is a **dispatch band** per machine per interval:

    committed      idle <= dispatch <= dispatch_max
    not committed  dispatch = 0

which encodes commitment as a *bound* rather than a flag. The inner layer then
needs no enable variable, no smoothed gate, and no rounding threshold -- the
band's lower edge is realised by pinning the enable (a committed machine draws
its idle load) and the upper edge is a box on the setpoint. Both are always
satisfiable, which is why this stage carries no slack variables.

Measured against the NLP planner it replaces, on the same plant and weather:

    horizon      NLP planner        this LP
      72 h       155 s              0.035 s
     168 h       fails (cap)        0.175 s
     240 h       fails (cap)        0.336 s

The 240 h horizon of the original specification is therefore back in reach, and
with it the multi-day trades the architecture was built around.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from sfp.control.base import ControlContext, Controller
from sfp.control.dispatch_model import (
    N_Z, SUBSYSTEMS, CalcinerThermal, DispatchModel, PWLMap, zi,
)

#: Machines with a piecewise-linear rate map. The calciner is thermal instead.
PWL_SUBSYSTEMS: tuple[str, ...] = ("contactor", "electrolyser", "sabatier")
from sfp.economics import Economics
from sfp.sim.bus import Request
from sfp.solvers.lp import LPBuilder, solve_lp
from sfp.units import M_CH4, M_H2O

BUS_BALANCE = "bus_balance"

#: Wear charge per start, EUR. Only the calciner and reactor figures come from
#: `economics.yaml`'s `startup_cost_EUR`, whose note attributes it to "thermal
#: cycling damage to the kiln refractory and the Sabatier catalyst". The fan and
#: the stack are cheaper to cycle and are assumed, not sourced.
START_COST_EUR: dict[str, float] = {
    "contactor": 0.5, "calciner": 40.0, "electrolyser": 5.0, "sabatier": 40.0,
}

#: Intervals a machine must stay committed once started. This is what carries
#: the kiln's thermal inertia now that its temperature is not a state: a kiln
#: worth lighting is a kiln worth running for a while.
MIN_UPTIME: dict[str, int] = {
    "contactor": 1, "calciner": 3, "electrolyser": 1, "sabatier": 2,
}


@dataclass(frozen=True)
class Band:
    """What one machine may do over one interval."""

    committed: bool
    setpoint_max: float
    dispatch: float          # what the plan itself intended

    @property
    def setpoint(self) -> float:
        return self.setpoint_max if self.committed else 0.0


@dataclass
class DispatchPlan:
    """A solved schedule, in the form the inner layer consumes."""

    t0_s: float
    dt_s: float
    states: np.ndarray                      # (N+1, N_Z)
    bands: list[dict[str, Band]]            # N entries
    lambda_EUR_per_kWh: np.ndarray          # (N,)
    objective_EUR: float
    horizon_s: float
    battery_W: np.ndarray = field(          # (N, 2) charge, discharge
        default_factory=lambda: np.zeros((0, 2)))
    curtail: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Operating profit the plan expects from each interval, EUR. Excludes the
    #: terminal inventory value, which belongs to the horizon rather than to any
    #: interval. Recovered as `-c_k . x_k` over each control block -- the
    #: objective is linear, so this is exact and needs no second copy of the
    #: cost coefficients.
    stage_profit_EUR: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wall_time_s: float = 0.0

    def profit_between(self, t_from: float, t_to: float) -> float:
        """Planned profit over a window, EUR. Intervals are whole-hour blocks."""
        if len(self.stage_profit_EUR) == 0:
            return float("nan")
        lo = self.index_at(t_from)
        hi = self.index_at(max(t_to - 1.0, t_from))
        return float(np.sum(self.stage_profit_EUR[lo:hi + 1]))

    def state_at(self, t_s: float) -> np.ndarray:
        """The buffer state the plan expects at the *start* of the interval."""
        return self.states[self.index_at(t_s)]

    def index_at(self, t_s: float) -> int:
        k = int((t_s - self.t0_s) // self.dt_s)
        return int(np.clip(k, 0, len(self.bands) - 1))

    def bands_at(self, t_s: float) -> dict[str, Band]:
        return self.bands[self.index_at(t_s)]

    def price_at(self, t_s: float) -> float:
        return float(self.lambda_EUR_per_kWh[self.index_at(t_s)])

    def target_at(self, t_s: float) -> np.ndarray:
        """The buffer state the plan expects at the *end* of the interval holding t."""
        return self.states[min(self.index_at(t_s) + 1, len(self.states) - 1)]


class _Layout:
    """Flat index map. States first, then one control block per interval.

    The calciner carries two columns of its own -- heater power and calcination
    rate -- rather than a segment family, because it is a thermal system whose
    input and output are only coupled through a temperature state.
    """

    def __init__(self, n: int, segments: Mapping[str, int]) -> None:
        self.n = n
        self.u0 = N_Z * (n + 1)
        off = 0
        self.ie = off; off += len(SUBSYSTEMS)
        self.iseg: dict[str, tuple[int, int]] = {}
        for key in PWL_SUBSYSTEMS:
            self.iseg[key] = (off, off + segments[key]); off += segments[key]
        self.ical_p = off; off += 1          # kiln heater power, W
        self.ical_r = off; off += 1          # calcination rate, mol/s
        self.ical_y = off; off += 1          # "hot enough to calcine", [0,1]
        self.istart = off; off += len(SUBSYSTEMS)
        self.ipc = off; off += 1
        self.ipd = off; off += 1
        self.igam = off; off += 1
        self.ivh = off; off += 1
        self.ivc = off; off += 1
        self.imw = off; off += 1
        self.n_u = off
        self.n_x = self.u0 + self.n_u * n

    def z(self, k: int, i: int) -> int:
        return N_Z * k + i

    def u(self, k: int, i: int) -> int:
        return self.u0 + self.n_u * k + i

    def e(self, k: int, key: str) -> int:
        return self.u(k, self.ie + SUBSYSTEMS.index(key))

    def start(self, k: int, key: str) -> int:
        return self.u(k, self.istart + SUBSYSTEMS.index(key))

    def seg(self, k: int, key: str) -> range:
        a, b = self.iseg[key]
        return range(self.u(k, a), self.u(k, b))


class EconomicDispatch(Controller):
    """Receding-horizon economic dispatch: a linear program over six buffers."""

    name = "dispatch"
    description = (
        "Economic dispatch LP: unit commitment over six buffers on a 10-day "
        "hourly horizon, re-solved every 3 h. Publishes dispatch bands."
    )

    def __init__(
        self,
        *,
        #: Ten days at hourly resolution -- the horizon the original formulation
        #: asked for and the NLP could never reach. It costs 0.34 s here.
        horizon_hours: int = 240,
        replan_interval_s: float = 3 * 3600.0,
        terminal_value_fraction: float = 0.5,
        #: Where a relaxed commitment rounds to a committed machine.
        commit_threshold: float = 0.5,
        #: How far out the kiln's "hot enough to calcine" indicator is required
        #: to be integral; beyond this it is relaxed to [0,1].
        #:
        #: It has to be binary somewhere -- the relaxation leaks badly enough to
        #: schedule calcination on a cold kiln (see `_build`) -- but only where
        #: the schedule is *implemented*. Measured at 240 h: 24 binary hours
        #: solves in 12 s, 48 in 28 s, 72 in 120 s (capped), while the objective
        #: moves 2818 -> 2813 -> 2810, i.e. 0.3 %. Only the first three hours
        #: are ever implemented, so 24 is a wide margin.
        binary_horizon_hours: int = 24,
        time_limit_s: float = 120.0,
        #: Recover lambda from the MILP. **Off by default: nothing reads it.**
        #: The filter has no economics, so the price survives only as a reported
        #: diagnostic, and recovering it costs a second LP solve per replan
        #: (integers fixed, continuous problem re-solved). Turn it on for a
        #: reporting run.
        recover_duals: bool = False,
        name: str | None = None,
    ) -> None:
        self.horizon_hours = int(horizon_hours)
        self.replan_interval_s = float(replan_interval_s)
        self.terminal_value_fraction = float(terminal_value_fraction)
        self.commit_threshold = float(commit_threshold)
        self.binary_horizon_hours = int(binary_horizon_hours)
        self.time_limit_s = float(time_limit_s)
        self.recover_duals = bool(recover_duals)
        if name:
            self.name = name

        self.model: DispatchModel | None = None
        self.plan: DispatchPlan | None = None
        self._next_replan_s = -np.inf
        self._diagnostics: dict[str, float] = {}
        self._failures = 0
        self._solves = 0
        self._last_t = 0.0

    # --- lifecycle --------------------------------------------------------
    def reset(self, context: ControlContext) -> None:
        super().reset(context)
        self.model = DispatchModel(context.plant)
        self.economics = context.economics or Economics()
        self.plan = None
        self._next_replan_s = -np.inf
        self._diagnostics = {}
        self._failures = 0
        self._solves = 0
        # What is running as the next plan begins. Seeded from the plant's own
        # initial enables (everything cold and off), then tracked from what this
        # controller actually commands.
        self._prev_commit = {key: 0.0 for key in SUBSYSTEMS}

    # --- the control law --------------------------------------------------
    def act(self, t, state, measurement, forecast=None) -> Request:
        self._last_t = t
        if t >= self._next_replan_s:
            self._replan(t, state, forecast)
            self._next_replan_s = t + self.replan_interval_s

        if self.plan is None:
            return Request.all_off()

        k = self.plan.index_at(t)
        bands = self.plan.bands[k]
        self._prev_commit = {key: 1.0 if bands[key].committed else 0.0
                             for key in SUBSYSTEMS}
        return Request(
            setpoints={key: bands[key].setpoint for key in SUBSYSTEMS},
            enables={key: 1.0 if bands[key].committed else 0.0 for key in SUBSYSTEMS},
            battery_charge_W=float(self.plan.battery_W[k, 0]),
            battery_discharge_W=float(self.plan.battery_W[k, 1]),
            curtail_fraction=float(self.plan.curtail[k]),
        )

    # --- planning ---------------------------------------------------------
    #: Buffers the divergence diagnostic reports on, with the scale each error
    #: is normalised by. These are the quantities the plan is actually steering;
    #: the kiln temperature is excluded because it is a means, not a store.
    DIVERGENCE_STATES: tuple[str, ...] = ("soc", "n_h2", "n_co2", "n_caco3")

    def _record_divergence(self, t: float, z0: np.ndarray) -> None:
        """How far the plant has drifted from the plan since the last replan.

        This is what decides whether a plan may stand in for a simulation. Two
        errors, answering different questions:

        `plan_divergence_*`  where the buffers are against where the previous
                             plan said they would be, normalised by capacity --
                             physical drift of the linearised model.

        `plan_profit_window_EUR`  what the superseded plan expected to earn over
                             the window just run, for comparison against what
                             was realised. Drift can be large while the
                             economics land, and small while a mis-timed
                             commitment costs real money.
        """
        # `self.plan` is still the plan in force since its own t0; it is
        # overwritten further down. A separate `_prev_plan` assigned at the end
        # of a replan would give the plan from two replans ago, spanning two
        # replan intervals against one realised -- which halves every ratio and
        # looks exactly like the model over-predicting by two.
        prev = self.plan
        if prev is None or len(prev.stage_profit_EUR) == 0:
            return
        if not (0.0 < t - prev.t0_s <= 1.5 * self.replan_interval_s):
            return
        predicted = prev.state_at(t)
        scale = np.array([
            float(self.model.battery.p.soc_max),
            float(self.model.gas.p.h2_capacity_mol),
            float(self.model.gas.p.co2_capacity_mol),
            float(self.model.n_total_mol),
        ])
        idx = [zi(n) for n in self.DIVERGENCE_STATES]
        err = (z0[idx] - predicted[idx]) / scale
        for name, e in zip(self.DIVERGENCE_STATES, err):
            self._diagnostics[f"plan_divergence_{name}"] = float(e)
        self._diagnostics["plan_divergence_rms"] = float(np.sqrt(np.mean(err ** 2)))
        self._diagnostics["plan_profit_window_EUR"] = prev.profit_between(
            prev.t0_s, t)
        self._diagnostics["plan_window_s"] = float(t - prev.t0_s)

    def _replan(self, t: float, state, forecast) -> None:
        started = time.perf_counter()
        model = self.model
        z0 = model.initial_state(state)
        self._record_divergence(t, z0)
        rows = self._forecast_rows(t, forecast)
        n = len(rows["pv_available_W"])
        if n == 0:
            return

        # Re-linearise around the conditions the horizon expects. The maps drift
        # with ambient temperature and with how full the sorbent is, so deriving
        # them per replan makes this a successive-linearisation scheme rather
        # than one fixed approximation carried through a whole run.
        loading = float(z0[zi("n_caco3")]) / max(
            model.n_total_mol * float(model.solids.max_conversion(z0[zi("cycle_number")])),
            1.0)
        nominal = {k: float(np.mean(v)) for k, v in rows.items() if k != "pv_available_W"}
        pwl = model.pwl(nominal, loading)
        kiln = model.thermal()

        problem, layout = self._build(n, z0, rows, pwl, kiln)
        solution = solve_lp(problem, time_limit_s=self.time_limit_s,
                            duals=self.recover_duals)
        self._solves += 1

        if not solution.success:
            self._failures += 1
            self._diagnostics.update(
                plan_solve_failed=1.0, plan_failures=float(self._failures))
            return

        self.plan = self._unpack(t, n, solution, layout, pwl, kiln, problem)
        self.plan.wall_time_s = time.perf_counter() - started
        self._diagnostics.update(
            #: monotonic, so the report layer can group a log by replan window
            #: without having to infer boundaries from values that may repeat
            plan_replan_index=float(self._solves),
            plan_objective_EUR=-solution.f,
            plan_solve_time_s=self.plan.wall_time_s,
            plan_lp_time_s=solution.wall_time_s,
            plan_solve_failed=0.0,
            plan_failures=float(self._failures),
            plan_intervals=float(n),
            plan_lambda_mean=float(np.mean(self.plan.lambda_EUR_per_kWh)),
            plan_lambda_max=float(np.max(self.plan.lambda_EUR_per_kWh)),
        )

    # --- LP assembly ------------------------------------------------------
    def _build(self, n, z0, rows, pwl, kiln: CalcinerThermal):
        model = self.model
        L = _Layout(n, {k: len(pwl[k].widths) for k in PWL_SUBSYSTEMS})
        b = LPBuilder(L.n_x)
        dt = 3600.0
        econ = self.economics
        battery = model.battery
        E = battery.nominal_energy_J
        n_tot = model.n_total_mol
        w_comp = model.gas.p.co2_compressor_kJ_per_mol * 1e3
        recovery = float(model.water.p.condensate_recovery)
        pv = rows["pv_available_W"]

        # ---- columns
        lo_z, hi_z = model.bounds(z0)
        for k in range(n + 1):
            b.bounds(np.arange(L.z(k, 0), L.z(k, 0) + N_Z), lo_z, hi_z)
        for k in range(n):
            u_lo = np.zeros(L.n_u)
            u_hi = np.zeros(L.n_u)
            u_hi[L.ie:L.ie + len(SUBSYSTEMS)] = 1.0
            for key in PWL_SUBSYSTEMS:
                a, _ = L.iseg[key]
                u_hi[a:a + len(pwl[key].widths)] = pwl[key].widths
            u_hi[L.ical_p] = kiln.rated_W
            u_hi[L.ical_r] = kiln.rate_max_mol_s
            u_hi[L.ical_y] = 1.0
            u_hi[L.istart:L.istart + len(SUBSYSTEMS)] = 1.0
            u_hi[L.ipc] = u_hi[L.ipd] = battery.max_power_W
            u_hi[L.igam] = 1.0
            u_hi[L.ivh] = u_hi[L.ivc] = 1.0
            u_hi[L.imw] = 0.02
            cols = np.arange(L.u(k, 0), L.u(k, 0) + L.n_u)
            b.bounds(cols, u_lo, u_hi)

        def rate(k, key):
            """Material rate, as (column, coefficient) pairs."""
            if key == "calciner":
                return [(L.u(k, L.ical_r), 1.0)]
            sl = pwl[key].slopes
            return [(c, float(sl[m])) for m, c in enumerate(L.seg(k, key))]

        def power(k, key):
            """Electrical draw, as (column, coefficient) pairs."""
            if key == "calciner":
                return [(L.u(k, L.ical_p), 1.0)]
            idle = float(pwl[key].idle_W)
            ent = [(L.e(k, key), idle)]
            if pwl[key].is_power:
                ent += [(c, 1.0) for c in L.seg(k, key)]
            return ent

        # ---- rows
        b.block("initial_state")
        for i in range(N_Z):
            b.equality([(L.z(0, i), 1.0)], float(z0[i]))

        b.block("dynamics")
        for k in range(n):
            b.equality(
                [(L.z(k + 1, 0), 1.0),
                 (L.z(k, 0), -1.0 + dt * battery.p.self_discharge_per_day / 86400.0),
                 (L.u(k, L.ipc), -dt * battery.p.eta_charge / E),
                 (L.u(k, L.ipd), dt / (battery.p.eta_discharge * E))], 0.0)
            b.equality(
                [(L.z(k + 1, 1), 1.0), (L.z(k, 1), -1.0)]
                + [(c, -dt * v) for c, v in rate(k, "contactor")]
                + [(c, dt * v) for c, v in rate(k, "calciner")], 0.0)
            b.equality(
                [(L.z(k + 1, 2), 1.0), (L.z(k, 2), -1.0)]
                + [(c, -dt * v / n_tot) for c, v in rate(k, "calciner")], 0.0)
            b.equality(
                [(L.z(k + 1, 3), 1.0), (L.z(k, 3), -1.0), (L.u(k, L.ivh), dt)]
                + [(c, -dt * v) for c, v in rate(k, "electrolyser")]
                + [(c, 4.0 * dt * v) for c, v in rate(k, "sabatier")], 0.0)
            b.equality(
                [(L.z(k + 1, 4), 1.0), (L.z(k, 4), -1.0), (L.u(k, L.ivc), dt)]
                + [(c, -dt * v) for c, v in rate(k, "calciner")]
                + [(c, dt * v) for c, v in rate(k, "sabatier")], 0.0)
            b.equality(
                [(L.z(k + 1, 5), 1.0), (L.z(k, 5), -1.0), (L.u(k, L.imw), -dt)]
                + [(c, -2.0 * dt * recovery * v * M_H2O) for c, v in rate(k, "sabatier")]
                + [(c, dt * v * M_H2O) for c, v in rate(k, "electrolyser")], 0.0)
            # kiln energy balance -- exactly linear. The one approximation is the
            # feed's sensible heat, evaluated at the operating temperature
            # rather than at T_k: a few per cent of the duty, during warm-up.
            t_amb = float(rows["temp_air"][k]) + 273.15
            b.equality(
                [(L.z(k + 1, 6), 1.0),
                 (L.z(k, 6), -1.0 + dt * kiln.ua_W_K / kiln.capacity_J_K),
                 (L.u(k, L.ical_p), -dt * kiln.eta / kiln.capacity_J_K),
                 (L.u(k, L.ical_r), dt * kiln.per_mol_J / kiln.capacity_J_K)],
                dt * kiln.ua_W_K * t_amb / kiln.capacity_J_K)

        # The dual of this block is lambda. Written demand-minus-supply with the
        # PV term moved to the right-hand side, so relaxing it means energy
        # arriving from nowhere and the dual is a positive price.
        b.block(BUS_BALANCE)
        for k in range(n):
            ent: list[tuple[int, float]] = []
            for key in SUBSYSTEMS:
                ent += power(k, key)
            ent += [(c, w_comp * v) for c, v in rate(k, "calciner")]
            ent += [(L.u(k, L.ipc), 1.0), (L.u(k, L.ipd), -1.0),
                    (L.u(k, L.igam), float(pv[k]))]
            b.equality(ent, float(pv[k]))

        # A segment can only be used by a committed machine, and only up to its
        # own width. These two rows together *are* the dispatch band.
        b.block("segment_cap")
        for k in range(n):
            for key in PWL_SUBSYSTEMS:
                width = pwl[key].widths
                for m, col in enumerate(L.seg(k, key)):
                    b.row([(col, 1.0), (L.e(k, key), -float(width[m]))], -np.inf, 0.0)

        # --- the kiln's band, and why it needs a temperature state.
        #
        # Heater power sits between standby and rated when committed. The rate
        # is capped by the feeder and by how hot the kiln is -- from cold it
        # takes 5.6 h at full heater before calcination is possible at all.
        #
        # That second cap cannot be written directly: `r <= slope (T - onset)`
        # rearranges to `T >= onset + r/slope`, demanding a permanently hot kiln
        # at `r = 0`, and the set it describes, `0 <= r <= max(0, slope
        # (T - onset))`, is genuinely non-convex. Hence an indicator `y`:
        #
        #     r <= r_max * y                       nothing produced unless hot
        #     r <= slope (T - onset) + M (1 - y)   the cap, released when y = 0
        #     y <= e                               and only when energised
        #
        # with `M = slope (onset - T_min)`, the smallest value letting a cold
        # kiln sit feasibly at `r = 0`.
        big_m = kiln.slope_mol_s_K * (kiln.onset_K - 250.0)
        b.block("kiln_band")
        for k in range(n):
            e = L.e(k, "calciner")
            b.row([(L.u(k, L.ical_p), 1.0), (e, -kiln.rated_W)], -np.inf, 0.0)
            b.row([(L.u(k, L.ical_p), 1.0), (e, -kiln.standby_W)], 0.0, np.inf)
            b.row([(L.u(k, L.ical_r), 1.0),
                   (L.u(k, L.ical_y), -kiln.rate_max_mol_s)], -np.inf, 0.0)
            b.row([(L.u(k, L.ical_r), 1.0), (L.z(k, 6), -kiln.slope_mol_s_K),
                   (L.u(k, L.ical_y), big_m)],
                  -np.inf, big_m - kiln.slope_mol_s_K * kiln.onset_K)
            b.row([(L.u(k, L.ical_y), 1.0), (e, -1.0)], -np.inf, 0.0)
            # The only integer columns in the problem, and they have to be.
            # Maximising the relaxation over y in [0,1] gives
            #     r* = r_max (slope (T - onset) + M) / (r_max + M),
            # which at 841 K permits 0.26 mol/s where the real kiln calcines
            # nothing, and the convex hull is no tighter because the
            # non-convexity is genuine. Relaxed, the layer schedules calcination
            # on a cold kiln, the sorbent saturates, and production collapses to
            # 17 kg/day. Only the near term needs it -- see
            # `binary_horizon_hours`.
            if k < self.binary_horizon_hours:
                b.integer(L.u(k, L.ical_y))

        # --- reactor availability: the taper the plant applies near a floor.
        #
        # The plant throttles the Sabatier as either buffer runs down,
        #     availability = min( clip(co2_free/50), clip(h2_free/200) ),
        # and without an equivalent the LP assumes full conversion right up to
        # the box floor. The reactor binds in every run, so this is the largest
        # single reason the plan out-predicts the plant.
        #
        # No indicator needed here: the taper reaches zero exactly *at* the box
        # floor, so `r_sab = 0` is feasible everywhere and the two rows are an
        # honest linear envelope of the min(). (The kiln's onset sits well above
        # its temperature floor, which is what made that set non-convex.)
        b.block("reactor_availability")
        gas = model.gas.p
        co2_floor = float(gas.co2_min_fraction * gas.co2_capacity_mol)
        h2_floor = float(gas.h2_min_fraction * gas.h2_capacity_mol)
        r_sab_max = float(pwl["sabatier"].rate_at_full())
        for k in range(n):
            for col, v in rate(k, "sabatier"):
                # r_sab <= r_max * (n_co2 - floor)/50
                b.row([(col, v), (L.z(k, zi("n_co2")), -r_sab_max / 50.0)],
                      -np.inf, -r_sab_max * co2_floor / 50.0)
                # r_sab <= r_max * (n_h2 - floor)/200
                b.row([(col, v), (L.z(k, zi("n_h2")), -r_sab_max / 200.0)],
                      -np.inf, -r_sab_max * h2_floor / 200.0)

        b.block("min_load")
        d_min = model.electrolyser_min_dispatch(pwl)
        for k in range(n):
            b.row([(c, 1.0) for c in L.seg(k, "electrolyser")]
                  + [(L.e(k, "electrolyser"), -d_min)], 0.0, np.inf)

        # A start is an *increase* in commitment, so interval 0 is measured
        # against what is already running, not against zero. Otherwise
        # `s_0 >= e_0` charges a fresh start-up for every running machine at
        # every replan -- EUR 80 every three hours for the kiln and reactor
        # alone -- and as the horizon shortens the layer shuts the plant down to
        # avoid a cost it was never owed. Measured: 19.2 kg/day.
        b.block("start_counter")
        for k in range(n):
            for key in SUBSYSTEMS:
                ent = [(L.start(k, key), -1.0), (L.e(k, key), 1.0)]
                if k > 0:
                    ent.append((L.e(k - 1, key), -1.0))
                    b.row(ent, -np.inf, 0.0)
                else:
                    b.row(ent, -np.inf, float(self._prev_commit.get(key, 0.0)))

        b.block("min_uptime")
        for k in range(n):
            for key in SUBSYSTEMS:
                tau = MIN_UPTIME[key]
                if tau > 1 and k + tau <= n:
                    b.row([(L.start(k, key), float(tau))]
                          + [(L.e(j, key), -1.0) for j in range(k, k + tau)],
                          -np.inf, 0.0)

        # ---- objective: minimise -(profit)
        price = econ.p.methane_price_per_kg
        c_batt = battery.cost_per_efc_EUR()
        # Cost per *cycle* of the whole inventory, to pair with dN = r/n_tot.
        # The solids model reports it per mole calcined; multiplying back by
        # n_tot keeps one definition in one place.
        c_sorb = n_tot * float(model.solids.marginal_deactivation_cost_per_mol(
            np.array([0.0, float(z0[zi("n_caco3")]), float(z0[zi("cycle_number")])]),
            float(model.solids.p.sorbent_cost_per_mol)))
        for k in range(n):
            for col, v in rate(k, "sabatier"):
                b.cost(col, -dt * price * v * M_CH4)
            for col, v in rate(k, "calciner"):
                b.cost(col, dt * c_sorb * v / n_tot)
            b.cost(L.u(k, L.ipc), dt * c_batt / (2.0 * E))
            b.cost(L.u(k, L.ipd), dt * c_batt / (2.0 * E))
            b.cost(L.u(k, L.imw), dt * econ.p.water_cost_per_m3 / 1000.0)
            for key in SUBSYSTEMS:
                b.cost(L.start(k, key), START_COST_EUR[key])

        # terminal value: leftover inventory priced at the methane it becomes
        f = self.terminal_value_fraction
        per_mol = price * M_CH4 * f
        b.cost(L.z(n, 1), -per_mol)
        b.cost(L.z(n, 3), -per_mol / 4.0)
        b.cost(L.z(n, 4), -per_mol)
        b.cost(L.z(n, 0), -per_mol * (E / 3.6e6) / 56.4 / 2.016e-3 / 4.0)

        return b.build(), L

    # --- reading the solution back ----------------------------------------
    def _unpack(self, t, n, solution, L, pwl, kiln, problem) -> DispatchPlan:
        x = solution.x
        states = x[:N_Z * (n + 1)].reshape(n + 1, N_Z)

        bands: list[dict[str, Band]] = []
        for k in range(n):
            row: dict[str, Band] = {}
            for key in SUBSYSTEMS:
                e = float(x[L.e(k, key)])
                if key == "calciner":
                    d = float(x[L.u(k, L.ical_p)])
                    setpoint = kiln.setpoint_for(d)
                else:
                    d = float(sum(x[c] for c in L.seg(k, key)))
                    setpoint = pwl[key].setpoint_for(d)

                # Commit whenever the plan intends output, not only when the
                # relaxed enable clears 0.5. `delta <= width * e` ties dispatch
                # to the enable, so the LP can settle at e = 0.45 with the
                # machine genuinely working at 45 % of span, and rounding that
                # down discards the dispatch with it (measured: twelve
                # contactor intervals at 38-42 % flow, all silently lost).
                # Rounding up costs only the idle draw.
                committed = e >= self.commit_threshold or setpoint > 1e-4
                row[key] = Band(committed=committed, setpoint_max=setpoint,
                                dispatch=d)
            bands.append(row)

        # One extra watt on the bus held for an hour is dt/3.6e6 kWh.
        lam = solution.dual(BUS_BALANCE) * 3.6e6 / 3600.0

        battery = np.array([[max(x[L.u(k, L.ipc)], 0.0), max(x[L.u(k, L.ipd)], 0.0)]
                            for k in range(n)], dtype=float)
        curtail = np.array([np.clip(x[L.u(k, L.igam)], 0.0, 1.0) for k in range(n)])

        # Per-interval profit, straight off the linear objective.
        c = problem.c
        stage = np.array([
            -float(np.dot(c[L.u(k, 0):L.u(k, 0) + L.n_u],
                          x[L.u(k, 0):L.u(k, 0) + L.n_u]))
            for k in range(n)
        ], dtype=float)

        return DispatchPlan(
            t0_s=t, dt_s=3600.0, states=states, bands=bands,
            lambda_EUR_per_kWh=lam, objective_EUR=-solution.f,
            horizon_s=float(n * 3600.0),
            battery_W=battery, curtail=curtail, stage_profit_EUR=stage,
        )

    # --- horizon data -----------------------------------------------------
    def _forecast_rows(self, t: float, forecast) -> dict[str, np.ndarray]:
        series = forecast if forecast is not None else self.context.forecast
        dense = series.densify(3600.0, None)
        offsets = dense["time_s"].to_numpy()
        start = max(int(np.searchsorted(offsets, t, side="right")) - 1, 0)
        window = dense.iloc[start:min(start + self.horizon_hours, len(dense))]
        pv = self.model.pv
        return {
            "pv_available_W": np.array(
                [float(pv.available_power(r)) for _, r in window.iterrows()]),
            "temp_air": window["temp_air"].to_numpy(dtype=float),
            "relative_humidity": window["relative_humidity"].to_numpy(dtype=float),
            "wind_speed": window["wind_speed"].to_numpy(dtype=float),
            "pressure": window["pressure"].to_numpy(dtype=float),
        }

    # --- reporting --------------------------------------------------------
    _battery: tuple[float, float] = (0.0, 0.0)
    _curtail: float = 0.0

    def diagnostics(self) -> dict[str, float]:
        out = dict(self._diagnostics)
        if self.plan is not None:
            out["plan_lambda_EUR_per_kWh"] = self.plan.price_at(self._last_t)
        return out
