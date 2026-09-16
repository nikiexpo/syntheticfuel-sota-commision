"""The dispatch layer's model: six buffers and four piecewise-linear machines.

This is the outer layer of the proposed architecture (`bookkeeping/07`), and it
is deliberately *less* faithful than `planner_model.py`. The argument for giving
up that fidelity is in the tex; the short version is three measured facts:

1. The outer layer's setpoint schedule is discarded -- only the hand-off reaches
   the plant -- so high-fidelity dynamics buy a schedule nobody runs.
2. The outer objective was already effectively linear. The reactor's conversion
   is flat to 0.25 % across its whole feed range, and every other term is exactly
   linear, so the NLP was paying nonlinear prices for a linear objective.
3. A 240 h horizon is unreachable as an NLP and takes 0.34 s as an LP, which is
   what makes a scenario tree possible at all.

What survives the simplification, because each carries a result worth keeping:

* **The buffers.** Non-negotiable -- co-scheduling them is the whole thesis. They
  are integrators, so they are free.
* **The electrolyser's part-load efficiency.** Kept as a piecewise-linear curve
  of falling marginal yield, which is what makes "run flat out when sunny"
  suboptimal. Measured slopes fall 2.85e-6 -> 2.32e-6 mol/J.
* **Sorbent deactivation.** `X_N` is frozen at the cycle number at the start of
  the horizon and updated between replans. `N` moves by about 2 per week, so the
  error over three days is negligible and the capacity ceiling is still real.

Exactness of the piecewise-linear relaxation
--------------------------------------------
Each machine's dispatch is split into segments of *decreasing* marginal yield.
Under maximisation the segments then fill in order on their own and **no segment
binaries are needed**. That holds only while each rate-versus-dispatch map is
concave, which is checked at construction -- `PWLMap` raises if it is not, rather
than silently returning a relaxation that is not tight.

The calciner's idle draw
------------------------
Set to the power needed to *hold* the kiln at temperature, `UA dT / eta`, rather
than the flat 3 kW standby in the YAML. This is both more physical -- a kiln that
is committed is a kiln being kept hot -- and what makes the rate map linear above
idle instead of convex-then-linear, which would have broken the exactness
argument above. It is a change from the existing model and it makes commitment
meaningfully more expensive: 16.6 kW against 3 kW at the reference sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from sfp.units import DH_CALCINATION, F_FARADAY, M_CACO3

#: States, in order. The kiln temperature is kept, which is a correction to the
#: proposal in `bookkeeping/07`.
#:
#: That document dropped it and replaced what it was *for* -- making start-up
#: expensive -- with a start cost and a minimum up-time. Measured, that is not
#: enough: the kiln needs 5.6 hours at full heater to climb from ambient to
#: 1200 K, and a dispatch layer that believes a committed kiln calcines
#: immediately schedules calcination that cannot happen. The sorbent then
#: saturates, the contactor stops, and the plant makes 17 kg/day.
#:
#: Keeping it costs nothing, because **the kiln's energy balance is linear**:
#:
#:     C dT/dt = eta P_heat - UA (T - T_amb) - (dH + m cp dT_feed) r_calc
#:
#: Every term is linear in a decision variable. The only genuinely nonlinear
#: part was ever the *rate* as a function of temperature -- Baker equilibrium
#: times Arrhenius -- and that enters here as a linear upper bound on the rate
#: instead of as a factor in it. See `CalcinerThermal`.
STATE_NAMES: tuple[str, ...] = (
    "soc", "n_caco3", "cycle_number", "n_h2", "n_co2", "water_kg",
    "kiln_temperature_K",
)

#: Machines the dispatch layer commits, in a fixed order.
SUBSYSTEMS: tuple[str, ...] = ("contactor", "calciner", "electrolyser", "sabatier")

N_Z = len(STATE_NAMES)
_IZ = {name: i for i, name in enumerate(STATE_NAMES)}


def zi(name: str) -> int:
    return _IZ[name]


@dataclass(frozen=True)
class PWLMap:
    """A machine's rate as a piecewise-linear function of its dispatch.

    `dispatch` is electrical power in watts for the three power-elastic machines
    and CO2 feed rate in mol/s for the reactor, whose draw is flat in feed and
    therefore carries no information about it.
    """

    name: str
    idle_W: float               # draw when committed and at zero dispatch
    widths: np.ndarray          # segment widths, in dispatch units
    slopes: np.ndarray          # rate per unit dispatch, strictly decreasing
    is_power: bool              # dispatch is watts (True) or mol/s (False)
    setpoint_at: np.ndarray     # setpoint corresponding to each segment edge
    dispatch_at: np.ndarray     # dispatch at each segment edge

    def __post_init__(self) -> None:
        if np.any(np.diff(self.slopes) > 1e-15):
            raise ValueError(
                f"{self.name}: rate-versus-dispatch map is not concave "
                f"(slopes {self.slopes}). The piecewise-linear relaxation is "
                "only exact for a concave map; a convex region needs segment "
                "binaries, which this formulation does not carry."
            )

    @property
    def span(self) -> float:
        return float(np.sum(self.widths))

    def rate_at_full(self) -> float:
        return float(np.dot(self.widths, self.slopes))

    def setpoint_for(self, dispatch: float) -> float:
        """Invert the map: the setpoint that commands this dispatch.

        Monotone by construction, so a linear interpolation on the sampled edges
        is exact at the edges and close between them. This is the only place the
        LP's world is translated back into something the plant understands.
        """
        d = float(np.clip(dispatch, self.dispatch_at[0], self.dispatch_at[-1]))
        return float(np.interp(d, self.dispatch_at, self.setpoint_at))


@dataclass(frozen=True)
class CalcinerThermal:
    """The kiln as a linear thermal system with a temperature-limited rate.

    The calciner is not a piecewise-linear machine like the others. Its heater
    power and its calcination rate are *separate* decisions coupled through a
    temperature state, which is what lets the dispatch layer understand that
    heating a cold kiln costs energy for hours before it yields anything.

        heater power   P in [standby * e, rated * e]
        rate           r >= 0,  r <= r_max * e,  r <= slope * (T - T_onset)
        temperature    C dT/dt = eta P - UA (T - T_amb) - per_mol * r

    The rate cap is the one approximation. The true rate against temperature is
    the Baker driving force times an Arrhenius factor: flat at zero below about
    1100 K, rising steeply, saturating near `r_max`. The chord from the point
    where it first becomes non-negligible to the point where it saturates is a
    two-piece linear upper bound. Between those points it is mildly optimistic;
    at the ends it is exact, and it gets the property that matters right -- a
    cold kiln cannot calcine at any heater setting.
    """

    standby_W: float
    rated_W: float
    eta: float
    capacity_J_K: float
    ua_W_K: float
    per_mol_J: float             # reaction enthalpy plus feed sensible heat
    rate_max_mol_s: float
    onset_K: float
    slope_mol_s_K: float
    operating_K: float

    def setpoint_for(self, power_W: float) -> float:
        return float(np.clip(power_W / self.rated_W, 0.0, 1.0))


def _concave_segments(dispatch, rate, n_seg):
    """Split a sampled monotone curve into `n_seg` segments of falling slope.

    Segment *edges* are placed on equal dispatch intervals rather than by fitting,
    because the point is a faithful envelope rather than a least-squares one: the
    chord between two points of a concave curve lies below it, so every segment
    under-promises and the LP cannot claim a rate the plant will not deliver.
    """
    edges = np.linspace(dispatch[0], dispatch[-1], n_seg + 1)
    r_at = np.interp(edges, dispatch, rate)
    widths = np.diff(edges)
    slopes = np.diff(r_at) / np.maximum(widths, 1e-12)
    return edges, widths, slopes


class DispatchModel:
    """Linearised plant, rebuilt at each replan around current conditions.

    The piecewise-linear maps depend on ambient temperature, humidity and the
    sorbent's loading, so they are derived *per replan* from the conditions the
    forecast expects rather than once at construction. That makes this a
    successive-linearisation scheme rather than a fixed approximation, which is
    what keeps a linear model honest about a nonlinear plant.
    """

    #: segments per machine. Three is enough to carry the electrolyser's
    #: part-load curvature; the reactor is exactly linear so one suffices.
    SEGMENTS: Mapping[str, int] = {
        "contactor": 3, "electrolyser": 3, "sabatier": 1,
    }

    #: Machines described by a piecewise-linear map. The calciner is not one of
    #: them; it is a thermal system, handled by `thermal()`.
    PWL_SUBSYSTEMS: tuple[str, ...] = ("contactor", "electrolyser", "sabatier")

    def __init__(self, plant: Any) -> None:
        self.plant = plant
        self.pv = plant["pv"]
        self.battery = plant["battery"]
        self.solids = plant["solids"]
        self.gas = plant["gas"]
        self.water = plant["water"]

    # --- capacities -------------------------------------------------------
    @property
    def n_total_mol(self) -> float:
        return float(self.solids.p.n_total_mol)

    def power_scale(self) -> float:
        return float(max(self.pv.rated_ac_W, 1e5))

    def initial_state(self, plant_state: Mapping[str, np.ndarray]) -> np.ndarray:
        """Project the full plant state onto the six buffers."""
        return np.array([
            float(plant_state["battery"][0]),
            float(plant_state["solids"][1]),
            float(plant_state["solids"][2]),
            float(plant_state["gas"][0]),
            float(plant_state["gas"][1]),
            float(plant_state["water"][0]),
            float(plant_state["calciner"][0]),
        ], dtype=float)

    def bounds(self, z0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Box bounds on the buffers.

        The carbonate ceiling is `n_tot * X_N(N0)`, not `n_tot`: deactivation
        means capacity only ever falls, and the nominal figure overstates it
        roughly threefold. Because `X_N` is frozen for the solve, this ceiling is
        a plain box rather than the nonlinear row it was in the NLP planner.
        """
        b, gas, water = self.battery.p, self.gas.p, self.water.p
        n0 = float(z0[zi("cycle_number")])
        cal = self.plant["calciner"].p
        lo = np.array([
            b.soc_min, 0.0, max(n0 - 0.1, 0.0),
            gas.h2_min_fraction * gas.h2_capacity_mol,
            gas.co2_min_fraction * gas.co2_capacity_mol,
            water.water_min_fraction * water.water_capacity_kg,
            250.0,
        ])
        hi = np.array([
            b.soc_max,
            self.n_total_mol * float(self.solids.max_conversion(n0)),
            n0 + 10.0,
            float(gas.h2_capacity_mol),
            float(gas.co2_capacity_mol),
            float(water.water_capacity_kg),
            float(cal.temperature_max_K),
        ])
        return np.minimum(lo, z0), np.maximum(hi, z0)

    # --- the linearisation ------------------------------------------------
    def pwl(self, weather: Mapping[str, float], loading: float) -> dict[str, PWLMap]:
        """Derive the piecewise-linear machines' rate maps at these conditions.

        The calciner is absent: it is a thermal system, not a static map, and
        `thermal()` describes it instead.
        """
        return {
            "contactor": self._contactor(weather, loading),
            "electrolyser": self._electrolyser(weather),
            "sabatier": self._sabatier(weather),
        }

    def thermal(self) -> CalcinerThermal:
        """Linear kiln parameters, with the rate cap fitted to the real curve."""
        cal = self.plant["calciner"]
        t_star = float(cal.p.temperature_max_K) - 50.0
        r_max = float(cal.p.calcination_rate_max_mol_s)

        # Sample the plant's own rate-versus-temperature curve at full feedstock
        # and fit the two points the linear cap runs between.
        temps = np.linspace(900.0, float(cal.p.temperature_max_K), 200)
        rate = np.array([
            r_max
            * float(cal.driving_force(T))
            * float(np.clip(cal.kinetic_factor(T), 0.0, 1.0))
            * float(cal.feedstock_factor(1e6))
            for T in temps
        ])
        hot = rate >= 0.99 * rate.max() if rate.max() > 0 else np.zeros_like(rate, bool)
        onset = float(temps[np.argmax(rate >= 0.02 * max(rate.max(), 1e-12))])
        saturate = float(temps[np.argmax(hot)]) if hot.any() else t_star
        slope = r_max / max(saturate - onset, 1.0)

        per_mol = DH_CALCINATION + M_CACO3 * cal.p.solids_heat_capacity_J_kg_K * (
            t_star - cal.p.solids_feed_temperature_K)
        return CalcinerThermal(
            standby_W=float(cal.standby_power_W),
            rated_W=float(cal.heater_power_rated_W),
            eta=float(cal.p.heater_efficiency),
            capacity_J_K=float(cal.p.thermal_capacity_J_K),
            ua_W_K=float(cal.p.heat_loss_UA_W_K),
            per_mol_J=float(per_mol),
            rate_max_mol_s=r_max,
            onset_K=onset,
            slope_mol_s_K=float(slope),
            operating_K=t_star,
        )

    def _sample(self, n: int = 40) -> np.ndarray:
        return np.linspace(0.0, 1.0, n)

    def _contactor(self, w, loading) -> PWLMap:
        con = self.plant["contactor"]
        s = self._sample()
        x = con.initial_state()
        P = np.array([con.power_for_setpoint(x, v, 1.0, w) for v in s])
        wc = dict(w, solids_loading=float(np.clip(loading, 0.0, 0.999)))
        r = np.array([float(con.capture_rate_mol_s(np.array([v, 1.0]), wc)) for v in s])
        idle = float(P[0])
        edges, widths, slopes = _concave_segments(
            P - idle, r - r[0], self.SEGMENTS["contactor"])
        return PWLMap("contactor", idle, widths, slopes, True,
                      np.interp(edges, P - idle, s), edges)

    def _electrolyser(self, w) -> PWLMap:
        ele = self.plant["electrolyser"]
        s = self._sample()
        x = np.array([ele.p.temperature_setpoint_K, 0.0])
        P = np.array([ele.power_for_setpoint(x, v, 1.0, w) for v in s])
        i = s * ele.p.current_density_max_A_cm2 * ele.p.active_area_cm2
        r = ele.p.faraday_efficiency * ele.p.n_cells * i / (2.0 * F_FARADAY)
        idle = float(P[0])
        edges, widths, slopes = _concave_segments(
            P - idle, r - r[0], self.SEGMENTS["electrolyser"])
        return PWLMap("electrolyser", idle, widths, slopes, True,
                      np.interp(edges, P - idle, s), edges)

    def _sabatier(self, w) -> PWLMap:
        """Dispatch is the feed, not the power.

        The reactor's electrical draw is flat in feed rate -- preheat and
        auxiliaries -- so a power band would say nothing about how hard it is
        running. This is the one machine whose dispatch variable is a material
        flow, and it is why the interface is a `dispatch` band rather than a
        power band.
        """
        sab = self.plant["sabatier"]
        x = np.array([sab.p.temperature_target_K, 1.0, 0.0])
        idle = float(sab.power_for_setpoint(x, 0.0, 1.0, w))
        feed_max = float(sab.p.co2_feed_max_mol_s)
        conv = float(sab.conversion(x, np.array([0.7, 1.0])))
        return PWLMap("sabatier", idle, np.array([feed_max]), np.array([conv]),
                      False, np.array([0.0, 1.0]), np.array([0.0, feed_max]))

    # --- electrolyser minimum load ---------------------------------------
    def electrolyser_min_dispatch(self, pwl: Mapping[str, PWLMap]) -> float:
        """Dispatch floor above idle set by the gas-crossover limit.

        Below the minimum current density hydrogen crosses into the oxygen
        stream, which is a safety limit rather than a preference. Expressed in
        dispatch units so the LP can carry it as a simple row.
        """
        ele = self.plant["electrolyser"]
        frac = float(ele.p.current_density_min_fraction)
        m = pwl["electrolyser"]
        return float(np.interp(frac, m.setpoint_at, m.dispatch_at))
