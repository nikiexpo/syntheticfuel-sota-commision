"""Economic dispatch (LP/MILP) tests.

Four things are pinned here, each because it failed once and the failure was
silent -- the layer kept producing plausible-looking schedules while the plant
quietly under-produced.

**Concavity of the piecewise-linear maps.** The whole relaxation is exact only
while every rate-versus-dispatch map has falling marginal yield. If one ever
becomes convex the LP will over-promise and nothing will raise, so `PWLMap`
checks it at construction and this suite checks that the check works.

**A cold kiln cannot calcine.** The rate cap is the one genuinely non-convex
part of the formulation, and a relaxed indicator permits 0.26 mol/s at 841 K
where the real kiln makes nothing. That reproduced exactly the failure the cap
was added to prevent: the sorbent saturated and production fell to 17 kg/day.

**A running machine is not restarted.** The start counter measures an *increase*
in commitment, so interval 0 has to be compared against what is already running.
Compared against zero instead, every replan charged a fresh start-up -- EUR 80
every three hours -- and the layer shut the plant down to avoid it as the
horizon shortened.

**Lambda is a price.** Same property the NLP planner is checked for, and for the
same reason: sign and magnitude are both easy to get wrong in ways that raise
nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from sfp.control.base import ControlContext
from sfp.control.dispatch import (
    PWL_SUBSYSTEMS,
    Band,
    DispatchPlan,
    EconomicDispatch,
)
from sfp.control.dispatch_model import (
    N_Z,
    STATE_NAMES,
    SUBSYSTEMS,
    DispatchModel,
    PWLMap,
    zi,
)
from sfp.economics import Economics

NOMINAL = {"temp_air": 20.0, "pressure": 101325.0,
           "relative_humidity": 0.5, "wind_speed": 2.0}


@pytest.fixture(scope="module")
def plant():
    from sfp.cli import build_reference_plant
    return build_reference_plant()


@pytest.fixture(scope="module")
def model(plant):
    return DispatchModel(plant)


@pytest.fixture(scope="module")
def maps(model):
    return model.pwl(NOMINAL, loading=0.3)


def _state(plant, *, kiln_K=293.15, n_co2=3000.0, n_h2=15000.0,
           n_caco3=8000.0, soc=0.5):
    return {
        "battery": np.array([soc, 0.0, 0.0]),
        "solids": np.array([50000.0 - n_caco3, n_caco3, 2.0]),
        "gas": np.array([n_h2, n_co2]),
        "water": np.array([2500.0, 0.0]),
        "calciner": np.array([kiln_K]),
    }


def _controller(plant, weather, **kwargs):
    c = EconomicDispatch(**kwargs)
    c.reset(ControlContext(plant=plant, site=None, dt_s=300.0,
                           forecast=weather, economics=Economics()))
    return c


# --- the linearisation ----------------------------------------------------
def test_every_pwl_map_is_concave(maps):
    """The exactness of the relaxation rests entirely on this."""
    for name, m in maps.items():
        assert np.all(np.diff(m.slopes) <= 1e-15), (
            f"{name} has rising marginal yield {m.slopes}; the piecewise-linear "
            "relaxation is not exact and the LP will over-promise"
        )


def test_pwl_map_rejects_a_convex_map():
    """The guard must actually fire -- a silent convex map is the failure mode."""
    with pytest.raises(ValueError, match="not concave"):
        PWLMap("bad", 0.0, np.array([1.0, 1.0]), np.array([1.0, 2.0]),
               True, np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0, 2.0]))


def test_electrolyser_marginal_yield_falls(maps):
    """Part-load efficiency is a headline result; it must survive linearisation."""
    slopes = maps["electrolyser"].slopes
    assert slopes[0] > slopes[-1] * 1.05, (
        "the electrolyser's falling hydrogen-per-watt is what makes 'run flat "
        "out when sunny' suboptimal; a flat map throws that result away"
    )


def test_setpoint_inversion_round_trips(maps):
    for name, m in maps.items():
        for frac in (0.0, 0.25, 0.5, 1.0):
            d = m.span * frac
            s = m.setpoint_for(d)
            assert 0.0 <= s <= 1.0
        assert m.setpoint_for(0.0) <= m.setpoint_for(m.span)


def test_sabatier_dispatch_is_feed_not_power(maps):
    """Its draw is flat in feed, so a power band would carry no information."""
    assert maps["sabatier"].is_power is False
    for name in ("contactor", "electrolyser"):
        assert maps[name].is_power is True


def test_kiln_rate_cap_is_zero_below_onset(model):
    kiln = model.thermal()
    assert kiln.onset_K > 1000.0
    assert kiln.slope_mol_s_K > 0.0
    # the cap the LP uses, evaluated cold, must not permit production
    assert kiln.slope_mol_s_K * (293.15 - kiln.onset_K) < 0.0


def test_state_vector_carries_the_kiln(model, plant):
    assert "kiln_temperature_K" in STATE_NAMES
    z = model.initial_state(_state(plant, kiln_K=1100.0))
    assert z[zi("kiln_temperature_K")] == pytest.approx(1100.0)
    assert len(z) == N_Z


# --- the LP ---------------------------------------------------------------
def test_plan_solves_and_covers_the_horizon(reference_plant, synthetic_weather):
    c = _controller(reference_plant, synthetic_weather, horizon_hours=24)
    c.act(0.0, _state(reference_plant), {}, synthetic_weather)
    assert c.plan is not None, "the dispatch LP did not produce a plan"
    assert len(c.plan.bands) > 0
    assert c.plan.states.shape == (len(c.plan.bands) + 1, N_Z)
    assert set(c.plan.bands[0]) == set(SUBSYSTEMS)


def test_a_cold_kiln_is_not_scheduled_to_calcine(reference_plant, synthetic_weather):
    """The binary indicator's whole job.

    Relaxed, this permits r = r_max (slope (T - onset) + M)/(r_max + M), which is
    0.26 mol/s at 841 K. The plan then banks CO2 that never arrives, the sorbent
    saturates, and the contactor stops.
    """
    c = _controller(reference_plant, synthetic_weather, horizon_hours=12)
    c.act(0.0, _state(reference_plant, kiln_K=293.15), {}, synthetic_weather)
    assert c.plan is not None

    kiln = DispatchModel(reference_plant).thermal()
    temps = c.plan.states[:-1, zi("kiln_temperature_K")]
    # CO2 can only come from calcination, so the tank must not fill while cold
    co2 = c.plan.states[:, zi("n_co2")]
    cold = temps < kiln.onset_K - 50.0
    if cold.any():
        rise = np.diff(co2)[cold]
        assert np.all(rise <= 1.0), (
            f"CO2 inventory rose by up to {rise.max():.1f} mol while the kiln "
            f"was below {kiln.onset_K - 50:.0f} K -- the rate cap is leaking"
        )


def test_a_running_machine_is_not_charged_a_restart(reference_plant, synthetic_weather):
    """Regression: interval 0's start counter must see the current commitment.

    Without this every replan billed a fresh start-up for machines that were
    already running. It stayed invisible while the horizon was long enough to
    repay the charge, then shut the plant down as the horizon shortened.
    """
    state = _state(reference_plant, kiln_K=1200.0, n_co2=4000.0, n_h2=20000.0)

    cold = _controller(reference_plant, synthetic_weather, horizon_hours=12)
    cold.act(0.0, state, {}, synthetic_weather)
    assert cold.plan is not None
    objective_cold = cold.plan.objective_EUR

    warm = _controller(reference_plant, synthetic_weather, horizon_hours=12)
    warm._prev_commit = {key: 1.0 for key in SUBSYSTEMS}
    warm.act(0.0, state, {}, synthetic_weather)
    assert warm.plan is not None

    assert warm.plan.objective_EUR >= objective_cold - 1e-6, (
        "telling the layer its machines are already running made the plan worse; "
        "interval 0 is charging a start-up it does not owe"
    )


def test_lambda_is_a_positive_price(reference_plant, synthetic_weather):
    c = _controller(reference_plant, synthetic_weather, horizon_hours=24)
    c.act(0.0, _state(reference_plant), {}, synthetic_weather)
    lam = c.plan.lambda_EUR_per_kWh
    assert np.all(lam >= -1e-9), f"negative shadow price: min {lam.min()}"
    assert lam.max() < 1.0, (
        f"lambda of {lam.max():.3f} EUR/kWh is far above anything this plant can "
        "justify; the scaling or the sign convention is wrong"
    )


def test_lambda_is_zero_where_the_plan_spills(reference_plant, synthetic_weather):
    """Curtailment interior implies energy is free -- the primal-side check."""
    c = _controller(reference_plant, synthetic_weather, horizon_hours=24)
    c.act(0.0, _state(reference_plant), {}, synthetic_weather)
    spilling = c.plan.curtail > 1e-3
    if spilling.any():
        assert np.all(c.plan.lambda_EUR_per_kWh[spilling] < 1e-3), (
            "the plan is spilling power while pricing it above zero, which no "
            "optimal solution can do"
        )


# --- the band interface ---------------------------------------------------
def test_band_setpoint_is_zero_when_not_committed():
    assert Band(committed=False, setpoint_max=0.9, dispatch=1.0).setpoint == 0.0
    assert Band(committed=True, setpoint_max=0.9, dispatch=1.0).setpoint == 0.9


def test_bands_pin_the_inner_nmpc(reference_plant):
    """A band must become a *bound*, which is the whole point of banding."""
    from sfp.control.nmpc import ENABLE_INDEX, SETPOINT_INDEX, InnerNMPC

    nmpc = InnerNMPC(reference_plant, Economics(), horizon_s=600.0, dt_s=300.0)
    bands = {
        "contactor": Band(True, 0.4, 0.0),
        "calciner": Band(False, 0.0, 0.0),
        "electrolyser": Band(True, 1.0, 0.0),
        "sabatier": Band(True, 0.7, 0.0),
    }
    lo, hi, guess, ceilings = nmpc._bind_bands(bands)

    i_e = nmpc._u_index("calciner", ENABLE_INDEX)
    i_s = nmpc._u_index("calciner", SETPOINT_INDEX)
    assert lo[i_e] == hi[i_e] == 0.0, "a decommitted machine must be pinned off"
    assert lo[i_s] == hi[i_s] == 0.0, (
        "a decommitted machine's setpoint is a free variable nothing depends on "
        "-- a flat manifold for the solver to wander over"
    )
    assert not np.isfinite(ceilings[i_s]), (
        "a pinned-off machine needs no soft ceiling; its setpoint is already zero"
    )

    i_e = nmpc._u_index("contactor", ENABLE_INDEX)
    i_s = nmpc._u_index("contactor", SETPOINT_INDEX)
    assert lo[i_e] == hi[i_e] == 1.0, "a committed machine must be pinned on"
    assert ceilings[i_s] == pytest.approx(0.4), "the band ceiling must be recorded"
    assert hi[i_s] > ceilings[i_s], (
        "the ceiling must NOT be written into the box. A setpoint is slew-limited,"
        " so a hard ceiling plus the rate limit has no feasible point whenever a"
        " band narrows faster than the actuator can follow -- measured, 115 of 216"
        " solves came back Infeasible_Problem_Detected. The ceiling is enforced as"
        " a penalised row instead, which admits the ramp the actuator must make."
    )
    assert lo[i_s] <= guess[i_s] <= ceilings[i_s], (
        "the guess should start inside the band even though the box is wider"
    )


def test_plan_indexing_is_within_range(reference_plant, synthetic_weather):
    c = _controller(reference_plant, synthetic_weather, horizon_hours=12)
    c.act(0.0, _state(reference_plant), {}, synthetic_weather)
    plan = c.plan
    for t in (0.0, 1800.0, 3600.0, 1e6, -1.0):
        k = plan.index_at(t)
        assert 0 <= k < len(plan.bands)
    assert plan.target_at(1e6).shape == (N_Z,)


# --- battery replacement economics ---------------------------------------
def test_wear_cost_accounts_for_depth_and_round_trip(reference_plant):
    """C_deg = CAPEX / (Capacity * DoD * CycleLife * RTE), not CAPEX/CycleLife.

    The bare form treats a cycle as moving the full nameplate with no losses.
    It moves `dod_reference` of it, and delivers `RTE` of that.
    """
    b = reference_plant["battery"]
    p = b.p
    expected = b.capex_EUR() / (
        p.capacity_kwh * p.dod_reference * p.cycle_life * b.round_trip_efficiency())
    assert b.cost_per_kWh_delivered_EUR() == pytest.approx(expected)
    naive = b.capex_EUR() / (p.cycle_life * p.capacity_kwh)
    assert b.cost_per_kWh_delivered_EUR() / naive == pytest.approx(
        1.0 / (p.dod_reference * b.round_trip_efficiency()), rel=1e-9), (
        "the correction should be exactly 1/(DoD * RTE); anything else means "
        "the two effects are not both being applied"
    )


def test_rated_depth_cycle_costs_exactly_one_rated_cycle(reference_plant):
    """The depth-stress term must not quietly extend the pack's life.

    Stress is evaluated instantaneously; `cycle_life` rates a whole cycle. A
    cycle at the reference depth sweeps normalised depth 0 -> 1 -> 0, so the mean
    of `depth^(k-1)` along it is `1/k`. Unnormalised this returned 17,450 cycles
    against a datasheet 6,000.
    """
    b = reference_plant["battery"]
    depth = np.linspace(0.0, 1.0, 4001)
    stress = b._dod_normalisation * np.maximum(
        depth, b.p.dod_stress_floor) ** (b.p.dod_exponent - 1.0)
    effective_cycles = b.p.cycle_life / stress.mean()
    assert effective_cycles == pytest.approx(b.p.cycle_life, rel=1e-3), (
        f"a rated-depth cycle should cost one rated cycle; this gives "
        f"{effective_cycles:.0f} against {b.p.cycle_life:.0f}"
    )


def test_deeper_cycling_shortens_life(reference_plant):
    b = reference_plant["battery"]
    shallow = b.life_years(365.0, mean_stress=0.3)
    deep = b.life_years(365.0, mean_stress=1.5)
    assert deep < shallow, "deep cycling must cost more life than shallow"


def test_only_the_calendar_share_of_replacement_is_uncharged(reference_plant):
    """Cycling is already funded by the operating objective; time is not.

    Charging the whole replacement here as well would count cycling twice.
    """
    b = reference_plant["battery"]
    kw = dict(project_years=25.0, discount_rate=0.07)

    idle = b.uncharged_replacement_PV_EUR(efc_per_year=0.0, **kw)
    assert idle > 0.0, (
        "a pack that is never cycled still ages out and still needs replacing "
        "before the project ends"
    )

    worked = b.uncharged_replacement_PV_EUR(efc_per_year=730.0, **kw)
    assert worked > idle, (
        "working the pack shortens its life, so more replacements are needed "
        "even though a smaller share of each is calendar-attributable"
    )

    cal, cyc = b.fade_rates_per_year(730.0)
    assert cal / (cal + cyc) < 0.5, "at two cycles a day, cycling should dominate"


# --- the solver seam ------------------------------------------------------
def test_mip_duals_come_from_the_fixed_relaxation():
    """A MILP has no duals; fixing the integers and re-solving is how to get one."""
    import scipy.sparse as sp

    from sfp.solvers.lp import LPProblem, solve_lp

    # minimise -x - y  s.t.  x + y <= 1.5, x integer in [0,1], y in [0,1]
    A = sp.csc_matrix(np.array([[1.0, 1.0]]))
    problem = LPProblem(
        c=np.array([-1.0, -1.0]), A=A,
        lo=np.array([-np.inf]), hi=np.array([1.5]),
        xl=np.array([0.0, 0.0]), xu=np.array([1.0, 1.0]),
        blocks={"cap": slice(0, 1)},
        integrality=np.array([1.0, 0.0]),
    )
    # `duals=True` is what buys the second, integer-fixed solve; it is off by
    # default because nothing in the shipped controller reads lambda.
    sol = solve_lp(problem, duals=True)
    assert sol.success
    assert sol.x[0] == pytest.approx(1.0, abs=1e-6), "the integer column should be 1"
    assert sol.f == pytest.approx(-1.5, abs=1e-6)
    # relaxing the cap by one unit is worth one more unit of objective
    assert sol.dual("cap")[0] == pytest.approx(1.0, abs=1e-6)

    # and without it there is no dual to read, rather than a silently wrong one
    assert solve_lp(problem).dual("cap")[0] == pytest.approx(0.0, abs=1e-12)
