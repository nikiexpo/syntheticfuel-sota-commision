"""Economic planner tests.

Two things are checked here that nothing else can catch.

**The reduced model must not drift from the plant.** The planner rewrites some
subsystem expressions to remove non-smooth guards, and a rewrite is a chance to
introduce a different model by accident. Every rate is pinned against the
subsystem it came from.

**Lambda must be a price.** It is the single number the whole hierarchy is
coordinated by, and both its sign and its magnitude are easy to get wrong in ways
that raise nothing at all. It is checked against the thermodynamics -- the value
of a kilowatt-hour to this plant is the methane it can become -- and against
scarcity, which is the property the inner layer will actually respond to.
"""

from __future__ import annotations

import numpy as np
import pytest

from sfp.cli import build_plant
from sfp.control.base import ControlContext
from sfp.control.planner import EconomicPlanner
from sfp.control.planner_model import (
    CONTROL_NAMES,
    N_U,
    N_Z,
    STATE_NAMES,
    PlannerModel,
    ui,
    zi,
)
from sfp.economics import Economics

HOT_KILN_K = 1173.15


@pytest.fixture(scope="module")
def plant():
    return build_plant(1100.0, 1500.0, 750.0)


@pytest.fixture(scope="module")
def model(plant):
    return PlannerModel(plant)


@pytest.fixture
def weather_row():
    return {"temp_air": 25.0, "relative_humidity": 55.0,
            "wind_speed": 2.0, "pressure": 101325.0}


def _control(**overrides) -> np.ndarray:
    u = np.zeros(N_U)
    for name, value in overrides.items():
        u[ui(name)] = value
    return u


def _hot_state(model) -> np.ndarray:
    z = model.initial_state(model.plant.split(model.plant.initial_state()))
    z[zi("kiln_temperature_K")] = HOT_KILN_K
    return z


# --- structure --------------------------------------------------------------


def test_state_and_control_names_are_unique_and_indexed_consistently():
    assert len(set(STATE_NAMES)) == len(STATE_NAMES) == N_Z
    assert len(set(CONTROL_NAMES)) == len(CONTROL_NAMES) == N_U
    for i, name in enumerate(STATE_NAMES):
        assert zi(name) == i
    for i, name in enumerate(CONTROL_NAMES):
        assert ui(name) == i


def test_reduced_state_projects_the_plant_state(model, plant):
    full = plant.split(plant.initial_state())
    z = model.initial_state(full)
    assert z[zi("soc")] == pytest.approx(full["battery"][0])
    assert z[zi("n_caco3")] == pytest.approx(full["solids"][1])
    assert z[zi("cycle_number")] == pytest.approx(full["solids"][2])
    assert z[zi("n_h2")] == pytest.approx(full["gas"][0])
    assert z[zi("n_co2")] == pytest.approx(full["gas"][1])
    assert z[zi("kiln_temperature_K")] == pytest.approx(full["calciner"][0])


def test_scales_are_positive_and_the_right_length(model):
    assert model.state_scale().shape == (N_Z,)
    assert model.control_scale().shape == (N_U,)
    assert np.all(model.state_scale() > 0)
    assert np.all(model.control_scale() > 0)
    assert model.power_scale() > 0


# --- fidelity against the plant ---------------------------------------------


def test_planner_rates_match_the_plant(model, weather_row):
    """The smoothed expressions must agree with the subsystems they replaced.

    Tolerances are 1e-6 rather than exact because the plant's own gates are
    smooth: `smooth_step(0.5, width=0.05)` is 0.99999999, not 1, so a subsystem
    at full enable is already a few parts in 1e9 below its nominal rate. That is
    the plant being differentiable, not a disagreement. A tolerance this tight
    still catches any real divergence -- a dropped term or a different constant
    would be parts in 1e2, not 1e9.
    """
    z = _hot_state(model)
    u = _control(contactor_flow=0.4, calciner_heat=1.0, electrolyser_load=0.6,
                 sabatier_feed=0.8, contactor_on=1.0, calciner_on=1.0,
                 electrolyser_on=1.0, sabatier_on=1.0)
    rates = {k: float(v) for k, v in model.rates(z, u, weather_row).items()}

    # contactor: identical call
    w_con = dict(weather_row)
    w_con["solids_loading"] = float(model.loading(z))
    expected = float(model.contactor.capture_rate_mol_s(np.array([0.4, 1.0]), w_con))
    assert rates["carbonation"] == pytest.approx(expected, rel=1e-6)

    # calciner: the plant clamps at zero, which is redundant for a hot kiln
    w_cal = dict(weather_row)
    w_cal["solids_n_caco3_mol"] = float(z[zi("n_caco3")])
    expected = float(model.calciner.calcination_rate_mol_s(
        np.array([HOT_KILN_K]), np.array([1.0, 1.0]), w_cal))
    assert rates["calcination"] == pytest.approx(expected, rel=1e-6)

    # electrolyser: Faraday's law, same constants
    expected = float(model.electrolyser.hydrogen_rate_mol_s(
        np.array([333.15, 0.0]), np.array([0.6, 1.0])))
    assert rates["electrolysis"] == pytest.approx(expected, rel=1e-6)

    # reactor: smooth_min replaces fmin, so agreement is close but not exact
    w_sab = dict(weather_row)
    w_sab["gas_co2_available_mol"] = float(z[zi("n_co2")]) - 120.0
    w_sab["gas_h2_available_mol"] = float(z[zi("n_h2")]) - 1500.0
    w_sab["water_available_kg"] = 1e6
    expected = float(model.sabatier.co2_rate_mol_s(
        np.array([573.15, 1.0, 0.0]), np.array([0.8, 1.0]), w_sab))
    assert rates["methanation"] == pytest.approx(expected, rel=1e-3)


def test_planner_powers_match_the_subsystems(model, weather_row):
    """At full commitment the relaxed parasitics equal the real ones."""
    z = _hot_state(model)
    u = _control(contactor_flow=0.4, calciner_heat=0.8, electrolyser_load=0.6,
                 contactor_on=1.0, calciner_on=1.0, electrolyser_on=1.0,
                 sabatier_on=1.0)
    powers = {k: float(v) for k, v in model.powers(z, u, weather_row).items()}

    assert powers["contactor"] == pytest.approx(
        float(model.contactor.fan_power_W(np.array([0.4, 1.0]))), rel=1e-6)
    assert powers["calciner"] == pytest.approx(
        model.calciner.power_for_fraction(0.8, 1.0), rel=1e-6)
    assert powers["electrolyser"] == pytest.approx(
        float(model.electrolyser.total_power_W(
            np.array([333.15, 0.0]), np.array([0.6, 1.0]))), rel=1e-3)
    assert powers["sabatier"] == pytest.approx(
        model.sabatier.p.auxiliary_power_kw * 1e3, rel=1e-6)


def test_parasitic_load_scales_with_the_relaxed_enable(model, weather_row):
    """Half-committed means half the parasitic draw -- the relaxation's whole job."""
    z = _hot_state(model)
    full = model.powers(z, _control(electrolyser_on=1.0), weather_row)
    half = model.powers(z, _control(electrolyser_on=0.5), weather_row)
    aux = model.electrolyser.auxiliary_power_W
    assert float(full["electrolyser"]) == pytest.approx(aux, abs=1.0)
    assert float(half["electrolyser"]) == pytest.approx(0.5 * aux, abs=1.0)


def test_rates_do_not_depend_on_the_enable_except_the_kiln_feeder(model, weather_row):
    """Rate quadratic in the commitment would make the plan silently unachievable."""
    z = _hot_state(model)
    kwargs = dict(contactor_flow=0.4, electrolyser_load=0.6, sabatier_feed=0.8)
    full = model.rates(z, _control(**kwargs, contactor_on=1.0, electrolyser_on=1.0,
                                   sabatier_on=1.0, calciner_on=1.0), weather_row)
    half = model.rates(z, _control(**kwargs, contactor_on=0.5, electrolyser_on=0.5,
                                   sabatier_on=0.5, calciner_on=1.0), weather_row)
    for key in ("carbonation", "electrolysis", "methanation"):
        assert float(full[key]) == pytest.approx(float(half[key]), rel=1e-12), key

    # the calciner's enable is the feeder, and is linear
    off = model.rates(z, _control(**kwargs, calciner_on=0.0), weather_row)
    assert float(off["calcination"]) == pytest.approx(0.0, abs=1e-12)
    mid = model.rates(z, _control(**kwargs, calciner_on=0.5), weather_row)
    assert float(mid["calcination"]) == pytest.approx(
        0.5 * float(full["calcination"]), rel=1e-9)


# --- dynamics ---------------------------------------------------------------


def test_cold_kiln_does_not_calcine(model, weather_row):
    z = model.initial_state(model.plant.split(model.plant.initial_state()))
    u = _control(calciner_heat=1.0, calciner_on=1.0)
    assert float(model.rates(z, u, weather_row)["calcination"]) == pytest.approx(0.0, abs=1e-9)


def test_carbon_is_conserved_through_the_calcium_loop(model, weather_row):
    """Every mole leaving the carbonate must arrive in the CO2 buffer."""
    z = _hot_state(model)
    u = _control(calciner_heat=1.0, calciner_on=1.0)
    dz = np.asarray(model.rhs(z, u, weather_row), dtype=float).ravel()
    assert dz[zi("n_caco3")] == pytest.approx(-dz[zi("n_co2")], rel=1e-9)


def test_cycle_counter_advances_with_calcination(model, weather_row):
    z = _hot_state(model)
    u = _control(calciner_heat=1.0, calciner_on=1.0)
    dz = np.asarray(model.rhs(z, u, weather_row), dtype=float).ravel()
    rate = float(model.rates(z, u, weather_row)["calcination"])
    assert dz[zi("cycle_number")] == pytest.approx(rate / model.n_total_mol, rel=1e-9)


def test_venting_removes_hydrogen_from_the_balance(model, weather_row):
    """Without a vent the dynamics and the tank bound are jointly infeasible."""
    z = _hot_state(model)
    base = _control(electrolyser_load=1.0, electrolyser_on=1.0)
    vented = _control(electrolyser_load=1.0, electrolyser_on=1.0, h2_vent_mol_s=0.3)
    d_base = np.asarray(model.rhs(z, base, weather_row), dtype=float).ravel()
    d_vent = np.asarray(model.rhs(z, vented, weather_row), dtype=float).ravel()
    assert d_base[zi("n_h2")] - d_vent[zi("n_h2")] == pytest.approx(0.3, rel=1e-9)


def test_rk4_step_agrees_with_a_fine_forward_euler(model, weather_row):
    z = _hot_state(model)
    u = _control(contactor_flow=0.4, calciner_heat=1.0, electrolyser_load=0.6,
                 sabatier_feed=0.8, contactor_on=1.0, calciner_on=1.0,
                 electrolyser_on=1.0, sabatier_on=1.0)
    coarse = np.asarray(model.step(z, u, weather_row, 3600.0), dtype=float).ravel()

    fine = np.array(z, dtype=float)
    steps = 4000
    h = 3600.0 / steps
    for _ in range(steps):
        fine = fine + h * np.asarray(model.rhs(fine, u, weather_row), dtype=float).ravel()

    scale = model.state_scale()
    assert np.max(np.abs(coarse - fine) / scale) < 2e-3


# --- the planner ------------------------------------------------------------


@pytest.fixture(scope="module")
def solved(plant, synthetic_weather):
    """One 24-hour solve, reused by the assertions below (it is not cheap)."""
    planner = EconomicPlanner(horizon_hours=24)
    context = ControlContext(
        plant=plant, site=synthetic_weather.site, dt_s=300.0, horizon_s=86400.0,
        forecast=synthetic_weather, economics=Economics(),
    )
    planner.reset(context)
    state = plant.split(plant.initial_state())
    request = planner.act(0.0, state, {}, synthetic_weather)
    return planner, request


def test_planner_solves_and_reports_a_converged_solve(solved):
    planner, _ = solved
    assert planner.plan is not None, "planner produced no plan"
    stats = planner.plan.solve_stats
    assert stats.success, stats.status
    assert stats.constraint_violation < 1e-6
    assert planner._failures == 0


def test_lambda_is_positive_and_scales_with_scarcity(solved):
    """Lambda must be a price: non-negative, and higher when energy is short.

    The economically meaningful check is the magnitude. A kilowatt-hour is worth
    to this plant whatever methane it can become, which is the methane price over
    the plant's specific energy -- around 0.046 EUR/kWh at the reference sizing.
    A lambda far from that band means the objective, the scaling or the dual sign
    is wrong.
    """
    planner, _ = solved
    lam = planner.plan.lambda_EUR_per_kWh
    assert np.all(lam >= -1e-6), "a negative price means the dual sign is inverted"
    assert lam.max() < 0.5, "implausibly high price; check the constraint scaling"

    price = planner.economics.p.methane_price_per_kg
    thermodynamic = price / 38.8  # EUR per kWh at ~38.8 kWh/kg CH4
    assert 0.2 * thermodynamic < lam.max() < 5.0 * thermodynamic


def test_lambda_is_lower_when_the_sun_is_up(solved):
    """Surplus energy is worth less. This is the signal the NMPC responds to."""
    planner, _ = solved
    lam = planner.plan.lambda_EUR_per_kWh
    pv = planner._forecast_rows(0.0, planner.context.forecast)["pv_available_W"]
    n = min(len(lam), len(pv))
    sunny, dark = pv[:n] > np.median(pv[:n]), pv[:n] <= 1.0
    if sunny.any() and dark.any():
        assert lam[:n][sunny].mean() < lam[:n][dark].mean()


def test_commitment_relaxation_settles_on_the_setpoint(solved):
    """`e = s` at the optimum, which is what makes the rounding harmless."""
    planner, _ = solved
    u = planner.plan.controls
    for key, setpoint in (("contactor", "contactor_flow"),
                          ("electrolyser", "electrolyser_load"),
                          ("sabatier", "sabatier_feed")):
        s = u[:, ui(setpoint)]
        e = u[:, ui(f"{key}_on")]
        assert np.all(e >= s - 1e-5), f"{key}: setpoint exceeded its enable"
        assert np.max(e - s) < 0.05, f"{key}: enable did not settle onto the setpoint"


def test_plan_respects_the_state_bounds(solved):
    planner, _ = solved
    limits = planner.model.limits()
    lo, hi = limits.lower(), limits.upper()
    assert np.all(planner.plan.states >= lo - 1e-6)
    assert np.all(planner.plan.states <= hi + 1e-6)


def test_electrolyser_never_runs_below_its_crossover_limit(solved):
    """A safety constraint, not a preference: below it H2 crosses into the O2."""
    planner, _ = solved
    u = planner.plan.controls
    load = u[:, ui("electrolyser_load")]
    enable = u[:, ui("electrolyser_on")]
    floor = planner.model.electrolyser_min_load
    running = enable > 1e-4
    assert np.all(load[running] >= floor * enable[running] - 1e-6)


def test_planner_beats_its_own_initial_guess(solved):
    """The rollout is a starting point, not the answer."""
    planner, _ = solved
    n = planner.plan.controls.shape[0]
    z0 = planner.plan.states[0]
    weather = planner._forecast_rows(0.0, planner.context.forecast)
    guess_z, guess_u = planner._rollout_guess(n, z0, weather)

    dt = planner.dt_plan_s
    guess_profit = 0.0
    for k in range(n):
        w = {key: float(weather[key][k]) for key in
             ("temp_air", "relative_humidity", "wind_speed", "pressure")}
        guess_profit += float(planner.model.stage_profit_EUR(
            guess_z[k], guess_u[k], w, planner.economics, dt))
    assert planner.plan.objective_EUR > guess_profit


def test_request_is_well_formed(solved):
    _, request = solved
    for key, value in request.setpoints.items():
        assert 0.0 <= value <= 1.0, key
    for key, value in request.enables.items():
        assert value in (0.0, 1.0), f"{key} was not rounded to a commitment"
    assert request.battery_charge_W >= 0.0
    assert request.battery_discharge_W >= 0.0
    assert 0.0 <= request.curtail_fraction <= 1.0


def test_plan_indexing_is_clamped_to_the_horizon(solved):
    planner, _ = solved
    plan = planner.plan
    assert plan.index_at(plan.t0_s - 1e6) == 0
    assert plan.index_at(plan.t0_s + 1e9) == len(plan.controls) - 1
    assert plan.control_at(plan.t0_s).shape == (N_U,)


def test_no_plan_means_everything_off_rather_than_a_guess(plant, synthetic_weather):
    """A controller with no plan must not invent one that looks like it works."""
    planner = EconomicPlanner(horizon_hours=4)
    context = ControlContext(
        plant=plant, site=synthetic_weather.site, dt_s=300.0, horizon_s=86400.0,
        forecast=synthetic_weather, economics=Economics(),
    )
    planner.reset(context)
    request = planner.act(0.0, plant.split(plant.initial_state()), {}, None)
    assert planner.plan is not None or all(
        v == 0.0 for v in request.setpoints.values()
    )
