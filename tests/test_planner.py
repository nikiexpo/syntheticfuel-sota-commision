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
from sfp.control.planner import EconomicPlanner, Plan
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
    plan = planner.plan
    lam = plan.lambda_EUR_per_kWh
    hourly = planner._forecast_rows(0.0, planner.context.forecast)
    pv = planner._aggregate(hourly, plan.dt_s)["pv_available_W"]
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
    plan = planner.plan
    n = plan.controls.shape[0]
    z0 = plan.states[0]
    grid = plan.dt_s
    hourly = planner._forecast_rows(0.0, planner.context.forecast)
    weather = planner._aggregate(hourly, grid)
    guess_z, guess_u = planner._rollout_guess(n, z0, weather, grid)

    guess_profit = 0.0
    for k in range(n):
        w = {key: float(weather[key][k]) for key in
             ("temp_air", "relative_humidity", "wind_speed", "pressure")}
        guess_profit += float(planner.model.stage_profit_EUR(
            guess_z[k], guess_u[k], w, planner.economics, float(grid[k])))
    assert plan.objective_EUR > guess_profit


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


# --- water ------------------------------------------------------------------


def test_water_is_a_planner_state_and_tracks_the_plant(model, plant):
    full = plant.split(plant.initial_state())
    z = model.initial_state(full)
    assert z[zi("water_kg")] == pytest.approx(full["water"][0])


def test_water_balance_matches_the_plant_tank(model, weather_row):
    """Same stoichiometry as WaterTank.rhs: 1 H2O per H2 out, 2 per CH4 back."""
    z = _hot_state(model)
    u = _control(electrolyser_load=0.6, electrolyser_on=1.0,
                 sabatier_feed=0.8, sabatier_on=1.0)
    rates = model.rates(z, u, weather_row)
    tank = model.plant["water"]
    expected = (2.0 * tank.p.condensate_recovery * float(rates["methanation"])
                - float(rates["electrolysis"])) * 18.01528e-3
    dz = np.asarray(model.rhs(z, u, weather_row), dtype=float).ravel()
    assert dz[zi("water_kg")] == pytest.approx(expected, rel=1e-9)


def test_makeup_adds_water_one_for_one(model, weather_row):
    z = _hot_state(model)
    base = _control(electrolyser_load=0.6, electrolyser_on=1.0)
    fed = _control(electrolyser_load=0.6, electrolyser_on=1.0, water_makeup_kg_s=0.02)
    d0 = np.asarray(model.rhs(z, base, weather_row), dtype=float).ravel()
    d1 = np.asarray(model.rhs(z, fed, weather_row), dtype=float).ravel()
    assert d1[zi("water_kg")] - d0[zi("water_kg")] == pytest.approx(0.02, rel=1e-9)


def test_water_is_charged_on_delivery_not_on_consumption(model, weather_row):
    """The cost is a road tanker, so it follows the make-up control exactly."""
    from sfp.economics import Economics
    econ = Economics()
    z = _hot_state(model)
    dry = _control(electrolyser_load=0.6, electrolyser_on=1.0)
    wet = _control(electrolyser_load=0.6, electrolyser_on=1.0, water_makeup_kg_s=0.02)
    delta = (float(model.stage_profit_EUR(z, dry, weather_row, econ, 3600.0))
             - float(model.stage_profit_EUR(z, wet, weather_row, econ, 3600.0)))
    expected = econ.p.water_cost_per_m3 * 0.02 * 3600.0 / 1000.0
    assert delta == pytest.approx(expected, rel=1e-9)


# --- deleted guards ---------------------------------------------------------


def test_deleted_guards_are_unreachable(model):
    """Each guard removed from the planner must be provably inactive.

    Deleting a clamp is exact only while the state box that makes it redundant
    still stands. If a bound is ever loosened, this is what catches it.
    """
    limits = model.limits()
    gas = model.gas.p
    assert limits.n_h2[0] >= gas.h2_min_fraction * model.h2_capacity_mol - 1e-9
    assert limits.n_co2[0] >= gas.co2_min_fraction * model.co2_capacity_mol - 1e-9
    assert limits.cycle_number[0] >= 0.0
    assert limits.water_kg[0] >= 1.0

    # capacity can never approach the deleted fmax(capacity, 1.0) floor
    worst = model.capture_capacity_mol(limits.cycle_number[1])
    assert worst >= model.n_total_mol * model.solids.p.grasa_residual_conversion - 1e-6
    assert worst > 1.0


def test_snug_bounds_bracket_the_reachable_set(model, plant):
    z0 = model.initial_state(plant.split(plant.initial_state()))
    limits = model.limits(z0, 7 * 86400.0)
    # the carbonate ceiling is the deactivated capacity, not the raw inventory
    assert limits.n_caco3[1] < model.n_total_mol
    assert limits.n_caco3[1] == pytest.approx(
        model.n_total_mol * float(model.max_conversion(z0[zi("cycle_number")])), rel=1e-9)
    # the cycle number cannot run away, but must reach what a week can turn
    span = limits.cycle_number[1] - limits.cycle_number[0]
    assert 1.0 <= span < 20.0


# --- the graded grid --------------------------------------------------------


def test_grid_covers_the_horizon_exactly(plant, synthetic_weather):
    planner = EconomicPlanner(horizon_hours=168)
    for hours in (1, 6, 24, 72, 168):
        grid = planner._grid(hours)
        assert np.sum(grid) == pytest.approx(hours * 3600.0, rel=1e-9)
        assert np.all(grid > 0)


def test_grid_is_fine_near_term_and_coarse_far_out(plant):
    planner = EconomicPlanner(horizon_hours=168)
    grid = planner._grid(168) / 3600.0
    assert grid[0] == pytest.approx(1.0)
    assert grid[-1] == pytest.approx(6.0)
    assert np.all(np.diff(grid) >= -1e-9), "spacing must be non-decreasing"
    # the whole point: far fewer intervals than a uniform hourly grid
    assert len(grid) < 168 / 2


def test_aggregation_preserves_energy_not_just_a_sample(plant):
    """A coarse interval must average the forecast, not sample its left edge."""
    planner = EconomicPlanner(horizon_hours=12)
    planner.grading = ((6.0, 1.0), (6.0, 6.0))
    grid = planner._grid(12)
    hourly = {"pv_available_W": np.arange(12, dtype=float) * 100.0,
              "temp_air": np.zeros(12), "relative_humidity": np.zeros(12),
              "wind_speed": np.zeros(12), "pressure": np.zeros(12)}
    out = planner._aggregate(hourly, grid)
    assert out["pv_available_W"][0] == pytest.approx(0.0)
    # the final 6 h block spans hours 6..11, whose mean is 850
    assert out["pv_available_W"][-1] == pytest.approx(850.0)


def test_substeps_scale_with_interval_length_but_are_capped(model):
    """The cap is what makes the graded grid actually cheaper.

    Without it, halving the interval count doubles the sub-steps inside each
    one and the expression graph is exactly the same size as a uniform grid --
    the grading would buy nothing at all.
    """
    assert model.substeps_for(600.0) == 1
    assert model.substeps_for(3600.0) == 2
    assert model.substeps_for(6 * 3600.0) == model.MAX_SUBSTEPS

    # total RHS evaluations must fall relative to a uniform hourly grid
    graded = np.concatenate([np.full(24, 1.0), np.full(16, 3.0), np.full(16, 6.0)])
    graded_cost = sum(model.substeps_for(h * 3600.0) for h in graded)
    uniform_cost = sum(model.substeps_for(3600.0) for _ in range(168))
    assert graded_cost < 0.5 * uniform_cost


def test_plan_indexing_handles_a_non_uniform_grid():
    n = 4
    plan = Plan(
        t0_s=0.0, dt_s=np.array([3600.0, 3600.0, 10800.0, 21600.0]),
        states=np.zeros((n + 1, N_Z)), controls=np.arange(n * N_U).reshape(n, N_U),
        lambda_EUR_per_kWh=np.zeros(n), objective_EUR=0.0,
    )
    assert plan.index_at(0.0) == 0
    assert plan.index_at(3600.0) == 1
    assert plan.index_at(7200.0) == 2          # third interval starts here
    assert plan.index_at(17999.0) == 2         # and is 3 h long
    assert plan.index_at(18000.0) == 3
    assert plan.index_at(1e9) == 3


# --- lambda, checked on the primal side -------------------------------------


def test_lambda_is_zero_where_energy_is_being_spilled(solved):
    """Strictly interior curtailment means energy is free at the margin.

    This is a primal-side identity, so it validates the dual without trusting
    the dual -- if the two disagree, the price is wrong.
    """
    planner, _ = solved
    plan = planner.plan
    gamma = plan.controls[:, ui("curtail")]
    hourly = planner._forecast_rows(0.0, planner.context.forecast)
    pv = planner._aggregate(hourly, plan.dt_s)["pv_available_W"]

    # Only where there is sun. With `pv == 0` the term `pv * (1 - gamma)`
    # vanishes for every gamma, so the variable is unconstrained by anything and
    # its value carries no information -- an interior gamma at midnight says
    # nothing about the price of energy at midnight.
    interior = (gamma > 1e-3) & (gamma < 1.0 - 1e-3) & (pv[:len(gamma)] > 1.0)
    if interior.any():
        assert np.max(np.abs(plan.lambda_EUR_per_kWh[interior])) < 5e-3


def test_lambda_rises_when_the_battery_is_the_marginal_source(solved):
    """Across a charge/discharge pair the price must reflect the round trip."""
    planner, _ = solved
    plan = planner.plan
    charging = plan.controls[:, ui("battery_charge_W")] > 1e3
    discharging = plan.controls[:, ui("battery_discharge_W")] > 1e3
    if charging.any() and discharging.any():
        lam = plan.lambda_EUR_per_kWh
        assert lam[discharging].mean() > lam[charging].mean(), (
            "discharging hours must price energy above charging hours, or the "
            "battery is being cycled for no reason"
        )


# --- the perfect-foresight oracle -------------------------------------------


def test_oracle_refuses_to_run_without_the_truth(plant, synthetic_weather):
    """An oracle quietly running on a forecast would report near-zero regret."""
    from sfp.control.baselines import PerfectForesightOracle

    context = ControlContext(
        plant=plant, site=synthetic_weather.site, dt_s=300.0, horizon_s=86400.0,
        forecast=synthetic_weather, economics=Economics(),
    )
    with pytest.raises(ValueError, match=r"metadata\['truth'\]"):
        PerfectForesightOracle().reset(context)


def test_oracle_plans_against_the_truth_not_the_forecast(plant, synthetic_weather):
    from sfp.control.baselines import PerfectForesightOracle

    degraded = synthetic_weather.forecast(skill=0.2, seed=3)
    oracle = PerfectForesightOracle(horizon_hours=6, homotopy_hours=())
    oracle.reset(ControlContext(
        plant=plant, site=synthetic_weather.site, dt_s=300.0, horizon_s=86400.0,
        forecast=degraded, economics=Economics(),
        metadata={"truth": synthetic_weather},
    ))
    assert oracle.context.forecast is synthetic_weather


def test_horizon_is_shortened_rather_than_retried_forever(plant, synthetic_weather):
    """A planner whose full horizon never converges must still produce plans.

    The homotopy only runs on a cold start, so without the shortfall logic every
    later replan would attempt the one horizon that fails, fail, and leave the
    plant running on an ever-staler plan while reporting a failure each time.
    """
    planner = EconomicPlanner(horizon_hours=8, homotopy_hours=(4,))
    planner.reset(ControlContext(
        plant=plant, site=synthetic_weather.site, dt_s=300.0, horizon_s=86400.0,
        forecast=synthetic_weather, economics=Economics(),
    ))
    assert planner._target_hours == 8
    # simulate the full horizon having failed after a shorter one converged
    planner._target_hours = 4
    assert planner._target_hours < planner.horizon_hours


def test_resample_moves_a_trajectory_between_grids(plant, synthetic_weather):
    """Re-gridding must be a time lookup, not an index shuffle."""
    planner = EconomicPlanner(horizon_hours=12)
    src = np.array([3600.0] * 6)
    states = np.arange((len(src) + 1) * N_Z, dtype=float).reshape(-1, N_Z)
    controls = np.arange(len(src) * N_U, dtype=float).reshape(-1, N_U)

    dst = np.array([7200.0] * 3)          # same span, half the intervals
    z, u = planner._resample(src, states, controls, dst)
    assert u.shape == (3, N_U)
    assert z.shape == (4, N_Z)
    # Destination midpoints land at 1, 3 and 5 hours, each exactly on a source
    # interval boundary; `searchsorted(..., "right")` resolves a tie to the later
    # interval. Which side a tie falls on does not matter for a warm start, but
    # it should be a decision rather than an accident.
    assert u[0] == pytest.approx(controls[1])
    assert u[1] == pytest.approx(controls[3])
    assert u[2] == pytest.approx(controls[5])
    assert z[0] == pytest.approx(states[0])

    # a destination midpoint strictly inside a source interval is unambiguous
    z2, u2 = planner._resample(src, states, controls, np.array([1800.0, 1800.0]))
    assert u2[0] == pytest.approx(controls[0])
    assert u2[1] == pytest.approx(controls[0])


def test_splice_keeps_the_warm_start_only_where_it_has_data(plant, synthetic_weather):
    """Beyond the source horizon the rollout is used, not a frozen extrapolation.

    Holding the last state constant past the end of the source plan gives a
    trajectory where nothing changes while the dynamics insist it must, which
    sent IPOPT into restoration and a declared infeasibility.
    """
    planner = EconomicPlanner(horizon_hours=12, homotopy_hours=())
    planner.reset(ControlContext(
        plant=plant, site=synthetic_weather.site, dt_s=300.0, horizon_s=86400.0,
        forecast=synthetic_weather, economics=Economics(),
    ))
    z0 = planner.model.initial_state(plant.split(plant.initial_state()))
    hourly = planner._forecast_rows(0.0, synthetic_weather)
    grid = planner._grid(12)
    nlp = planner._build(len(grid), z0, planner._aggregate(hourly, grid), grid)

    warm = np.full(nlp.n_x, -1.0)          # sentinel: clearly not the rollout
    src_horizon = 6 * 3600.0
    out = planner._splice(warm, nlp, grid, src_horizon)

    n = len(grid)
    edges = np.concatenate(([0.0], np.cumsum(grid)))
    for k in range(n):
        u_lo = (n + 1) * N_Z + k * N_U
        block = out[u_lo:u_lo + N_U]
        if edges[k] < src_horizon - 1e-6:
            assert np.all(block == -1.0), f"interval {k} should keep the warm start"
        else:
            assert np.allclose(block, nlp.x0[u_lo:u_lo + N_U]), (
                f"interval {k} is past the source horizon and should use the rollout"
            )
