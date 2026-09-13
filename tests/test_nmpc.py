"""Inner NMPC tests.

The NMPC is where the plant's own model meets a gradient-based solver, so these
tests are mostly about that boundary: the dynamics it optimises against must be
the plant's, its decision vector must come back in physical units, and the three
things that stopped it working must stay fixed.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pytest

from sfp.cli import build_reference_plant
from sfp.control.nmpc import ENABLE_INDEX, SETPOINT_INDEX, InnerNMPC
from sfp.economics import Economics

ALL_ON = {"contactor": 1.0, "calciner": 1.0, "electrolyser": 1.0, "sabatier": 1.0}


@pytest.fixture(scope="module")
def plant():
    return build_reference_plant()


@pytest.fixture(scope="module")
def nmpc(plant):
    return InnerNMPC(plant, Economics(), horizon_s=1800.0, dt_s=300.0)


@pytest.fixture
def sunny_rows(nmpc):
    row = {"poa_global": 950.0, "ghi": 850.0, "ghi_clear": 900.0,
           "clearsky_index": 0.94, "temp_air": 28.0, "relative_humidity": 45.0,
           "wind_speed": 2.0, "pressure": 101325.0, "cos_zenith": 0.9}
    return [dict(row) for _ in range(nmpc.n_steps)]


def _running_state(plant):
    x = plant.initial_state()
    names = plant.state_names()
    x[names.index("calciner.temperature_K")] = 1173.15
    x[names.index("sabatier.temperature_K")] = 573.15
    return x


def _symbolic_inputs(plant):
    return {key: (ca.MX.sym(f"u_{key}", sub.n_inputs) if sub.n_inputs
                  else np.zeros(0)) for key, sub in plant}


WEATHER = {"poa_global": 800.0, "ghi": 700.0, "ghi_clear": 750.0,
           "clearsky_index": 0.9, "temp_air": 25.0, "relative_humidity": 55.0,
           "wind_speed": 2.0, "pressure": 101325.0, "cos_zenith": 0.8}


def _terminal(plant, x0, weight=0.5):
    """Targets and inventory prices, as the hierarchical controller supplies them.

    Required, not optional: lambda prices only the battery, so without these
    nothing rewards making hydrogen or carbonate inside a one-hour horizon.
    """
    names = plant.state_names()
    keys = ("gas.n_h2", "gas.n_co2", "solids.n_caco3", "battery.soc")
    targets = {n: float(x0[names.index(n)]) for n in keys}
    per_mol = Economics().p.methane_price_per_kg * 16.0425e-3
    per_mol *= weight
    kwh_per_soc = plant["battery"].nominal_energy_J / 3.6e6
    prices = {"gas.n_h2": per_mol / 4.0, "gas.n_co2": per_mol,
              "solids.n_caco3": per_mol,
              "battery.soc": per_mol * kwh_per_soc / 56.4 / 2.016e-3 / 4.0}
    return targets, prices


# --- the symbolic plant -----------------------------------------------------


def test_the_plant_differentiates_symbolically(plant):
    """The controller uses the plant's equations, not a re-typed copy."""
    x = ca.MX.sym("x", plant.n_states)
    dx = plant.rhs(0.0, x, _symbolic_inputs(plant), WEATHER)
    assert dx.shape == (plant.n_states, 1)


def test_step_without_clipping_is_required_for_symbolic_use(plant):
    """`clip_state` casts to float and cannot accept an expression.

    Skipping it is also correct for an optimiser: bounds are constraints it can
    see and respect, where clipping would hide a violation inside the dynamics.
    """
    x = ca.MX.sym("x", plant.n_states)
    u = _symbolic_inputs(plant)
    assert plant.step(0.0, x, u, WEATHER, 300.0, clip=False).shape == (
        plant.n_states, 1)
    with pytest.raises(Exception):
        plant.step(0.0, x, u, WEATHER, 300.0, clip=True)


def test_step_function_matches_the_plant_numerically(nmpc, plant, sunny_rows):
    """The compiled one-step map must agree with integrating the plant directly.

    It exists only as a speed measure -- inlining `Plant.step` put 48 full
    coupled evaluations in one expression graph -- so it must not also be a
    different model.
    """
    x0 = _running_state(plant)
    u = nmpc._guess_u()
    from_fn = np.asarray(
        nmpc._step_fn(x=x0, u=u, w=nmpc._weather_vector(sunny_rows[0]))["x_next"]
    ).ravel()
    direct = plant.step(0.0, x0, nmpc._split_u(u), sunny_rows[0],
                        nmpc.dt_s, clip=False)
    assert from_fn == pytest.approx(
        np.asarray(direct, dtype=float).ravel(), rel=1e-9)


# --- scaling and units ------------------------------------------------------


def test_state_scale_spans_the_real_magnitudes(nmpc):
    scale = nmpc._x_scale
    assert scale.shape == (nmpc.n_x,)
    assert np.all(scale >= 1.0)
    # the point of it: raw states differ by four orders of magnitude
    assert scale.max() / scale.min() > 1e3


def test_solution_is_returned_in_physical_units(nmpc, plant, sunny_rows):
    """A missing de-scale would report battery power as a fraction, not watts."""
    x0 = _running_state(plant)
    targets, terminal = _terminal(plant, x0)
    sol = nmpc.solve(0.0, x0, np.full(nmpc.n_steps, 0.05), sunny_rows,
                     enables=ALL_ON, targets=targets, terminal_prices=terminal)
    assert sol.stats.success, sol.stats.status
    p_max = plant["battery"].max_power_W
    assert 0.0 <= sol.battery_charge_W <= p_max + 1.0
    assert 0.0 <= sol.battery_discharge_W <= p_max + 1.0
    for key, value in sol.setpoints.items():
        assert 0.0 <= value <= 1.0, key
    assert 0.0 <= sol.curtail_fraction <= 1.0


# --- the three things that stopped it working -------------------------------


def test_commitment_is_fixed_not_optimised(nmpc):
    """The plant gates enables with a near-step; as free variables they stall it.

    Commitment is the planner's decision anyway -- it is one of the three things
    the planner publishes -- so pinning it is both the numerical fix and the
    architecturally correct division of labour.
    """
    lo, hi, _ = nmpc._bind_enables({**ALL_ON, "calciner": 0.0})
    i_e = nmpc._u_index("calciner", ENABLE_INDEX)
    i_s = nmpc._u_index("calciner", SETPOINT_INDEX)
    assert lo[i_e] == hi[i_e] == 0.0
    # a shut-down subsystem's setpoint is pinned too, or it is a flat manifold
    assert lo[i_s] == hi[i_s] == 0.0
    i_on = nmpc._u_index("contactor", ENABLE_INDEX)
    assert lo[i_on] == hi[i_on] == 1.0


def test_a_shut_down_subsystem_stays_off(nmpc, plant, sunny_rows):
    x0 = _running_state(plant)
    targets, terminal = _terminal(plant, x0)
    sol = nmpc.solve(0.0, x0, np.full(nmpc.n_steps, 0.05), sunny_rows,
                     enables={**ALL_ON, "calciner": 0.0},
                     targets=targets, terminal_prices=terminal)
    assert sol.stats.success, sol.stats.status
    assert sol.enables["calciner"] == 0.0
    assert sol.setpoints["calciner"] == pytest.approx(0.0, abs=1e-6)


def test_battery_does_not_charge_and_discharge_at_once(nmpc, plant, sunny_rows):
    """Without the wear term nothing penalises churn and the solver exploits it.

    Only *process* power is priced, so a simultaneous 635 kW in / 500 kW out --
    58 kW of pure round-trip loss -- was free, and that is exactly what came
    back. The wear cost makes it strictly worse than either alone, which is how
    this formulation avoids an explicit complementarity constraint.
    """
    x0 = _running_state(plant)
    targets, terminal = _terminal(plant, x0)
    sol = nmpc.solve(0.0, x0, np.full(nmpc.n_steps, 0.05), sunny_rows,
                     enables=ALL_ON, targets=targets, terminal_prices=terminal)
    assert sol.stats.success, sol.stats.status
    assert min(sol.battery_charge_W, sol.battery_discharge_W) < 1e3


def test_it_converges_and_warm_starts_cheaply(nmpc, plant, sunny_rows):
    """A cold solve must converge, and the next must not cost more."""
    x0 = _running_state(plant)
    targets, terminal = _terminal(plant, x0)
    prices = np.full(nmpc.n_steps, 0.05)
    nmpc._warm = None
    kw = dict(enables=ALL_ON, targets=targets, terminal_prices=terminal)
    first = nmpc.solve(0.0, x0, prices, sunny_rows, **kw)
    second = nmpc.solve(0.0, x0, prices, sunny_rows, **kw)
    assert first.stats.success and second.stats.success
    assert second.stats.iterations <= first.stats.iterations


# --- the price has to change the answer -------------------------------------


def test_a_heavier_terminal_weight_tracks_the_target_more_closely(
        nmpc, plant, sunny_rows):
    """The planner's inventory targets must actually pull the inner solution.

    With a quadratic terminal penalty the lever is its weight: heavier means the
    NMPC gives up more local profit to land nearer where the plan wants the
    buffers. If the weight does nothing, the hand-off is decorative.
    """
    x0 = _running_state(plant)
    names = plant.state_names()
    # a target the NMPC has to work to reach: more hydrogen than it starts with
    i_h2 = names.index("gas.n_h2")
    targets, prices = _terminal(plant, x0)
    targets["gas.n_h2"] = float(x0[i_h2]) + 400.0

    def solve_at(weight):
        nmpc._warm = None
        nmpc.terminal_weight = weight
        return nmpc.solve(0.0, x0, np.zeros(nmpc.n_steps), sunny_rows,
                          enables=ALL_ON, targets=targets, terminal_prices=prices)

    try:
        light = solve_at(0.1)
        heavy = solve_at(10.0)
    finally:
        nmpc.terminal_weight = 1.0

    assert light.stats.success, light.stats.status
    assert heavy.stats.success, heavy.stats.status

    def miss(sol):
        return abs(sol.states[-1][i_h2] - targets["gas.n_h2"])

    assert miss(heavy) <= miss(light) + 1e-6, (
        "a heavier terminal weight did not pull the end state nearer the target"
    )


def test_free_energy_is_used_by_whatever_can_use_it(nmpc, plant, sunny_rows):
    """Given something worth making, a sunny hour at zero price is absorbed.

    This is the failure the first objective produced: with process power priced
    against an enforced bus balance, curtailing was rewarded, and a closed-loop
    run threw away 36 per cent of the array.

    The assertion is on the *setpoints*, not on curtailment. Curtailment at
    midday is largely structural here -- the electrolyser saturates at ~330 kW
    against ~900 kW available, which is the electrolyser-limited regime the
    sizing audit found past 1100 kWp -- so a curtailment threshold would be
    measuring the plant, not the controller. What the controller must do is run
    whatever *can* absorb free energy, flat out.

    The targets ask for more inventory than the plant starts with, which is what
    a planner target at midday looks like. Told instead to leave the buffers
    where they are, the NMPC shuts down and curtails, and it is right to.
    """
    x0 = _running_state(plant)
    names = plant.state_names()
    targets, prices = _terminal(plant, x0)
    targets["gas.n_h2"] = float(x0[names.index("gas.n_h2")]) + 600.0
    targets["solids.n_caco3"] = float(x0[names.index("solids.n_caco3")]) + 400.0

    nmpc._warm = None
    sol = nmpc.solve(0.0, x0, np.zeros(nmpc.n_steps), sunny_rows,
                     enables=ALL_ON, targets=targets, terminal_prices=prices)
    assert sol.stats.success, sol.stats.status

    # the electrolyser is 72 % of the plant's load and the only large sink for
    # free electricity; the contactor is what banks carbon for later
    assert sol.setpoints["electrolyser"] > 0.8, sol.setpoints
    assert sol.setpoints["contactor"] > 0.8, sol.setpoints
    # and it must not be calcining, which would undo the carbonate it was asked
    # to bank
    assert sol.setpoints["calciner"] < 0.1, sol.setpoints
