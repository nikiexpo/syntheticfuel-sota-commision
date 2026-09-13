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

from sfp.cli import build_plant
from sfp.control.nmpc import ENABLE_INDEX, SETPOINT_INDEX, InnerNMPC
from sfp.economics import Economics

ALL_ON = {"contactor": 1.0, "calciner": 1.0, "electrolyser": 1.0, "sabatier": 1.0}


@pytest.fixture(scope="module")
def plant():
    return build_plant(1100.0, 1500.0, 750.0)


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
    sol = nmpc.solve(0.0, _running_state(plant), np.full(nmpc.n_steps, 0.05),
                     sunny_rows, enables=ALL_ON)
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
    sol = nmpc.solve(0.0, _running_state(plant), np.full(nmpc.n_steps, 0.05),
                     sunny_rows, enables={**ALL_ON, "calciner": 0.0})
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
    sol = nmpc.solve(0.0, _running_state(plant), np.full(nmpc.n_steps, 0.05),
                     sunny_rows, enables=ALL_ON)
    assert sol.stats.success, sol.stats.status
    assert min(sol.battery_charge_W, sol.battery_discharge_W) < 1e3


def test_it_converges_and_warm_starts_cheaply(nmpc, plant, sunny_rows):
    """A cold solve must converge, and the next must not cost more."""
    x0 = _running_state(plant)
    prices = np.full(nmpc.n_steps, 0.05)
    nmpc._warm = None
    first = nmpc.solve(0.0, x0, prices, sunny_rows, enables=ALL_ON)
    second = nmpc.solve(0.0, x0, prices, sunny_rows, enables=ALL_ON)
    assert first.stats.success and second.stats.success
    assert second.stats.iterations <= first.stats.iterations


# --- the price has to change the answer -------------------------------------


def test_a_higher_price_buys_less_energy(nmpc, plant, sunny_rows):
    """The whole architecture rests on this: lambda must change behaviour.

    If the setpoints are identical at both prices then the price is decorative
    and the hierarchy is a hand-off in name only.

    The two prices bracket what the planner actually publishes -- it computes
    0.00 EUR/kWh at midday when the plant is curtailing and about 0.054
    overnight. Testing at an extreme instead (0.5, ten times realistic) drives
    the answer to "shut everything down", which is a corner solution that an
    interior-point method converges to poorly and which tells us nothing about
    behaviour in the range the plant ever sees.

    Terminal prices are supplied, and they are not decoration either. Without
    them nothing in a thirty-minute horizon rewards making hydrogen -- the
    methane it becomes is hours away -- so the electrolyser setpoint sits on a
    flat manifold and the solver returns an arbitrary point on it. Measured that
    way the comparison is noise, and it duly came back backwards. The terminal
    value is what makes the local problem well-posed, which is exactly why the
    hierarchy hands one down.
    """
    x0 = _running_state(plant)
    names = plant.state_names()
    targets = {n: float(x0[names.index(n)]) for n in
               ("gas.n_h2", "gas.n_co2", "solids.n_caco3", "battery.soc")}
    price_per_mol_ch4 = Economics().p.methane_price_per_kg * 16.0425e-3
    terminal = {
        "gas.n_h2": price_per_mol_ch4 / 4.0,
        "gas.n_co2": price_per_mol_ch4,
        "solids.n_caco3": price_per_mol_ch4,
        "battery.soc": price_per_mol_ch4 * (
            plant["battery"].nominal_energy_J / 3.6e6) / 56.4 / 2.016e-3 / 4.0,
    }

    def solve_at(price):
        nmpc._warm = None
        return nmpc.solve(0.0, x0, np.full(nmpc.n_steps, price), sunny_rows,
                          enables=ALL_ON, targets=targets,
                          terminal_prices=terminal)

    cheap = solve_at(0.005)
    dear = solve_at(0.06)
    assert cheap.stats.success, cheap.stats.status
    assert dear.stats.success, dear.stats.status

    def process_load(sol):
        return sum(sol.setpoints[k]
                   for k in ("calciner", "electrolyser", "contactor"))

    assert process_load(dear) <= process_load(cheap) + 1e-6, (
        "raising the price of electricity did not reduce the energy bought"
    )
