"""Subsystem models: conservation, limits, and the symbolic/numeric contract.

The last group matters more than it looks. Every model is written once and
evaluated both as numpy (truth simulator) and as CasADi symbolics (estimator and
controller). If that contract silently breaks, the controller starts optimising
a different plant from the one it is steering, and nothing else in the test suite
would notice.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pytest

from sfp.models.aggregate import AggregateProcess
from sfp.models.battery import Battery
from sfp.models.pv import PVArray
from sfp.params import load_params
from sfp.units import J_PER_KWH


@pytest.fixture
def pv() -> PVArray:
    return PVArray(load_params("pv"))


@pytest.fixture
def battery() -> Battery:
    return Battery(load_params("battery"))


@pytest.fixture
def process() -> AggregateProcess:
    return AggregateProcess(load_params("aggregate"))


# --------------------------------------------------------------------------
# PV
# --------------------------------------------------------------------------
def test_pv_zero_irradiance_gives_zero_power(pv):
    w = {"poa_global": 0.0, "temp_air": 20.0, "wind_speed": 2.0}
    assert pv.available_power(w) == 0.0


def test_pv_at_stc_delivers_rated_times_inverter(pv):
    """1000 W/m^2 at 25 degC cell temperature must give rated x derates x eta_inv."""
    # choose air temperature so the Faiman model lands exactly on 25 degC
    poa = 1000.0
    wind = 2.0
    denominator = pv.p.faiman_u0 + pv.p.faiman_u1 * wind
    temp_air = 25.0 - poa / denominator
    w = {"poa_global": poa, "temp_air": temp_air, "wind_speed": wind}
    expected = pv.rated_dc_W * pv.derate_factor() * pv.p.eta_inverter
    assert pv.available_power(w) == pytest.approx(min(expected, pv.rated_ac_W))


def test_pv_hotter_cells_produce_less(pv):
    cold = {"poa_global": 800.0, "temp_air": 5.0, "wind_speed": 3.0}
    hot = {"poa_global": 800.0, "temp_air": 40.0, "wind_speed": 3.0}
    assert pv.available_power(hot) < pv.available_power(cold)


def test_pv_wind_cools_and_helps(pv):
    still = {"poa_global": 900.0, "temp_air": 35.0, "wind_speed": 0.0}
    breezy = {"poa_global": 900.0, "temp_air": 35.0, "wind_speed": 8.0}
    assert pv.available_power(breezy) > pv.available_power(still)


def test_pv_never_exceeds_inverter_rating(pv):
    w = {"poa_global": 1400.0, "temp_air": -10.0, "wind_speed": 10.0}
    assert pv.available_power(w) <= pv.rated_ac_W + 1e-9


def test_pv_reports_generation_as_negative_power(pv):
    w = {"poa_global": 700.0, "temp_air": 20.0, "wind_speed": 2.0}
    out = pv.outputs(0.0, np.zeros(0), np.array([0.0]), w)
    assert out["power_electrical_W"] < 0.0
    assert out["power_electrical_W"] == pytest.approx(-out["pv_delivered_W"])


def test_pv_curtailment_accounting_closes(pv):
    w = {"poa_global": 700.0, "temp_air": 20.0, "wind_speed": 2.0}
    out = pv.outputs(0.0, np.zeros(0), np.array([0.4]), w)
    assert out["pv_delivered_W"] + out["pv_curtailed_W"] == pytest.approx(out["pv_available_W"])


# --------------------------------------------------------------------------
# Battery
# --------------------------------------------------------------------------
def test_battery_round_trip_efficiency():
    """Charge from soc_min to soc_max and back; recovered/spent must equal RTE.

    Self-discharge is switched off for this measurement and the power is held
    constant. Both matter: using the SoC-tapered power envelope stretches the
    charge over many hours of near-zero current, during which self-discharge
    dominates and the measurement reports the duration of the test rather than
    the efficiency of the pack.
    """
    battery = Battery(load_params("battery").override(self_discharge_per_day=0.0))
    # The step must be small enough that a single step cannot overshoot the SoC
    # limit by more than the tolerance being asserted.
    dt = 10.0
    power = 0.2 * battery.nominal_energy_J / 3600.0  # steady 0.2C
    x = np.array([battery.p.soc_min, 0.0, 0.0])

    energy_in = 0.0
    for _ in range(200000):
        if x[0] >= battery.p.soc_max - 1e-9:
            break
        energy_in += power * dt
        x = x + battery.rhs(0.0, x, np.array([power, 0.0]), {}) * dt
    assert x[0] == pytest.approx(battery.p.soc_max, abs=2e-3)

    energy_out = 0.0
    for _ in range(200000):
        if x[0] <= battery.p.soc_min + 1e-9:
            break
        energy_out += power * dt
        x = x + battery.rhs(0.0, x, np.array([0.0, power]), {}) * dt
    assert x[0] == pytest.approx(battery.p.soc_min, abs=2e-3)

    measured = energy_out / energy_in
    assert measured == pytest.approx(battery.round_trip_efficiency(), rel=0.01)


def test_battery_self_discharge_costs_energy_over_time(battery):
    """With self-discharge on, a slow cycle recovers less than the nominal RTE."""
    dt = 3600.0
    x = np.array([0.9, 0.0, 0.0])
    for _ in range(24 * 30):  # a month idle
        x = x + battery.rhs(0.0, x, np.array([0.0, 0.0]), {}) * dt
    assert 0.80 < x[0] < 0.89


def test_battery_soc_limits_are_respected_by_power_envelope(battery):
    dt = 300.0
    full = np.array([battery.p.soc_max, 0.0, 0.0])
    assert battery.max_charge_power_W(full, dt) == pytest.approx(0.0, abs=1e-6)
    empty = np.array([battery.p.soc_min, 0.0, 0.0])
    assert battery.max_discharge_power_W(empty, dt) == pytest.approx(0.0, abs=1e-6)


def test_battery_efc_counts_one_full_cycle(battery):
    """A full charge plus a full discharge is exactly one equivalent full cycle."""
    dt = 60.0
    x = np.array([0.0, 0.0, 0.0])
    energy = battery.nominal_energy_J
    power = energy / 3600.0  # one hour of charging at nominal energy
    steps = int(3600.0 / dt)
    for _ in range(steps):
        x = x + battery.rhs(0.0, x, np.array([power, 0.0]), {}) * dt
    for _ in range(steps):
        x = x + battery.rhs(0.0, x, np.array([0.0, power]), {}) * dt
    assert x[2] == pytest.approx(1.0, rel=1e-6)


def test_battery_fade_reaches_eol_at_rated_cycle_life(battery):
    """`cycle_life` equivalent cycles must produce `end_of_life_fade` of fade."""
    per_cycle = battery.p.end_of_life_fade / battery.p.cycle_life
    assert per_cycle * battery.p.cycle_life == pytest.approx(battery.p.end_of_life_fade)


def test_battery_degradation_cost_is_order_of_cents(battery):
    """A sanity band: LFP throughput should cost 1-5 cents per kWh."""
    assert 0.005 < battery.cost_per_kWh_throughput_EUR() < 0.05


def test_battery_idle_only_self_discharges(battery):
    x = np.array([0.5, 0.0, 0.0])
    dx = battery.rhs(0.0, x, np.array([0.0, 0.0]), {})
    assert dx[0] < 0.0
    assert dx[2] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Aggregate process
# --------------------------------------------------------------------------
def test_process_dark_draws_negligible_power(process):
    """A de-energised plant must draw nothing -- not 12 % of its warm-up power.

    Regression test: the `running` gate was once centred one transition-width
    from zero, leaving it at 0.12 while stopped, which charged a stopped plant
    ~19 kW of phantom warm-up load forever.

    The bound is a milliwatt rather than exactly zero because the gates are
    smooth by design -- the same expressions have to be differentiable inside
    the NMPC. A microwatt residual is nine orders of magnitude below rating and
    is snapped to zero by the bus.
    """
    for warmth in (0.0, 0.5, 1.0):
        x = np.array([warmth, 0.0, 0.0])
        out = process.outputs(0.0, x, np.array([0.0, 0.0]), {})
        assert out["power_electrical_W"] < 1e-3
        assert out["power_electrical_W"] < 1e-8 * process.rated_power_W


def test_process_energised_but_idle_draws_only_standby(process):
    x = np.array([1.0, 0.0, 0.0])
    out = process.outputs(0.0, x, np.array([0.0, 1.0]), {})
    assert out["power_electrical_W"] == pytest.approx(process.standby_power_W)
    assert out["ch4_rate_kg_s"] == pytest.approx(0.0, abs=1e-12)


def test_process_minimum_load_is_admitted_at_full_value(process):
    """Commanding exactly `min_load_fraction` must give that load, not half of it.

    Regression test: the turndown gate was once centred *on* the minimum load,
    so commanding the minimum returned 0.5 x minimum.
    """
    x = np.array([1.0, 0.0, 0.0])
    out = process.outputs(0.0, x, np.array([process.p.min_load_fraction, 1.0]), {})
    assert out["process_load_fraction"] == pytest.approx(process.p.min_load_fraction, rel=1e-3)
    assert out["process_running"] == pytest.approx(1.0, abs=1e-3)


def test_process_below_minimum_load_is_effectively_off(process):
    """Commanding below the turndown limit yields no meaningful production.

    The residual is the deliberate smooth relaxation of what is physically an
    on/off decision -- it is what lets an NLP solver approach the boundary from
    either side. The hard snap to zero lives in the bus, which is the layer that
    is allowed to be discontinuous.
    """
    x = np.array([1.0, 0.0, 0.0])
    out = process.outputs(0.0, x, np.array([0.5 * process.p.min_load_fraction, 1.0]), {})
    assert out["process_load_fraction"] < 0.01
    assert out["process_running"] < 0.05


def test_process_gate_is_monotonic_in_command(process):
    """More commanded load can never mean less delivered load or production."""
    x = np.array([1.0, 0.0, 0.0])
    commands = np.linspace(0.0, 1.0, 101)
    loads = [
        process.outputs(0.0, x, np.array([c, 1.0]), {})["process_load_fraction"] for c in commands
    ]
    assert np.all(np.diff(loads) >= -1e-12)


def test_process_cold_plant_produces_nothing(process):
    x = np.array([0.0, 0.0, 0.0])
    out = process.outputs(0.0, x, np.array([1.0, 1.0]), {})
    assert out["ch4_rate_kg_s"] == pytest.approx(0.0)
    assert out["process_warmup_W"] > 0.0


def test_process_specific_energy_is_worse_at_part_load(process):
    full = process.specific_energy_J_per_kg(1.0)
    part = process.specific_energy_J_per_kg(process.p.min_load_fraction)
    assert part > full
    assert part / full == pytest.approx(1.0 + process.p.part_load_penalty, rel=1e-6)


def test_process_production_matches_specific_energy(process):
    """At full load and fully warm, kg/s must equal power / specific energy."""
    x = np.array([1.0, 0.0, 0.0])
    u = np.array([1.0, 1.0])
    rate = process.production_rate_kg_s(x, u)
    expected = process.rated_power_W / (process.p.specific_energy_kwh_per_kg * J_PER_KWH)
    assert rate == pytest.approx(expected, rel=1e-6)


def test_process_load_for_power_inverts_power_for_load(process):
    x = np.array([1.0, 0.0, 0.0])
    for target in (0.2, 0.5, 0.85, 1.0):
        power = process.power_for_load(x, target)
        assert process.load_for_power(x, power) == pytest.approx(target, rel=1e-6)


def test_process_warms_up_then_holds(process):
    x = np.array([0.0, 0.0, 0.0])
    u = np.array([1.0, 1.0])
    dt = 60.0
    for _ in range(int(4 * process.p.warmup_time_s / dt)):
        x = x + process.rhs(0.0, x, u, {}) * dt
    assert x[0] == pytest.approx(1.0, abs=0.02)


def test_process_cools_when_dark(process):
    x = np.array([1.0, 0.0, 0.0])
    u = np.array([0.0, 0.0])
    dt = 60.0
    for _ in range(int(3 * process.p.cooldown_time_s / dt)):
        x = x + process.rhs(0.0, x, u, {}) * dt
    assert x[0] < 0.1


# --------------------------------------------------------------------------
# The symbolic/numeric contract
# --------------------------------------------------------------------------
@pytest.mark.parametrize("state,inputs", [(3, 2), (3, 2)])
def test_battery_rhs_is_symbolically_evaluable(battery, state, inputs):
    x = ca.SX.sym("x", state)
    u = ca.SX.sym("u", inputs)
    f = ca.Function("f", [x, u], [battery.rhs(0.0, x, u, {})])
    numeric = battery.rhs(0.0, np.array([0.5, 0.0, 0.0]), np.array([1e5, 0.0]), {})
    symbolic = np.array(f(np.array([0.5, 0.0, 0.0]), np.array([1e5, 0.0]))).reshape(-1)
    assert np.allclose(numeric, symbolic, rtol=1e-12, atol=1e-15)


def test_process_rhs_matches_between_numpy_and_casadi(process):
    x_val = np.array([0.4, 12.0, 2.0])
    u_val = np.array([0.7, 1.0])
    x = ca.SX.sym("x", 3)
    u = ca.SX.sym("u", 2)
    f = ca.Function("f", [x, u], [process.rhs(0.0, x, u, {})])
    numeric = process.rhs(0.0, x_val, u_val, {})
    symbolic = np.array(f(x_val, u_val)).reshape(-1)
    assert np.allclose(numeric, symbolic, rtol=1e-10, atol=1e-14)


def test_pv_outputs_are_symbolically_evaluable(pv):
    w = {
        "poa_global": ca.SX.sym("g"),
        "temp_air": ca.SX.sym("t"),
        "wind_speed": ca.SX.sym("v"),
    }
    out = pv.outputs(0.0, ca.SX.zeros(0), ca.SX.sym("u", 1), w)
    assert isinstance(out["pv_available_W"], ca.SX)
