"""End-to-end closed-loop runs: conservation, determinism, and sanity bands.

Everything here uses synthetic weather so the suite runs offline and
deterministically. The point is not to validate against a real site -- that is
what the PVGIS runs are for -- but to guarantee that the harness conserves what
it should and that results do not drift silently between refactors.
"""

from __future__ import annotations

import numpy as np
import pytest

from sfp.control.baselines import GreedyController, RuleBasedController
from sfp.economics import Economics
from sfp.models.aggregate import AggregateProcess
from sfp.models.battery import Battery
from sfp.models.pv import PVArray
from sfp.params import load_params
from sfp.report.metrics import compute_metrics
from sfp.sim.plant import Plant
from sfp.sim.simulator import SimulationConfig, simulate
from sfp.weather import pvgis
from sfp.weather.series import Site, WeatherSeries

SITE = Site(latitude=37.39, longitude=-5.99, altitude=10.0, name="test-site")


def build_plant(pv_kwp=500.0, battery_kwh=1000.0, process_kw=350.0) -> Plant:
    return Plant(
        {
            "pv": PVArray(load_params("pv").override(capacity_kwp=pv_kwp)),
            "battery": Battery(load_params("battery").override(capacity_kwh=battery_kwh)),
            "process": AggregateProcess(load_params("aggregate").override(rated_power_kw=process_kw)),
        }
    )


@pytest.fixture(scope="module")
def weather() -> WeatherSeries:
    frame = pvgis.synthetic_tmy(SITE.latitude, SITE.longitude, SITE.altitude, seed=7)
    return WeatherSeries(pvgis.slice_days(frame, 172, 5), SITE)


@pytest.fixture(scope="module")
def config() -> SimulationConfig:
    return SimulationConfig(days=5, start_day=172, dt_s=60.0, control_interval_s=300.0)


@pytest.fixture(scope="module")
def runs(weather, config):
    out = {}
    for controller in (GreedyController(), RuleBasedController()):
        plant = build_plant()
        result = simulate(plant, controller, weather, config)
        out[controller.name] = (result, compute_metrics(result, Economics()))
    return out


# --------------------------------------------------------------------------
# conservation
# --------------------------------------------------------------------------
def test_power_balance_holds_at_every_timestep(runs):
    for name, (result, _) in runs.items():
        log = result.log
        supply = log["pv_used_W"] + log["battery_discharge_W"]
        demand = log["process_power_W"] + log["battery_charge_W"] + log["process_standby_W"] + log["process_warmup_W"]
        # the logged process_power_W is the load term only; overheads are separate
        residual = supply - demand
        assert np.abs(residual).max() < 1.0, f"{name}: bus residual {np.abs(residual).max():.3f} W"


def test_pv_energy_accounting_closes(runs):
    for name, (_, m) in runs.items():
        assert m.pv_used_kwh + m.pv_curtailed_kwh == pytest.approx(m.pv_available_kwh, rel=1e-6)


def test_no_unserved_energy(runs):
    for name, (_, m) in runs.items():
        assert m.unserved_kwh == pytest.approx(0.0, abs=1e-6), name


def test_state_bounds_are_respected(runs):
    for name, (result, _) in runs.items():
        log = result.log
        assert log["battery_soc"].min() >= -1e-9
        assert log["battery_soc"].max() <= 1.0 + 1e-9
        assert log["process_warmth"].min() >= -1e-9
        assert log["process_warmth"].max() <= 1.0 + 1e-9
        assert log["process_load_fraction"].min() >= -1e-9
        assert log["ch4_total_kg"].is_monotonic_increasing


def test_methane_matches_integrated_rate(runs):
    """Cumulative product must equal the integral of the production rate."""
    for name, (result, _) in runs.items():
        log = result.log
        integrated = float(log["ch4_rate_kg_s"].iloc[:-1].sum() * result.dt_s)
        assert log["ch4_total_kg"].iloc[-1] == pytest.approx(integrated, rel=2e-3), name


def test_specific_energy_is_at_least_the_ideal(runs):
    """Real specific energy can never beat the nameplate figure."""
    for name, (result, m) in runs.items():
        ideal = result.plant["process"].p.specific_energy_kwh_per_kg
        assert m.specific_energy_kwh_per_kg >= ideal - 1e-6, name


# --------------------------------------------------------------------------
# behaviour
# --------------------------------------------------------------------------
def test_both_controllers_produce_methane(runs):
    for name, (_, m) in runs.items():
        assert m.ch4_kg > 0.0, name


def test_night_production_is_zero_without_stored_energy(runs):
    """The placeholder chain has no chemical buffer, so darkness plus an empty
    battery must stop production entirely. When M1 lands this test should start
    failing -- that is the point of the buffers."""
    result, _ = runs["greedy"]
    log = result.log
    dark_and_empty = (log["cos_zenith"] <= 0.0) & (log["battery_soc"] <= 0.1001)
    if dark_and_empty.any():
        assert log.loc[dark_and_empty, "ch4_rate_kg_s"].max() < 1e-9


def test_rule_based_never_trips(runs):
    """A competent supervisory controller should not black out its own plant."""
    result, _ = runs["rule-based"]
    assert int(result.log["bus_tripped"].sum()) == 0


def test_rule_based_holds_more_battery_reserve_than_greedy(runs):
    _, greedy = runs["greedy"]
    _, rule = runs["rule-based"]
    assert rule.soc_min >= greedy.soc_min


def test_utilisation_is_a_fraction(runs):
    for name, (_, m) in runs.items():
        assert 0.0 <= m.utilisation <= 1.0, name


def test_efficiency_is_physical(runs):
    for name, (_, m) in runs.items():
        assert 0.0 < m.system_efficiency_lhv < 1.0, name


def test_limiting_subsystem_is_reported(runs):
    for name, (_, m) in runs.items():
        assert m.limiting_subsystem
        assert sum(m.limiting_distribution.values()) == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------
def test_runs_are_deterministic(weather, config):
    a = simulate(build_plant(), GreedyController(), weather, config)
    b = simulate(build_plant(), GreedyController(), weather, config)
    assert a.log["ch4_total_kg"].iloc[-1] == b.log["ch4_total_kg"].iloc[-1]
    assert np.allclose(a.log["battery_soc"].to_numpy(), b.log["battery_soc"].to_numpy())


def test_bigger_array_produces_at_least_as_much(weather, config):
    small = compute_metrics(simulate(build_plant(pv_kwp=400.0), GreedyController(), weather, config))
    large = compute_metrics(simulate(build_plant(pv_kwp=900.0), GreedyController(), weather, config))
    assert large.ch4_kg >= small.ch4_kg


def test_oversized_array_produces_curtailment(weather, config):
    """Curtailment must actually appear once the array outgrows the loads."""
    m = compute_metrics(simulate(build_plant(pv_kwp=2000.0), GreedyController(), weather, config))
    assert m.pv_curtailed_kwh > 0.0
    assert m.curtailment_fraction > 0.05


def test_fault_injection_is_refused_until_implemented(weather, config):
    """Passing a fault schedule before M2 must fail loudly, not silently no-op."""
    with pytest.raises(NotImplementedError):
        simulate(build_plant(), GreedyController(), weather, config, fault_schedule={"any": 1})
