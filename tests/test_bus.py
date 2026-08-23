"""The DC bus must balance. Always, under every request, including absurd ones.

This is the safety layer, so it is tested adversarially: the controllers used
here deliberately ask for impossible things. A bus that can be talked into an
unbalanced dispatch by a bad controller is not a safety layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from sfp.models.aggregate import AggregateProcess
from sfp.models.battery import Battery
from sfp.params import load_params
from sfp.sim.bus import Request, reconcile

DT = 60.0


@pytest.fixture
def parts():
    process = AggregateProcess(load_params("aggregate"))
    battery = Battery(load_params("battery"))
    return process, battery


def _reconcile(parts, request, pv_W, soc=0.5, warmth=1.0):
    process, battery = parts
    return reconcile(
        request,
        pv_available_W=pv_W,
        pv_clipped_W=0.0,
        process=process,
        process_state=np.array([warmth, 0.0, 0.0]),
        battery=battery,
        battery_state=np.array([soc, 0.0, 0.0]),
        dt_s=DT,
    )


@pytest.mark.parametrize("pv_W", [0.0, 1e3, 5e4, 2e5, 4e5, 1e7])
@pytest.mark.parametrize("soc", [0.10, 0.30, 0.50, 0.95])
@pytest.mark.parametrize(
    "request_kwargs",
    [
        {},
        {"load_fraction": 1.0},
        {"load_fraction": 1.0, "battery_charge_W": 1e9},
        {"load_fraction": 1.0, "battery_discharge_W": 1e9},
        {"load_fraction": 5.0},  # out of range on purpose
        {"load_fraction": -3.0},
        {"load_fraction": 0.5, "curtail_fraction": 1.0},
        {"load_fraction": 0.5, "curtail_fraction": 0.5, "battery_charge_W": 1e9},
        {"load_fraction": 1.0, "enable": 0.0},
    ],
)
def test_bus_always_balances(parts, pv_W, soc, request_kwargs):
    """Whatever is asked for, supply must equal demand."""
    dispatch = _reconcile(parts, Request(**request_kwargs), pv_W, soc=soc)
    assert abs(dispatch.balance_residual_W) < 1e-3


@pytest.mark.parametrize("pv_W", [0.0, 1e4, 3e5])
@pytest.mark.parametrize("soc", [0.10, 0.5, 0.95])
def test_bus_never_exceeds_battery_limits(parts, pv_W, soc):
    process, battery = parts
    state = np.array([soc, 0.0, 0.0])
    dispatch = _reconcile(
        parts, Request(load_fraction=1.0, battery_charge_W=1e9, battery_discharge_W=1e9), pv_W, soc=soc
    )
    assert dispatch.battery_charge_W <= battery.max_charge_power_W(state, DT) + 1e-6
    assert dispatch.battery_discharge_W <= battery.max_discharge_power_W(state, DT) + 1e-6
    assert dispatch.battery_charge_W <= battery.max_power_W + 1e-6
    assert dispatch.battery_discharge_W <= battery.max_power_W + 1e-6


def test_bus_curtails_when_nothing_can_absorb(parts):
    """Full battery, process at rating, huge array -> the remainder is curtailed."""
    pv_W = 5e6
    dispatch = _reconcile(
        parts, Request(load_fraction=1.0, battery_charge_W=1e9), pv_W, soc=0.95
    )
    assert dispatch.pv_curtailed_W > 0.0
    assert dispatch.pv_used_W + dispatch.pv_curtailed_W == pytest.approx(pv_W)


def test_curtailment_accounting_always_closes(parts):
    for pv_W in (0.0, 1e4, 1e5, 1e6):
        for soc in (0.1, 0.6, 0.95):
            d = _reconcile(parts, Request(load_fraction=0.6, battery_charge_W=1e5), pv_W, soc=soc)
            assert d.pv_used_W + d.pv_curtailed_W == pytest.approx(d.pv_available_W, abs=1e-6)


def test_bus_trips_when_parasitics_cannot_be_served(parts):
    """No sun, empty battery -> the plant goes dark rather than running a debt."""
    dispatch = _reconcile(parts, Request(load_fraction=1.0), pv_W=0.0, soc=0.10)
    assert dispatch.tripped
    assert dispatch.process_enable == 0.0
    assert dispatch.process_power_W == pytest.approx(0.0, abs=1e-3)
    assert dispatch.unserved_W == pytest.approx(0.0)


def test_bus_does_not_trip_when_parasitics_are_affordable(parts):
    process, _ = parts
    parasitic = process.parasitic_power_W(np.array([1.0, 0.0, 0.0]))
    dispatch = _reconcile(parts, Request(load_fraction=0.0), pv_W=parasitic * 2.0, soc=0.10)
    assert not dispatch.tripped
    assert dispatch.process_enable == 1.0


def test_no_unserved_energy_in_any_configuration(parts):
    """After the trip logic, unserved load should never appear."""
    for pv_W in (0.0, 5e2, 5e3, 5e4, 5e5):
        for soc in (0.10, 0.1001, 0.3, 0.95):
            for load in (0.0, 0.15, 0.6, 1.0):
                d = _reconcile(parts, Request(load_fraction=load), pv_W, soc=soc)
                assert d.unserved_W == pytest.approx(0.0, abs=1e-6), (pv_W, soc, load)


def test_bus_sheds_load_rather_than_overdrawing(parts):
    """A load request beyond supply is reduced, not served from nowhere."""
    dispatch = _reconcile(parts, Request(load_fraction=1.0), pv_W=4e4, soc=0.101)
    assert dispatch.process_load_fraction < 1.0
    assert dispatch.load_shed_W > 0.0
    assert dispatch.intervened
    assert abs(dispatch.balance_residual_W) < 1e-3


def test_bus_honours_a_deliberate_refusal_to_charge(parts):
    """Asking for no charging must curtail instead -- a legitimate economic choice."""
    dispatch = _reconcile(parts, Request(load_fraction=0.0, battery_charge_W=0.0), pv_W=2e5, soc=0.5)
    assert dispatch.battery_charge_W == pytest.approx(0.0)
    assert dispatch.pv_curtailed_W > 0.0


def test_bus_respects_explicit_disable(parts):
    dispatch = _reconcile(parts, Request(load_fraction=1.0, enable=0.0), pv_W=5e5, soc=0.9)
    assert dispatch.process_enable == 0.0
    assert dispatch.process_power_W == pytest.approx(0.0, abs=1e-3)
    assert dispatch.pv_curtailed_W > 0.0


def test_below_minimum_load_request_is_snapped_off(parts):
    process, _ = parts
    dispatch = _reconcile(
        parts, Request(load_fraction=0.5 * process.p.min_load_fraction), pv_W=3e5, soc=0.6
    )
    assert dispatch.process_load_fraction == 0.0


def test_clean_request_is_not_flagged_as_intervention(parts):
    """A feasible, modest request must pass through untouched.

    The intervention counter is a diagnostic for controller quality, so routine
    clamping of a charge request must not inflate it.
    """
    dispatch = _reconcile(
        parts, Request(load_fraction=0.5, battery_charge_W=5e4), pv_W=3e5, soc=0.5
    )
    assert not dispatch.intervened
    assert dispatch.load_shed_W == 0.0
