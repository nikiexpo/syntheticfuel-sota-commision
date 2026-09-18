"""The DC bus must balance. Always, under every request, including absurd ones.

This is the safety layer, so it is tested adversarially: the requests here
deliberately ask for impossible things. A bus that can be talked into an
unbalanced dispatch by a bad controller is not a safety layer.

The bus also arbitrates between four competing loads and copes with a draw it
cannot command at all (the CO2 compressor follows the calcination rate), so the
coverage below is about arbitration as much as arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from sfp.cli import build_plant, build_reference_plant
from sfp.sim.bus import CONTROLLABLE, SHED_ORDER, Request, reconcile

DT = 60.0
WEATHER = {
    # plant.evaluate() runs every subsystem including the PV array
    "poa_global": 600.0,
    "ghi": 550.0,
    "temp_air": 20.0,
    "relative_humidity": 60.0,
    "pressure": 101325.0,
    "wind_speed": 2.0,
    "cos_zenith": 0.5,
}


@pytest.fixture(scope="module")
def plant():
    return build_reference_plant()


def _state(plant, soc: float = 0.5, kiln_K: float = 1173.15, reactor_K: float = 573.15):
    """A plant state with the buffers comfortably stocked."""
    x = plant.initial_state()
    sl = plant.split(x)
    sl["battery"][0] = soc
    sl["calciner"][0] = kiln_K
    sl["sabatier"][0] = reactor_K
    sl["gas"][0] = 0.5 * plant["gas"].p.h2_capacity_mol
    sl["gas"][1] = 0.5 * plant["gas"].p.co2_capacity_mol
    return plant.join(sl)


def _reconcile(plant, request, pv_W, soc=0.5, **kwargs):
    return reconcile(
        request,
        plant=plant,
        state=_state(plant, soc=soc, **kwargs),
        weather=WEATHER,
        pv_available_W=pv_W,
        pv_clipped_W=0.0,
        dt_s=DT,
    )


# --------------------------------------------------------------------------
# balance
# --------------------------------------------------------------------------
@pytest.mark.parametrize("pv_W", [0.0, 5e3, 5e4, 2e5, 5e5, 1e7])
@pytest.mark.parametrize("soc", [0.10, 0.30, 0.60, 0.95])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"setpoints": {k: 1.0 for k in CONTROLLABLE}},
        {"setpoints": {k: 1.0 for k in CONTROLLABLE}, "battery_charge_W": 1e9},
        {"setpoints": {k: 1.0 for k in CONTROLLABLE}, "battery_discharge_W": 1e9},
        {"setpoints": {k: 5.0 for k in CONTROLLABLE}},   # out of range on purpose
        {"setpoints": {k: -3.0 for k in CONTROLLABLE}},
        {"setpoints": {k: 0.5 for k in CONTROLLABLE}, "curtail_fraction": 1.0},
        {"enables": {k: 0.0 for k in CONTROLLABLE}},
    ],
)
def test_bus_always_balances(plant, pv_W, soc, kwargs):
    kwargs = dict(kwargs)
    kwargs.setdefault("enables", {k: 1.0 for k in CONTROLLABLE})
    kwargs.setdefault("battery_discharge_W", 1e9)
    dispatch = _reconcile(plant, Request(**kwargs), pv_W, soc=soc)
    assert abs(dispatch.balance_residual_W) < 1e-3


def test_curtailment_accounting_always_closes(plant):
    for pv_W in (0.0, 1e4, 1e5, 1e6):
        for soc in (0.1, 0.6, 0.95):
            d = _reconcile(
                plant,
                Request.uniform(0.6, battery_charge_W=1e5, battery_discharge_W=1e9),
                pv_W,
                soc=soc,
            )
            assert d.pv_used_W + d.pv_curtailed_W == pytest.approx(d.pv_available_W, abs=1e-6)


def test_bus_never_exceeds_battery_limits(plant):
    battery = plant["battery"]
    for pv_W in (0.0, 1e4, 3e5):
        for soc in (0.10, 0.5, 0.95):
            state = plant.split(_state(plant, soc=soc))["battery"]
            d = _reconcile(
                plant,
                Request.uniform(1.0, battery_charge_W=1e9, battery_discharge_W=1e9),
                pv_W,
                soc=soc,
            )
            assert d.battery_charge_W <= battery.max_charge_power_W(state, DT) + 1e-6
            assert d.battery_discharge_W <= battery.max_discharge_power_W(state, DT) + 1e-6


def test_bus_curtails_when_nothing_can_absorb(plant):
    d = _reconcile(plant, Request.uniform(1.0, battery_charge_W=1e9), 5e6, soc=0.95)
    assert d.pv_curtailed_W > 0.0
    assert d.pv_used_W + d.pv_curtailed_W == pytest.approx(5e6)


# --------------------------------------------------------------------------
# trips and shedding
# --------------------------------------------------------------------------
def test_bus_trips_when_parasitics_cannot_be_served(plant):
    """No sun, empty battery -> subsystems go dark rather than running a debt."""
    d = _reconcile(plant, Request.uniform(1.0, battery_discharge_W=1e9), 0.0, soc=0.10)
    assert d.tripped
    assert len(d.tripped_subsystems) > 0
    assert d.unserved_W == pytest.approx(0.0, abs=1e-3)


def test_trip_order_follows_shed_priority(plant):
    """The calciner goes first and the reactor last -- the reactor protects
    product in progress and a 31 kWh relight."""
    d = _reconcile(plant, Request.uniform(1.0, battery_discharge_W=1e9), 0.0, soc=0.10)
    if len(d.tripped_subsystems) < len(SHED_ORDER):
        expected = SHED_ORDER[: len(d.tripped_subsystems)]
        assert d.tripped_subsystems == expected


def test_reactor_survives_a_shortage_that_kills_the_calciner(plant):
    """With just enough power for the reactor, the kiln is what gets shed."""
    reactor_only = plant["sabatier"].p.auxiliary_power_kw * 1e3 * 1.5
    d = _reconcile(plant, Request.uniform(1.0, battery_discharge_W=1e9), reactor_only, soc=0.10)
    assert d.enables["sabatier"] >= 0.5
    assert d.setpoints["calciner"] == pytest.approx(0.0)


def test_no_unserved_energy_in_any_configuration(plant):
    for pv_W in (0.0, 5e2, 5e3, 5e4, 5e5):
        for soc in (0.10, 0.1001, 0.3, 0.95):
            for setpoint in (0.0, 0.3, 1.0):
                d = _reconcile(
                    plant,
                    Request.uniform(setpoint, battery_discharge_W=1e9),
                    pv_W,
                    soc=soc,
                )
                assert d.unserved_W == pytest.approx(0.0, abs=1e-3), (pv_W, soc, setpoint)


def test_bus_sheds_rather_than_overdrawing(plant):
    d = _reconcile(plant, Request.uniform(1.0, battery_discharge_W=1e9), 4e4, soc=0.101)
    assert d.total_load_W <= d.pv_used_W + d.battery_discharge_W + 1e-3
    assert d.intervened


def test_total_load_includes_the_uncommanded_compressor(plant):
    """The CO2 compressor draw follows the calcination rate and cannot be
    commanded, so the bus must account for it in the balance."""
    d = _reconcile(plant, Request.uniform(1.0, battery_discharge_W=1e9), 1e6, soc=0.6)
    assert "gas" in d.loads_W
    if d.setpoints["calciner"] > 0.1:
        assert d.loads_W["gas"] > 0.0
    assert d.total_load_W == pytest.approx(sum(d.loads_W.values()), rel=1e-9)


# --------------------------------------------------------------------------
# honouring controller intent
# --------------------------------------------------------------------------
def test_bus_honours_a_requested_discharge_cap(plant):
    """The requested discharge is a cap, not a demand -- this is what lets a
    controller hold an evening reserve. Ignoring it defeats every reserve policy
    and blacks the plant out at 3 a.m. with charge still in the pack."""
    capped = _reconcile(
        plant, Request.uniform(1.0, battery_discharge_W=20e3), 0.0, soc=0.8
    )
    assert capped.battery_discharge_W <= 20e3 + 1e-6

    uncapped = _reconcile(
        plant, Request.uniform(1.0, battery_discharge_W=1e9), 0.0, soc=0.8
    )
    assert uncapped.battery_discharge_W > capped.battery_discharge_W


def test_bus_honours_a_deliberate_refusal_to_charge(plant):
    """Asking for no charging must curtail instead -- legitimate when the wear
    cost of a cycle exceeds the value of the stored energy."""
    d = _reconcile(
        plant, Request(setpoints={k: 0.0 for k in CONTROLLABLE},
                       enables={k: 0.0 for k in CONTROLLABLE},
                       battery_charge_W=0.0), 2e5, soc=0.5
    )
    assert d.battery_charge_W == pytest.approx(0.0)
    assert d.pv_curtailed_W > 0.0


def test_bus_respects_explicit_disable(plant):
    d = _reconcile(
        plant,
        Request(setpoints={k: 1.0 for k in CONTROLLABLE},
                enables={k: 0.0 for k in CONTROLLABLE},
                battery_discharge_W=1e9),
        5e5,
        soc=0.9,
    )
    assert all(v < 0.5 for v in d.enables.values())
    assert d.total_load_W == pytest.approx(0.0, abs=1e-3)
    assert d.pv_curtailed_W > 0.0


def test_electrolyser_minimum_load_is_never_violated(plant):
    """Gas crossover is a safety limit: the stack runs above it or not at all."""
    minimum = plant["electrolyser"].min_setpoint
    for pv_W in np.linspace(0.0, 5e5, 25):
        d = _reconcile(plant, Request.uniform(1.0, battery_discharge_W=1e9), float(pv_W), soc=0.5)
        value = d.setpoints["electrolyser"]
        assert value == 0.0 or value >= minimum - 1e-9


def test_clean_request_is_not_flagged_as_intervention(plant):
    """A feasible, modest request must pass through untouched -- the intervention
    counter is a diagnostic for controller quality."""
    d = _reconcile(
        plant, Request.uniform(0.3, battery_charge_W=5e4, battery_discharge_W=1e9), 6e5, soc=0.5
    )
    assert not d.intervened
    assert d.shed_W == pytest.approx(0.0, abs=1.0)
