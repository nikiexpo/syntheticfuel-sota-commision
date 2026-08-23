"""Subsystem models: conservation, limits, thermodynamics, and the symbolic contract.

Many of these pin numbers that were *verified* during development rather than
assumed -- the calcination threshold, the electrolyser's part-load efficiency
peak, the Sabatier reactor's self-sustaining operating point. Those are the
claims the writeup rests on, so they are the ones a refactor must not silently
break.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pytest

from sfp.models.battery import Battery
from sfp.models.buffers import GasBuffer, WaterTank
from sfp.models.calciner import Calciner
from sfp.models.contactor import AirContactor
from sfp.models.electrolyser import Electrolyser
from sfp.models.pv import PVArray
from sfp.models.sabatier import SabatierReactor
from sfp.models.solids import SolidsInventory
from sfp.params import load_params
from sfp.units import M_H2, R_GAS


@pytest.fixture
def pv():
    return PVArray(load_params("pv"))


@pytest.fixture
def battery():
    return Battery(load_params("battery"))


@pytest.fixture
def solids():
    return SolidsInventory(load_params("solids"))


@pytest.fixture
def contactor():
    return AirContactor(load_params("contactor"))


@pytest.fixture
def calciner():
    return Calciner(load_params("calciner"))


@pytest.fixture
def gas():
    return GasBuffer(load_params("buffers"))


@pytest.fixture
def water():
    return WaterTank(load_params("buffers"))


@pytest.fixture
def electrolyser():
    return Electrolyser(load_params("electrolyser"))


@pytest.fixture
def sabatier():
    return SabatierReactor(load_params("sabatier"))


AMBIENT = {"temp_air": 20.0, "relative_humidity": 60.0, "pressure": 101325.0, "wind_speed": 2.0}


# --------------------------------------------------------------------------
# PV
# --------------------------------------------------------------------------
def test_pv_zero_irradiance_gives_zero_power(pv):
    assert pv.available_power({"poa_global": 0.0, "temp_air": 20.0, "wind_speed": 2.0}) == 0.0


def test_pv_at_stc_delivers_rated_times_inverter(pv):
    poa, wind = 1000.0, 2.0
    temp_air = 25.0 - poa / (pv.p.faiman_u0 + pv.p.faiman_u1 * wind)
    expected = pv.rated_dc_W * pv.derate_factor() * pv.p.eta_inverter
    got = pv.available_power({"poa_global": poa, "temp_air": temp_air, "wind_speed": wind})
    assert got == pytest.approx(min(expected, pv.rated_ac_W))


def test_pv_hotter_cells_produce_less(pv):
    cold = pv.available_power({"poa_global": 800.0, "temp_air": 5.0, "wind_speed": 3.0})
    hot = pv.available_power({"poa_global": 800.0, "temp_air": 40.0, "wind_speed": 3.0})
    assert hot < cold


def test_pv_never_exceeds_inverter_rating(pv):
    assert pv.available_power(
        {"poa_global": 1400.0, "temp_air": -10.0, "wind_speed": 10.0}
    ) <= pv.rated_ac_W + 1e-9


def test_pv_reports_generation_as_negative_power(pv):
    out = pv.outputs(0.0, np.zeros(0), np.array([0.0]), {"poa_global": 700.0, "temp_air": 20.0, "wind_speed": 2.0})
    assert out["power_electrical_W"] == pytest.approx(-out["pv_delivered_W"])


# --------------------------------------------------------------------------
# Battery
# --------------------------------------------------------------------------
def test_battery_round_trip_efficiency():
    battery = Battery(load_params("battery").override(self_discharge_per_day=0.0))
    dt = 10.0
    power = 0.2 * battery.nominal_energy_J / 3600.0
    x = np.array([battery.p.soc_min, 0.0, 0.0])

    energy_in = 0.0
    while x[0] < battery.p.soc_max - 1e-9:
        energy_in += power * dt
        x = x + battery.rhs(0.0, x, np.array([power, 0.0]), {}) * dt
    energy_out = 0.0
    while x[0] > battery.p.soc_min + 1e-9:
        energy_out += power * dt
        x = x + battery.rhs(0.0, x, np.array([0.0, power]), {}) * dt

    assert energy_out / energy_in == pytest.approx(battery.round_trip_efficiency(), rel=0.01)


def test_battery_soc_limits_respected(battery):
    assert battery.max_charge_power_W(np.array([battery.p.soc_max, 0.0, 0.0]), 300.0) == pytest.approx(0.0, abs=1e-6)
    assert battery.max_discharge_power_W(np.array([battery.p.soc_min, 0.0, 0.0]), 300.0) == pytest.approx(0.0, abs=1e-6)


def test_battery_degradation_cost_is_order_of_cents(battery):
    assert 0.005 < battery.cost_per_kWh_throughput_EUR() < 0.05


# --------------------------------------------------------------------------
# Solids inventory and sorbent deactivation
# --------------------------------------------------------------------------
def test_grasa_conversion_starts_at_one_and_decays(solids):
    assert float(solids.max_conversion(0.0)) == pytest.approx(1.0, rel=1e-9)
    values = [float(solids.max_conversion(n)) for n in (0, 5, 10, 20, 50, 200)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_grasa_conversion_matches_published_shape(solids):
    """Conversion should fall to roughly 0.16 by cycle 20 (Grasa & Abanades 2006)."""
    assert float(solids.max_conversion(20.0)) == pytest.approx(0.16, abs=0.03)


def test_grasa_conversion_asymptotes_to_residual(solids):
    assert float(solids.max_conversion(1e5)) == pytest.approx(solids.p.grasa_residual_conversion, abs=1e-3)


def test_solids_total_is_conserved(solids):
    """CaO + CaCO3 must be constant with no make-up or purge."""
    x = solids.initial_state()
    w = {"r_carbonation_mol_s": 0.2, "r_calcination_mol_s": 0.35}
    total0 = x[0] + x[1]
    for _ in range(1000):
        x = x + solids.rhs(0.0, x, np.zeros(0), w) * 60.0
    assert x[0] + x[1] == pytest.approx(total0, rel=1e-12)


def test_cycle_counter_advances_with_calcination(solids):
    """One full pass of the inventory through the kiln is exactly one cycle."""
    x = solids.initial_state()
    rate = 1.0
    duration = solids.p.n_total_mol / rate  # seconds to calcine the whole inventory
    w = {"r_carbonation_mol_s": rate, "r_calcination_mol_s": rate}
    n0 = x[2]
    dt = duration / 2000.0
    for _ in range(2000):
        x = x + solids.rhs(0.0, x, np.zeros(0), w) * dt
    assert x[2] - n0 == pytest.approx(1.0, rel=1e-6)


def test_loading_is_a_fraction(solids):
    x = solids.initial_state()
    assert 0.0 <= float(solids.loading(x)) <= 1.0


# --------------------------------------------------------------------------
# Air contactor
# --------------------------------------------------------------------------
def test_contactor_zero_flow_captures_nothing(contactor):
    w = dict(AMBIENT, solids_loading=0.2)
    assert float(contactor.capture_rate_mol_s(np.array([0.0, 1.0]), w)) == pytest.approx(0.0, abs=1e-12)


def test_contactor_capture_fraction_falls_with_flow(contactor):
    """Less residence time means a smaller single-pass capture fraction."""
    fractions = [float(contactor.capture_fraction(v)) for v in (5.0, 15.0, 25.0, 40.0)]
    assert all(a > b for a, b in zip(fractions, fractions[1:]))


def test_contactor_capture_fraction_at_design_flow(contactor):
    got = float(contactor.capture_fraction(contactor.p.air_flow_design_m3_s))
    assert got == pytest.approx(contactor.p.capture_fraction_design, rel=1e-6)


def test_contactor_diminishing_returns_is_severe(contactor):
    """The headline nonlinearity: much more power buys only slightly more CO2.

    Verified figures: 60 % flow captures 0.237 mol/s for 4.7 kW; 100 % flow
    captures 0.284 mol/s for 20.4 kW. That is 20 % more CO2 for 4.3x the power.
    """
    w = dict(AMBIENT, solids_loading=0.2)
    r60 = float(contactor.capture_rate_mol_s(np.array([0.6, 1.0]), w))
    p60 = float(contactor.fan_power_W(np.array([0.6, 1.0])))
    r100 = float(contactor.capture_rate_mol_s(np.array([1.0, 1.0]), w))
    p100 = float(contactor.fan_power_W(np.array([1.0, 1.0])))
    assert r100 / r60 < 1.3, "capture should saturate with flow"
    assert p100 / p60 > 3.5, "fan power should rise far faster than capture"


def test_contactor_fan_power_is_cubic(contactor):
    """Doubling flow must raise shaft power by about eight times."""
    p_half = contactor.fan_power_W(np.array([0.5, 1.0])) - contactor.fan_idle_W
    p_full = contactor.fan_power_W(np.array([1.0, 1.0])) - contactor.fan_idle_W
    assert p_full / p_half == pytest.approx(8.0, rel=1e-6)


def test_contactor_stops_when_sorbent_saturated(contactor):
    w = dict(AMBIENT, solids_loading=1.0)
    saturated = float(contactor.capture_rate_mol_s(np.array([1.0, 1.0]), w))
    fresh = float(contactor.capture_rate_mol_s(np.array([1.0, 1.0]), dict(AMBIENT, solids_loading=0.0)))
    # smooth gates approach zero asymptotically by design, so assert a physical
    # bound rather than exact zero: less than 1 % of the fresh-bed rate
    assert saturated < 0.01 * fresh


def test_contactor_humidity_and_temperature_help(contactor):
    dry = dict(AMBIENT, relative_humidity=15.0, solids_loading=0.2)
    humid = dict(AMBIENT, relative_humidity=90.0, solids_loading=0.2)
    cold = dict(AMBIENT, temp_air=2.0, solids_loading=0.2)
    warm = dict(AMBIENT, temp_air=35.0, solids_loading=0.2)
    u = np.array([0.6, 1.0])
    assert contactor.capture_rate_mol_s(u, humid) > contactor.capture_rate_mol_s(u, dry)
    assert contactor.capture_rate_mol_s(u, warm) > contactor.capture_rate_mol_s(u, cold)


def test_contactor_flow_for_capture_rate_inverts(contactor):
    w = dict(AMBIENT, solids_loading=0.2)
    for target in (0.05, 0.10, 0.20):
        flow = contactor.flow_for_capture_rate(target, w)
        assert float(contactor.capture_rate_mol_s(np.array([flow, 1.0]), w)) == pytest.approx(target, rel=1e-3)


def test_contactor_dark_draws_nothing(contactor):
    assert float(contactor.fan_power_W(np.array([1.0, 0.0]))) < 1e-3


# --------------------------------------------------------------------------
# Calciner
# --------------------------------------------------------------------------
def test_baker_equilibrium_is_one_atm_at_897C(calciner):
    """The textbook calcination point: p_eq = 1 atm at about 1170 K."""
    assert float(calciner.equilibrium_pressure_atm(1170.0)) == pytest.approx(1.0, rel=0.1)


def test_calcination_threshold_temperature(calciner):
    """At 0.3 atm operating pressure the kiln becomes productive at ~1092 K."""
    threshold = calciner.threshold_temperature_K()
    assert threshold == pytest.approx(1092.0, abs=5.0)
    assert float(calciner.equilibrium_pressure_atm(threshold)) == pytest.approx(
        calciner.p.operating_pressure_atm, rel=1e-6
    )


def test_no_calcination_below_threshold(calciner):
    """Below the threshold the kiln consumes power and produces nothing."""
    w = {"solids_n_caco3_mol": 40000.0}
    below = np.array([calciner.threshold_temperature_K() - 50.0])
    at_temperature = float(calciner.calcination_rate_mol_s(np.array([1173.15]), np.array([1.0, 1.0]), w))
    assert float(calciner.calcination_rate_mol_s(below, np.array([1.0, 1.0]), w)) < 1e-3 * at_temperature


def test_calcination_rate_rises_with_temperature(calciner):
    w = {"solids_n_caco3_mol": 40000.0}
    rates = [
        float(calciner.calcination_rate_mol_s(np.array([float(t)]), np.array([1.0, 1.0]), w))
        for t in (1100, 1133, 1173, 1213)
    ]
    assert all(a < b for a, b in zip(rates, rates[1:]))


def test_calcination_never_exceeds_throughput_limit(calciner):
    w = {"solids_n_caco3_mol": 40000.0}
    for t in (1173, 1250, 1273):
        rate = float(calciner.calcination_rate_mol_s(np.array([float(t)]), np.array([1.0, 1.0]), w))
        assert rate <= calciner.p.calcination_rate_max_mol_s + 1e-9


def test_calciner_stops_without_feedstock(calciner):
    empty = float(calciner.calcination_rate_mol_s(np.array([1173.0]), np.array([1.0, 1.0]), {"solids_n_caco3_mol": 0.0}))
    stocked = float(calciner.calcination_rate_mol_s(np.array([1173.0]), np.array([1.0, 1.0]), {"solids_n_caco3_mol": 40000.0}))
    assert empty < 0.01 * stocked


def test_calciner_cold_start_cost_and_time(calciner):
    """Verified: 772 kWh and 5.15 h at rated power."""
    assert calciner.cold_start_energy_J() / 3.6e6 == pytest.approx(772.0, rel=0.02)
    assert calciner.cold_start_energy_J() / 3.6e6 / calciner.p.heater_power_rated_kw == pytest.approx(5.15, rel=0.02)


def test_calciner_overnight_hold_versus_reheat_is_close(calciner):
    """The decision the planner exists to make: holding and reheating are within
    a few kWh of each other over a 12 h night."""
    w = {"solids_n_caco3_mol": 40000.0, "temp_air": 20.0, "wind_speed": 2.0}
    x = np.array([calciner.p.temperature_target_K])
    dt = 60.0
    for _ in range(int(12 * 3600 / dt)):
        x = x + calciner.rhs(0.0, x, np.array([0.0, 0.0]), w) * dt
    reheat_kwh = calciner.p.thermal_capacity_J_K * (calciner.p.temperature_target_K - x[0]) / calciner.p.heater_efficiency / 3.6e6
    hold_kwh = calciner.standing_loss_W() * 12 * 3600 / 3.6e6
    assert abs(reheat_kwh - hold_kwh) / hold_kwh < 0.15


def test_calciner_overtemperature_interlock_cuts_heater(calciner):
    """Above the refractory limit the heater must be cut regardless of command."""
    hot = np.array([calciner.p.temperature_max_K + 20.0])
    assert float(calciner._heater_fraction(np.array([1.0, 1.0]), hot[0])) < 1e-3


def test_calciner_cannot_run_away_past_refractory_limit(calciner):
    """Integrate at full heater duty for a day; the interlock must hold the line."""
    w = {"solids_n_caco3_mol": 45000.0, "temp_air": 25.0, "wind_speed": 1.0}
    x = np.array([calciner.p.temperature_initial_K])
    for _ in range(int(24 * 3600 / 60.0)):
        x = x + calciner.rhs(0.0, x, np.array([1.0, 1.0]), w) * 60.0
    assert x[0] < calciner.p.temperature_max_K + 15.0


def test_calciner_dark_draws_nothing(calciner):
    out = calciner.outputs(0.0, np.array([1173.0]), np.array([1.0, 0.0]), {"solids_n_caco3_mol": 40000.0})
    assert out["power_electrical_W"] < 1e-3
    assert float(out["r_calcination_mol_s"]) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Electrolyser
# --------------------------------------------------------------------------
def test_faraday_law_hydrogen_rate(electrolyser):
    """Two electrons per H2, times the Faradaic efficiency."""
    x = np.array([333.15, 0.0])
    u = np.array([1.0, 1.0])
    current = electrolyser.rated_current_A
    expected = electrolyser.p.faraday_efficiency * electrolyser.p.n_cells * current / (2 * 96485.332)
    assert float(electrolyser.hydrogen_rate_mol_s(x, u)) == pytest.approx(expected, rel=1e-6)


def test_cell_voltage_is_physical(electrolyser):
    """At rated current and 60 degC a PEM cell should sit near 1.9-2.0 V."""
    v = float(electrolyser.cell_voltage(1.5, 333.15, 0.0))
    assert 1.80 < v < 2.05


def test_cell_voltage_rises_with_current(electrolyser):
    voltages = [float(electrolyser.cell_voltage(i, 333.15, 0.0)) for i in (0.1, 0.5, 1.0, 1.5)]
    assert all(a < b for a, b in zip(voltages, voltages[1:]))


def test_activation_overpotential_is_finite_at_zero_current(electrolyser):
    """The asinh form must not diverge at i = 0, unlike Tafel.

    This is why the model uses asinh: an NLP will evaluate the stack at zero
    load, and a logarithmic Tafel term would return -inf and poison the solve.
    """
    value = float(electrolyser.activation_overpotential(0.0, 333.15))
    assert value == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(value)


def test_warmer_stack_is_more_efficient(electrolyser):
    """Membrane resistance falls with temperature."""
    cold = float(electrolyser.cell_voltage(1.0, 300.0, 0.0))
    warm = float(electrolyser.cell_voltage(1.0, 350.0, 0.0))
    assert warm < cold


def test_efficiency_peaks_at_part_load(electrolyser):
    """The headline nonlinearity, verified: total LHV efficiency peaks near 53 %.

    Stack efficiency falls monotonically with current while the fixed auxiliary
    load is spread over more hydrogen, so the total has an interior maximum.
    """
    peak = electrolyser.best_efficiency_fraction()
    assert 0.35 < peak < 0.70
    x = np.array([333.15, 0.0])
    eta_peak = float(electrolyser.efficiency_lhv(x, np.array([peak, 1.0])))
    eta_full = float(electrolyser.efficiency_lhv(x, np.array([1.0, 1.0])))
    eta_min = float(electrolyser.efficiency_lhv(x, np.array([electrolyser.p.current_density_min_fraction, 1.0])))
    assert eta_peak > eta_full
    assert eta_peak > eta_min


def test_specific_energy_is_realistic(electrolyser):
    """A PEM stack at rated load should need 50-60 kWh per kg of hydrogen."""
    x = np.array([333.15, 0.0])
    u = np.array([1.0, 1.0])
    power_kw = float(electrolyser.total_power_W(x, u)) / 1e3
    rate_kg_h = float(electrolyser.hydrogen_rate_mol_s(x, u)) * M_H2 * 3600.0
    assert 48.0 < power_kw / rate_kg_h < 62.0


def test_minimum_load_is_enforced(electrolyser):
    """Below the crossover limit the stack must not draw current."""
    x = np.array([333.15, 0.0])
    below = 0.5 * electrolyser.p.current_density_min_fraction
    assert float(electrolyser.current_density(np.array([below, 1.0]))) < 0.01


def test_power_inversion_round_trips(electrolyser):
    x = np.array([333.15, 0.0])
    for target in (60e3, 150e3, 250e3, 320e3):
        frac = electrolyser.fraction_for_power(x, target)
        assert float(electrolyser.total_power_W(x, np.array([frac, 1.0]))) == pytest.approx(target, rel=1e-3)


def test_electrolyser_heats_up_under_load(electrolyser):
    x = np.array([293.15, 0.0])
    for _ in range(60):
        x = x + electrolyser.rhs(0.0, x, np.array([1.0, 1.0]), {"temp_air": 20.0}) * 60.0
    assert x[0] > 320.0


def test_electrolyser_thermal_management_bounds_temperature(electrolyser):
    x = np.array([293.15, 0.0])
    for _ in range(int(6 * 3600 / 60)):
        x = x + electrolyser.rhs(0.0, x, np.array([1.0, 1.0]), {"temp_air": 40.0}) * 60.0
    assert x[0] < electrolyser.p.temperature_max_K


# --------------------------------------------------------------------------
# Sabatier
# --------------------------------------------------------------------------
BUFFERS_FULL = {
    "gas_co2_available_mol": 3000.0,
    "gas_h2_available_mol": 12000.0,
    "water_available_kg": 3000.0,
    "temp_air": 20.0,
}


def test_equilibrium_conversion_falls_with_temperature(sabatier):
    """The reaction is exothermic, so hotter means a lower equilibrium ceiling."""
    values = [float(sabatier.equilibrium_conversion(t)) for t in (523, 573, 673, 773, 823)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_equilibrium_conversion_matches_fitted_points(sabatier):
    assert float(sabatier.equilibrium_conversion(573.15)) == pytest.approx(0.97, abs=0.03)
    assert float(sabatier.equilibrium_conversion(773.15)) == pytest.approx(0.63, abs=0.05)


def test_conversion_never_exceeds_equilibrium(sabatier):
    for t in (500, 573, 673, 773, 823):
        x = np.array([float(t), 1.0, 0.0])
        for feed in (0.1, 0.5, 1.0):
            u = np.array([feed, 1.0])
            assert float(sabatier.conversion(x, u)) <= float(sabatier.equilibrium_conversion(t)) + 1e-12


def test_conversion_has_an_interior_optimum(sabatier):
    """Kinetics rise and equilibrium falls with temperature; the product peaks
    near 300 degC. Verified optimum: 299 degC."""
    best = sabatier.best_temperature_K()
    assert 280.0 < best - 273.15 < 360.0


def test_cold_reactor_produces_nothing(sabatier):
    cold = float(sabatier.outputs(0.0, np.array([300.0, 1.0, 0.0]), np.array([1.0, 1.0]), BUFFERS_FULL)["ch4_rate_kg_s"])
    lit = float(sabatier.outputs(0.0, np.array([573.15, 1.0, 0.0]), np.array([1.0, 1.0]), BUFFERS_FULL)["ch4_rate_kg_s"])
    assert cold < 1e-3 * lit


def test_reactor_is_self_sustaining_at_full_feed(sabatier):
    """Verified: settles near 316 degC drawing only its 8 kW auxiliaries, and
    needing cooling rather than heating. This is the property the whole
    night-time strategy depends on."""
    x = np.array([573.15, 1.0, 0.0])
    u = np.array([1.0, 1.0])
    for _ in range(int(6 * 3600 / 60)):
        x = x + sabatier.rhs(0.0, x, u, BUFFERS_FULL) * 60.0
    out = sabatier.outputs(0.0, x, u, BUFFERS_FULL)
    assert x[0] > sabatier.p.temperature_ignition_K
    assert float(out["sabatier_preheat_W"]) < 1e-3, "should need no electric heat"
    assert float(out["sabatier_cooling_W"]) > 1e3, "should need active cooling"
    assert float(out["power_electrical_W"]) == pytest.approx(sabatier.p.auxiliary_power_kw * 1e3, rel=0.01)


def test_night_methane_is_nearly_free(sabatier):
    """Running on banked reactants costs about 0.7 kWh/kg against ~28 kWh/kg for
    the electrolysis that made the hydrogen."""
    x = np.array([573.15, 1.0, 0.0])
    u = np.array([0.8, 1.0])
    energy_J = 0.0
    for _ in range(int(12 * 3600 / 60)):
        energy_J += float(sabatier.outputs(0.0, x, u, BUFFERS_FULL)["power_electrical_W"]) * 60.0
        x = x + sabatier.rhs(0.0, x, u, BUFFERS_FULL) * 60.0
    specific = (energy_J / 3.6e6) / x[2]
    assert specific < 2.0


def test_stoichiometric_starvation_throttles_the_reactor(sabatier):
    """Whichever reactant is short throttles the reactor, however much of the
    other is banked -- the 4:1 coupling that forces co-scheduling."""
    x = np.array([573.15, 1.0, 0.0])
    u = np.array([1.0, 1.0])
    full = float(sabatier.co2_rate_mol_s(x, u, BUFFERS_FULL))
    co2_rich = dict(BUFFERS_FULL, gas_h2_available_mol=0.0)
    h2_rich = dict(BUFFERS_FULL, gas_co2_available_mol=0.0)
    assert float(sabatier.co2_rate_mol_s(x, u, co2_rich)) < 1e-3 * full
    assert float(sabatier.co2_rate_mol_s(x, u, h2_rich)) < 1e-3 * full


def test_catalyst_deactivates_faster_when_hot(sabatier):
    cool = np.array([573.15, 1.0, 0.0])
    hot = np.array([773.15, 1.0, 0.0])
    u = np.array([1.0, 1.0])
    d_cool = float(sabatier.rhs(0.0, cool, u, BUFFERS_FULL)[1])
    d_hot = float(sabatier.rhs(0.0, hot, u, BUFFERS_FULL)[1])
    assert d_hot < d_cool < 0.0
    assert abs(d_hot) > 10 * abs(d_cool)


def test_reactor_shut_down_draws_nothing(sabatier):
    out = sabatier.outputs(0.0, np.array([573.15, 1.0, 0.0]), np.array([1.0, 0.0]), BUFFERS_FULL)
    assert float(out["power_electrical_W"]) < 1e-3


# --------------------------------------------------------------------------
# Buffers
# --------------------------------------------------------------------------
def test_gas_buffer_stoichiometry(gas):
    """The reactor draws 4 mol H2 per mol CO2."""
    x = gas.initial_state()
    w = {"r_electrolysis_h2_mol_s": 0.0, "r_calcination_mol_s": 0.0, "r_sabatier_co2_mol_s": 0.1}
    d = gas.rhs(0.0, x, np.zeros(0), w)
    assert float(d[0]) == pytest.approx(-0.4)
    assert float(d[1]) == pytest.approx(-0.1)


def test_gas_buffer_stops_accepting_when_full(gas):
    x = np.array([gas.p.h2_capacity_mol, gas.p.co2_capacity_mol])
    w = {"r_electrolysis_h2_mol_s": 1.0, "r_calcination_mol_s": 1.0, "r_sabatier_co2_mol_s": 0.0}
    d = gas.rhs(0.0, x, np.zeros(0), w)
    assert float(d[0]) < 1e-3, "a full tank must not keep accepting hydrogen"
    assert float(d[1]) < 1e-3, "a full tank must not keep accepting CO2"


def test_gas_buffer_holds_a_day_of_electrolyser_output(gas):
    """Sized so the H2 tank does not fill and stall the electrolyser mid-afternoon."""
    day_mol = 0.8162 * 9.0 * 3600.0
    assert gas.p.h2_capacity_mol > day_mol


def test_co2_compressor_power_follows_calcination(gas):
    out = gas.outputs(0.0, gas.initial_state(), np.zeros(0), {"r_calcination_mol_s": 0.2})
    assert float(out["power_electrical_W"]) == pytest.approx(0.2 * gas.p.co2_compressor_kJ_per_mol * 1e3)


def test_water_balance_matches_stoichiometry(water):
    """Net make-up should be about 2.36 kg per kg of methane after recovery."""
    assert water.specific_consumption_kg_per_kg_ch4() == pytest.approx(2.36, abs=0.05)


def test_water_consumed_when_electrolysing(water):
    x = water.initial_state()
    w = {"r_electrolysis_water_mol_s": 0.8, "r_sabatier_co2_mol_s": 0.0}
    d = water.rhs(0.0, x, np.zeros(0), w)
    assert float(d[0]) < 0.0
    assert float(d[1]) > 0.0


# --------------------------------------------------------------------------
# The symbolic/numeric contract
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,n_states,n_inputs,context",
    [
        ("solids", 3, 0, {"r_carbonation_mol_s": 0.1, "r_calcination_mol_s": 0.2}),
        ("calciner", 1, 2, {"solids_n_caco3_mol": 40000.0, "temp_air": 20.0, "wind_speed": 2.0}),
        ("electrolyser", 2, 2, {"temp_air": 20.0}),
        ("gas", 2, 0, {"r_electrolysis_h2_mol_s": 0.8, "r_calcination_mol_s": 0.3, "r_sabatier_co2_mol_s": 0.2}),
        ("water", 2, 0, {"r_electrolysis_water_mol_s": 0.8, "r_sabatier_co2_mol_s": 0.2}),
        ("sabatier", 3, 2, BUFFERS_FULL),
    ],
)
def test_rhs_matches_between_numpy_and_casadi(name, n_states, n_inputs, context, request):
    """The single most important invariant in the project.

    Every model is written once and evaluated numerically by the simulator and
    symbolically by the estimator and controller. If that contract breaks, the
    controller starts steering a different plant from the one it is watching, and
    nothing else in the suite would notice.
    """
    model = request.getfixturevalue(name)
    rng = np.random.default_rng(0)
    lower, upper = model.state_bounds()
    x_val = np.clip(rng.uniform(0.3, 0.7, n_states) * (upper - lower) + lower, lower, upper)
    u_val = np.array([0.6, 1.0])[:n_inputs] if n_inputs else np.zeros(0)

    x = ca.SX.sym("x", n_states)
    u = ca.SX.sym("u", max(n_inputs, 0))
    f = ca.Function("f", [x, u], [model.rhs(0.0, x, u, context)])

    numeric = np.asarray(model.rhs(0.0, x_val, u_val, context), dtype=float).reshape(-1)
    symbolic = np.array(f(x_val, u_val)).reshape(-1)
    assert np.allclose(numeric, symbolic, rtol=1e-9, atol=1e-14)


def test_contactor_outputs_are_symbolically_evaluable(contactor):
    w = dict(AMBIENT)
    w["solids_loading"] = ca.SX.sym("load")
    out = contactor.outputs(0.0, ca.SX.zeros(0), ca.SX.sym("u", 2), w)
    assert isinstance(out["r_carbonation_mol_s"], ca.SX)


def test_uniform_dispatch_protocol_is_monotonic(contactor, calciner, electrolyser):
    """Draw must be non-decreasing in setpoint, or bisection inversion breaks."""
    cases = [
        (contactor, np.zeros(0)),
        (calciner, np.array([1173.15])),
        (electrolyser, np.array([333.15, 0.0])),
    ]
    for model, x in cases:
        powers = [model.power_for_setpoint(x, s, 1.0, AMBIENT) for s in np.linspace(0.0, 1.0, 40)]
        assert all(a <= b + 1e-6 for a, b in zip(powers, powers[1:])), type(model).__name__
