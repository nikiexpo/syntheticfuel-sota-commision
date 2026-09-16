"""Run metrics -- every number the challenge brief asks a submission to report.

    synthetic-methane production and plant utilisation
    battery state over time
    energy lost through curtailment
    the limiting subsystem
    responses to at least one injected fault          (M2 onwards)

plus a comparison against a simple baseline. Computed once here so the report,
the baseline table and the siting sweep cannot disagree.

The limiting subsystem
----------------------
With real buffers this becomes the most informative number in the report, and it
is emphatically not a static property of the design -- what limits the plant
changes hour by hour, and the answer differs between control strategies on
identical hardware. So it is classified per timestep and reported as a time
distribution plus a daylight headline.

The diagnosis asks "what stopped the reactor making more methane right now?" and
the candidate answers are genuinely different engineering problems:

    reactor capacity    the plant is running flat out -- buy a bigger reactor
    CO2 supply          the reactor is starved of carbon -- more calciner, or more contactor
    H2 supply           starved of hydrogen -- more electrolyser or more solar
    sorbent saturated   the contactor has nowhere to put CO2 -- calcine more
    CO2 buffer full     the calciner has nowhere to put CO2 -- the reactor is the bottleneck
    H2 buffer full      the electrolyser is ahead of the reactor
    solar / battery     no energy -- more array or more storage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sfp.economics import CapexBreakdown, Economics
from sfp.sim.simulator import SimulationResult
from sfp.units import J_PER_KWH, LHV_CH4_MASS, M_CO2

LIMIT_SOLAR = "solar"
LIMIT_BATTERY = "battery energy"
LIMIT_REACTOR = "reactor capacity"
LIMIT_CO2_SUPPLY = "CO2 supply"
LIMIT_H2_SUPPLY = "H2 supply"
LIMIT_SORBENT_FULL = "sorbent saturated"
LIMIT_CO2_FULL = "CO2 buffer full"
LIMIT_H2_FULL = "H2 buffer full"
LIMIT_NONE = "unconstrained"


@dataclass
class RunMetrics:
    """Everything worth reporting about one closed-loop run."""

    controller: str
    days: float

    # production
    ch4_kg: float
    ch4_kg_per_day: float
    ch4_energy_kwh: float
    utilisation: float
    co2_captured_kg: float
    h2_produced_kg: float
    h2_vented_kg: float
    h2_vented_fraction: float
    water_consumed_kg: float

    # energy
    pv_available_kwh: float
    pv_used_kwh: float
    pv_curtailed_kwh: float
    pv_clipped_kwh: float
    curtailment_fraction: float
    load_energy_kwh: float
    battery_charge_kwh: float
    battery_discharge_kwh: float
    specific_energy_kwh_per_kg: float
    system_efficiency_lhv: float

    # per-subsystem energy share
    energy_by_subsystem_kwh: dict[str, float] = field(default_factory=dict)

    # storage and degradation
    battery_efc: float = 0.0
    battery_fade: float = 0.0
    soc_min: float = 0.0
    soc_max: float = 0.0
    soc_final: float = 0.0
    sorbent_cycles: float = 0.0
    sorbent_conversion_start: float = 0.0
    sorbent_conversion_end: float = 0.0
    catalyst_activity_end: float = 1.0
    electrolyser_degradation_V: float = 0.0

    # buffer utilisation
    solids_loading_min: float = 0.0
    solids_loading_max: float = 0.0
    h2_fill_min: float = 0.0
    h2_fill_max: float = 0.0
    co2_fill_min: float = 0.0
    co2_fill_max: float = 0.0
    night_production_fraction: float = 0.0

    # constraint analysis
    limiting_subsystem: str = LIMIT_NONE
    limiting_distribution: dict[str, float] = field(default_factory=dict)

    # control-layer health
    bus_interventions: int = 0
    bus_trips: int = 0
    shed_kwh: float = 0.0
    unserved_kwh: float = 0.0

    # economics
    lcom_eur_per_kg: float = float("inf")
    lcom_eur_per_mwh: float = float("inf")
    operating_margin_eur: float = 0.0
    capex_eur: float = 0.0

    # provenance
    weather_provenance: str = "unknown"
    wall_time_s: float = 0.0

    def as_row(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not isinstance(v, dict)}


def _limiting_factor(log: pd.DataFrame, plant) -> pd.Series:
    """Classify, per timestep, what stopped the plant making more methane."""
    n = len(log)
    labels = np.full(n, LIMIT_NONE, dtype=object)

    def col(name: str, default: float = 0.0) -> np.ndarray:
        return log[name].to_numpy() if name in log else np.full(n, default)

    feed = col("sabatier_feed_fraction")
    availability = col("sabatier_feed_availability", 1.0)
    co2_avail = col("gas_co2_available_mol")
    h2_avail = col("gas_h2_available_mol")
    loading = col("solids_loading")
    co2_fill = col("gas_co2_fill")
    h2_fill = col("gas_h2_fill")
    cos_z = col("cos_zenith", 1.0)
    soc = col("battery_soc", 1.0)
    shed = col("shed_W")
    tripped = col("bus_tripped")

    battery = plant["battery"]
    sabatier = plant["sabatier"]

    # start from the energy-side diagnosis
    labels[:] = LIMIT_SOLAR
    labels[(soc <= battery.p.soc_min + 1e-3) & (cos_z <= 0.0)] = LIMIT_BATTERY
    labels[(shed > 1.0) & (cos_z > 0.0)] = LIMIT_SOLAR
    labels[tripped > 0] = LIMIT_BATTERY

    # material-side diagnoses override, because they are more actionable
    starved = availability < 0.98
    co2_short = starved & (co2_avail / 50.0 < h2_avail / 200.0)
    labels[co2_short] = LIMIT_CO2_SUPPLY
    labels[starved & ~co2_short] = LIMIT_H2_SUPPLY

    labels[loading > 0.95] = LIMIT_SORBENT_FULL
    labels[co2_fill > 0.97] = LIMIT_CO2_FULL
    labels[h2_fill > 0.97] = LIMIT_H2_FULL

    # running flat out beats every other explanation
    at_capacity = feed >= 0.995
    labels[at_capacity] = LIMIT_REACTOR

    return pd.Series(labels, index=log.index, name="limiting")


def compute_metrics(
    result: SimulationResult,
    economics: Economics | None = None,
    annual_scale: float | None = None,
) -> RunMetrics:
    """Reduce a simulation log to the reportable metrics."""
    log = result.log
    dt = result.dt_s
    plant = result.plant
    economics = economics or Economics()

    duration_s = len(log) * dt
    days = duration_s / 86400.0

    def energy_kwh(column: str) -> float:
        return float(log[column].sum() * dt / J_PER_KWH) if column in log else 0.0

    def final(column: str, default: float = 0.0) -> float:
        return float(log[column].iloc[-1]) if column in log else default

    def first(column: str, default: float = 0.0) -> float:
        return float(log[column].iloc[0]) if column in log else default

    ch4_kg = final("ch4_total_kg")

    # per-subsystem energy, from the canonical `power.<key>` channel
    energy_by_subsystem = {
        key: energy_kwh(f"power.{key}")
        for key in plant.subsystems
        if f"power.{key}" in log and key not in ("pv", "battery")
    }
    load_energy = sum(energy_by_subsystem.values())

    pv_available = energy_kwh("pv_available_W")
    pv_used = energy_kwh("pv_used_W")
    pv_curtailed = energy_kwh("pv_curtailed_W")

    # utilisation: energy actually consumed against what the plant could have
    # drawn had every subsystem run at its rating for the whole window
    rated_W = 0.0
    for key, sub in plant:
        if key in ("pv", "battery") or sub.n_inputs < 2:
            continue
        x0 = sub.initial_state()
        try:
            rated_W += float(sub.power_for_setpoint(x0, 1.0, 1.0, {}))
        except Exception:  # pragma: no cover - defensive
            pass
    rated_kwh = rated_W * duration_s / J_PER_KWH
    utilisation = load_energy / rated_kwh if rated_kwh > 0 else 0.0

    ch4_energy_kwh = ch4_kg * LHV_CH4_MASS / J_PER_KWH
    specific = load_energy / ch4_kg if ch4_kg > 1e-9 else float("inf")
    efficiency = ch4_energy_kwh / load_energy if load_energy > 1e-9 else 0.0

    co2_captured = float(log["r_carbonation_mol_s"].sum() * dt * M_CO2) if "r_carbonation_mol_s" in log else 0.0
    h2_produced = float(log["electrolyser_h2_rate_kg_s"].sum() * dt) if "electrolyser_h2_rate_kg_s" in log else 0.0
    from sfp.units import M_H2 as _M_H2
    h2_vented = float(log["gas_h2_vented_mol_s"].sum() * dt * _M_H2) if "gas_h2_vented_mol_s" in log else 0.0

    # night production: the headline test of whether the buffers are working
    if "cos_zenith" in log and "ch4_rate_kg_s" in log:
        night = log["cos_zenith"] <= 0.0
        night_kg = float(log.loc[night, "ch4_rate_kg_s"].sum() * dt)
        night_fraction = night_kg / ch4_kg if ch4_kg > 1e-9 else 0.0
    else:
        night_fraction = 0.0

    limiting = _limiting_factor(log, plant)
    distribution = limiting.value_counts(normalize=True).to_dict()
    daylight = log["cos_zenith"] > 0.0 if "cos_zenith" in log else pd.Series(True, index=log.index)
    daylight_limits = limiting[daylight]
    headline = daylight_limits.value_counts().idxmax() if len(daylight_limits) else LIMIT_NONE

    capex = economics.capex_full(plant)
    scale = annual_scale if annual_scale is not None else 31_536_000.0 / max(duration_s, 1.0)
    annual_ch4 = ch4_kg * scale
    water_consumed = final("water_consumed_kg")
    battery_efc = final("battery_efc")
    # Battery replacement, in two disjoint parts.
    #
    # The cycle-attributable part is what the controller already pays through
    # `cost_per_kWh_delivered` on every kWh moved; over a pack's rated cycle life
    # those charges total exactly one pack.
    #
    # The calendar-attributable part is paid by nobody. The capital annuity
    # amortises the battery over the project's 25 years and the pack does not
    # last them -- 13.3 years on calendar fade alone, and about five at two
    # equivalent full cycles a day. Counting only the first part understated the
    # cost of storage; counting the whole replacement in both places would
    # double-charge the cycling.
    battery = plant["battery"]
    annual_battery_cost = battery.cost_per_efc_EUR() * battery_efc * scale
    efc_per_year = battery_efc * scale
    throughput = (log.get("battery_charge_W", pd.Series([0.0]))
                  + log.get("battery_discharge_W", pd.Series([0.0]))).to_numpy()
    stress = battery.mean_stress_over(
        log["battery_soc"].to_numpy() if "battery_soc" in log else [0.5],
        throughput)
    annual_battery_cost += economics.capital_recovery_factor() * (
        battery.uncharged_replacement_PV_EUR(
            project_years=float(economics.p.project_lifetime_years),
            discount_rate=float(economics.p.discount_rate),
            efc_per_year=efc_per_year, mean_stress=stress,
        )
    )

    lcom = economics.lcom(capex, annual_ch4, water_consumed * scale, annual_battery_cost)
    margin = economics.marginal_objective_EUR(
        ch4_kg=ch4_kg,
        battery_efc=battery_efc,
        battery_cost_per_efc=plant["battery"].cost_per_efc_EUR(),
        starts=0.0,
        water_kg=water_consumed,
    )

    soc = log["battery_soc"] if "battery_soc" in log else pd.Series([0.0])

    return RunMetrics(
        controller=result.controller_name,
        days=days,
        ch4_kg=ch4_kg,
        ch4_kg_per_day=ch4_kg / days if days > 0 else 0.0,
        ch4_energy_kwh=ch4_energy_kwh,
        utilisation=utilisation,
        co2_captured_kg=co2_captured,
        h2_produced_kg=h2_produced,
        h2_vented_kg=h2_vented,
        h2_vented_fraction=h2_vented / h2_produced if h2_produced > 1e-9 else 0.0,
        water_consumed_kg=water_consumed,
        pv_available_kwh=pv_available,
        pv_used_kwh=pv_used,
        pv_curtailed_kwh=pv_curtailed,
        pv_clipped_kwh=energy_kwh("pv_clipped_W"),
        curtailment_fraction=pv_curtailed / pv_available if pv_available > 1e-9 else 0.0,
        load_energy_kwh=load_energy,
        battery_charge_kwh=energy_kwh("battery_charge_W"),
        battery_discharge_kwh=energy_kwh("battery_discharge_W"),
        specific_energy_kwh_per_kg=specific,
        system_efficiency_lhv=efficiency,
        energy_by_subsystem_kwh=energy_by_subsystem,
        battery_efc=battery_efc,
        battery_fade=final("battery_fade"),
        soc_min=float(soc.min()),
        soc_max=float(soc.max()),
        soc_final=float(soc.iloc[-1]),
        sorbent_cycles=final("solids_cycle_number") - first("solids_cycle_number"),
        sorbent_conversion_start=first("solids_max_conversion", 1.0),
        sorbent_conversion_end=final("solids_max_conversion", 1.0),
        catalyst_activity_end=final("sabatier_catalyst_activity", 1.0),
        electrolyser_degradation_V=final("electrolyser_v_degradation"),
        solids_loading_min=float(log["solids_loading"].min()) if "solids_loading" in log else 0.0,
        solids_loading_max=float(log["solids_loading"].max()) if "solids_loading" in log else 0.0,
        h2_fill_min=float(log["gas_h2_fill"].min()) if "gas_h2_fill" in log else 0.0,
        h2_fill_max=float(log["gas_h2_fill"].max()) if "gas_h2_fill" in log else 0.0,
        co2_fill_min=float(log["gas_co2_fill"].min()) if "gas_co2_fill" in log else 0.0,
        co2_fill_max=float(log["gas_co2_fill"].max()) if "gas_co2_fill" in log else 0.0,
        night_production_fraction=night_fraction,
        limiting_subsystem=headline,
        limiting_distribution=distribution,
        bus_interventions=int(log["bus_intervened"].sum()) if "bus_intervened" in log else 0,
        bus_trips=int(log["bus_tripped"].sum()) if "bus_tripped" in log else 0,
        shed_kwh=energy_kwh("shed_W"),
        unserved_kwh=energy_kwh("unserved_W"),
        lcom_eur_per_kg=lcom,
        lcom_eur_per_mwh=economics.lcom_per_mwh(lcom),
        operating_margin_eur=margin,
        capex_eur=capex.total,
        weather_provenance=result.weather_provenance,
        wall_time_s=result.wall_time_s,
    )


def comparison_table(metrics: list[RunMetrics]) -> pd.DataFrame:
    """Side-by-side comparison of several strategies, best LCOM first."""
    frame = pd.DataFrame([m.as_row() for m in metrics]).set_index("controller")
    columns = [
        "ch4_kg",
        "ch4_kg_per_day",
        "night_production_fraction",
        "utilisation",
        "co2_captured_kg",
        "h2_produced_kg",
        "h2_vented_fraction",
        "pv_curtailed_kwh",
        "curtailment_fraction",
        "specific_energy_kwh_per_kg",
        "system_efficiency_lhv",
        "sorbent_cycles",
        "sorbent_conversion_end",
        "battery_efc",
        "soc_min",
        "bus_trips",
        "limiting_subsystem",
        "lcom_eur_per_kg",
    ]
    available = [c for c in columns if c in frame.columns]
    return frame[available].sort_values("lcom_eur_per_kg")


def improvement(candidate: RunMetrics, baseline: RunMetrics) -> dict[str, float]:
    """Relative improvement of `candidate` over `baseline` on the headline metrics."""

    def rel(a: float, b: float) -> float:
        return float("nan") if b == 0 or not np.isfinite(b) else (a - b) / abs(b)

    return {
        "ch4_kg": rel(candidate.ch4_kg, baseline.ch4_kg),
        "lcom_eur_per_kg": rel(candidate.lcom_eur_per_kg, baseline.lcom_eur_per_kg),
        "curtailment_fraction": rel(candidate.curtailment_fraction, baseline.curtailment_fraction),
        "sorbent_cycles": rel(candidate.sorbent_cycles, baseline.sorbent_cycles),
        "battery_efc": rel(candidate.battery_efc, baseline.battery_efc),
        "specific_energy_kwh_per_kg": rel(
            candidate.specific_energy_kwh_per_kg, baseline.specific_energy_kwh_per_kg
        ),
    }
