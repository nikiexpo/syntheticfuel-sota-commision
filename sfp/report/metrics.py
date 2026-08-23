"""Run metrics -- every number the challenge brief asks a submission to report.

The brief's reference challenge lists five required outputs:

    synthetic-methane production and plant utilisation
    battery state over time
    energy lost through curtailment
    the limiting subsystem
    responses to at least one injected fault          (M2 onwards)

plus a comparison against a simple baseline. This module computes all of them
from a `SimulationResult`, so the report, the baseline table and the siting sweep
all read from one implementation and cannot disagree.

The one that needs explaining is **the limiting subsystem**. It is not a static
property of the design -- what limits the plant changes hour by hour. So it is
computed per timestep by asking which constraint was actually binding, and
reported as a time distribution plus a single headline (the most frequently
binding constraint while the sun was up). That is a more useful answer than a
single label, and it is what tells a designer which component to buy more of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sfp.economics import CapexBreakdown, Economics
from sfp.sim.simulator import SimulationResult
from sfp.units import J_PER_KWH, LHV_CH4_MASS

# constraint labels used by the limiting-subsystem analysis
LIMIT_SOLAR = "solar"
LIMIT_PROCESS_CAPACITY = "process capacity"
LIMIT_BATTERY_POWER = "battery power"
LIMIT_BATTERY_ENERGY = "battery energy"
LIMIT_MIN_LOAD = "minimum load"
LIMIT_INVERTER = "inverter clipping"
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
    starts: float

    # energy
    pv_available_kwh: float
    pv_used_kwh: float
    pv_curtailed_kwh: float
    pv_clipped_kwh: float
    curtailment_fraction: float
    process_energy_kwh: float
    standby_energy_kwh: float
    warmup_energy_kwh: float
    battery_charge_kwh: float
    battery_discharge_kwh: float
    battery_loss_kwh: float
    specific_energy_kwh_per_kg: float
    system_efficiency_lhv: float

    # storage
    battery_efc: float
    battery_fade: float
    soc_min: float
    soc_max: float
    soc_final: float

    # constraint analysis
    limiting_subsystem: str
    limiting_distribution: dict[str, float] = field(default_factory=dict)

    # health of the control layer
    bus_interventions: int = 0
    load_shed_kwh: float = 0.0
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
        """Flat mapping for a comparison table."""
        out = {k: v for k, v in self.__dict__.items() if not isinstance(v, dict)}
        return out


def _limiting_factor(log: pd.DataFrame, plant) -> pd.Series:
    """Classify, per timestep, which constraint was binding.

    Order matters: we report the *most binding* reason production was not
    higher. A plant that is off because there is no sun is solar-limited even
    though its battery is also empty.
    """
    process = plant["process"]
    battery = plant["battery"]

    rated = process.rated_power_W
    tol = 1e-3

    load = log["process_load_fraction"].to_numpy()
    pv_avail = log["pv_available_W"].to_numpy()
    curtailed = log["pv_curtailed_W"].to_numpy()
    discharge = log["battery_discharge_W"].to_numpy()
    soc = log["battery_soc"].to_numpy()
    clipped = log["pv_clipped_W"].to_numpy()
    cos_z = log["cos_zenith"].to_numpy() if "cos_zenith" in log else np.ones(len(log))

    labels = np.full(len(log), LIMIT_NONE, dtype=object)

    # running at rated: nothing is limiting production
    at_rated = load >= 1.0 - 1e-6
    # running below rated with solar spare and the battery not discharging hard:
    # the process itself is the limit only when it is at its ceiling
    surplus = curtailed > tol

    # battery at its power limit while trying to sustain the load
    battery_power_bound = discharge >= battery.max_power_W - 1.0
    battery_energy_bound = soc <= battery.p.soc_min + 1e-4

    off = load <= 1e-9
    below_min = (~off) & (load <= process.p.min_load_fraction + 1e-6)

    labels[:] = LIMIT_SOLAR
    labels[at_rated & surplus] = LIMIT_PROCESS_CAPACITY
    labels[at_rated & ~surplus] = LIMIT_PROCESS_CAPACITY
    labels[(~at_rated) & battery_power_bound] = LIMIT_BATTERY_POWER
    labels[(~at_rated) & battery_energy_bound] = LIMIT_BATTERY_ENERGY
    labels[(~at_rated) & below_min] = LIMIT_MIN_LOAD
    labels[at_rated & (clipped > tol)] = LIMIT_INVERTER

    # at night nothing is limiting except the sun
    labels[(cos_z <= 0.0) & off] = LIMIT_SOLAR

    return pd.Series(labels, index=log.index, name="limiting")


def compute_metrics(
    result: SimulationResult,
    economics: Economics | None = None,
    annual_scale: float | None = None,
) -> RunMetrics:
    """Reduce a simulation log to the reportable metrics.

    `annual_scale` multiplies run totals to reach an annual figure for the LCOM.
    Leave it None for a full-year run; for a short window pass the factor
    explicitly so the extrapolation is visible at the call site rather than
    hidden here. A 10-day midsummer window scaled by 36.5 overstates a European
    plant badly, and the report says so.
    """
    log = result.log
    dt = result.dt_s
    plant = result.plant
    economics = economics or Economics()

    duration_s = len(log) * dt
    days = duration_s / 86400.0

    def energy_kwh(column: str) -> float:
        return float(log[column].sum() * dt / J_PER_KWH) if column in log else 0.0

    ch4_kg = float(log["ch4_total_kg"].iloc[-1]) if "ch4_total_kg" in log else 0.0
    starts = float(log["process_starts"].iloc[-1]) if "process_starts" in log else 0.0

    pv_available = energy_kwh("pv_available_W")
    pv_used = energy_kwh("pv_used_W")
    pv_curtailed = energy_kwh("pv_curtailed_W")
    pv_clipped = energy_kwh("pv_clipped_W")
    process_energy = energy_kwh("process_power_W")
    standby_energy = energy_kwh("process_standby_W")
    warmup_energy = energy_kwh("process_warmup_W")
    charge = energy_kwh("battery_charge_W")
    discharge = energy_kwh("battery_discharge_W")
    battery_loss = energy_kwh("battery_loss_W")

    # utilisation: energy actually put through the process as a fraction of
    # what it could have taken had it run flat out for the whole window
    rated_kwh = plant["process"].rated_power_W * duration_s / J_PER_KWH
    utilisation = process_energy / rated_kwh if rated_kwh > 0 else 0.0

    ch4_energy_kwh = ch4_kg * LHV_CH4_MASS / J_PER_KWH
    total_electrical = process_energy + standby_energy + warmup_energy
    specific = total_electrical / ch4_kg if ch4_kg > 1e-9 else float("inf")
    efficiency = ch4_energy_kwh / total_electrical if total_electrical > 1e-9 else 0.0

    limiting = _limiting_factor(log, plant)
    distribution = (limiting.value_counts(normalize=True)).to_dict()
    # headline: the binding constraint during daylight, which is what a
    # designer would act on
    daylight = log["cos_zenith"] > 0.0 if "cos_zenith" in log else pd.Series(True, index=log.index)
    daylight_limits = limiting[daylight]
    headline = (
        daylight_limits.value_counts().idxmax() if len(daylight_limits) else LIMIT_NONE
    )

    soc = log["battery_soc"]
    capex = economics.capex(plant["pv"], plant["battery"], plant["process"])

    scale = annual_scale if annual_scale is not None else 31_536_000.0 / max(duration_s, 1.0)
    annual_ch4 = ch4_kg * scale
    annual_water = economics.stoichiometric_water_kg(annual_ch4)
    battery_efc = float(log["battery_efc"].iloc[-1]) if "battery_efc" in log else 0.0
    annual_battery_cost = plant["battery"].cost_per_efc_EUR() * battery_efc * scale

    lcom = economics.lcom(capex, annual_ch4, annual_water, annual_battery_cost)

    margin = economics.marginal_objective_EUR(
        ch4_kg=ch4_kg,
        battery_efc=battery_efc,
        battery_cost_per_efc=plant["battery"].cost_per_efc_EUR(),
        starts=starts,
        water_kg=economics.stoichiometric_water_kg(ch4_kg),
    )

    return RunMetrics(
        controller=result.controller_name,
        days=days,
        ch4_kg=ch4_kg,
        ch4_kg_per_day=ch4_kg / days if days > 0 else 0.0,
        ch4_energy_kwh=ch4_energy_kwh,
        utilisation=utilisation,
        starts=starts,
        pv_available_kwh=pv_available,
        pv_used_kwh=pv_used,
        pv_curtailed_kwh=pv_curtailed,
        pv_clipped_kwh=pv_clipped,
        curtailment_fraction=pv_curtailed / pv_available if pv_available > 1e-9 else 0.0,
        process_energy_kwh=process_energy,
        standby_energy_kwh=standby_energy,
        warmup_energy_kwh=warmup_energy,
        battery_charge_kwh=charge,
        battery_discharge_kwh=discharge,
        battery_loss_kwh=battery_loss,
        specific_energy_kwh_per_kg=specific,
        system_efficiency_lhv=efficiency,
        battery_efc=battery_efc,
        battery_fade=float(log["battery_fade"].iloc[-1]) if "battery_fade" in log else 0.0,
        soc_min=float(soc.min()),
        soc_max=float(soc.max()),
        soc_final=float(soc.iloc[-1]),
        limiting_subsystem=headline,
        limiting_distribution=distribution,
        bus_interventions=int(log["bus_intervened"].sum()) if "bus_intervened" in log else 0,
        load_shed_kwh=energy_kwh("load_shed_W"),
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
        "utilisation",
        "starts",
        "pv_curtailed_kwh",
        "curtailment_fraction",
        "specific_energy_kwh_per_kg",
        "system_efficiency_lhv",
        "battery_efc",
        "soc_min",
        "limiting_subsystem",
        "lcom_eur_per_kg",
        "operating_margin_eur",
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
        "curtailment_fraction": rel(
            candidate.curtailment_fraction, baseline.curtailment_fraction
        ),
        "starts": rel(candidate.starts, baseline.starts),
        "battery_efc": rel(candidate.battery_efc, baseline.battery_efc),
        "operating_margin_eur": rel(candidate.operating_margin_eur, baseline.operating_margin_eur),
    }
