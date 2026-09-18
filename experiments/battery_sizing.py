"""Experiment A: closed-loop battery sizing, and what the safety layer does.

Why the sweeps are closed-loop
------------------------------
An earlier generation of grids scored 240 h dispatch plans instead of
simulations, on the strength of a measured plan-versus-realised bias of -0.8 %
at three days. Two things broke that licence:

* Re-measured after the battery economics were corrected, the bias is **-34 %
  at three days and -11.7 % at seven** -- not -0.8 %.
* Worse, it is **not common-mode across the swept axis**. Measured at 1100 kWp,
  EUR 6.00/kg, three days:

      pack     planned   realised     bias
       100      1838.8      849.4    -53.8 %
       250      1739.6     1345.3    -22.7 %
       500      1983.1     1399.7    -29.4 %
      1000      1973.1     1707.5    -13.5 %
      1500      2218.7     2164.6     -2.4 %

  A bias that varies by 51 points along the axis does not cancel in an ordering.
  It *tilts* the surface, and it inverted the conclusion: the plan-based grids
  put the optimum at the smallest pack; the correction runs the other way.

The mechanism is the point of the experiment. The outer layer is an hourly
optimiser; buffering demand lives below an hour; so it credits a small pack with
riding through transients that in reality trip the bus. Only a closed loop with
a sub-hourly inner layer can see that.

What is fixed and what is swept
-------------------------------
**The controller's methane price is fixed at its tuned value** rather than
swept. `methane_price_per_kg` is a control tuning parameter, not a market price:
sweeping it changes two things per cell -- what the plant is worth and how the
controller behaves. Fixed, production is fixed too, and the economics at any
market price are arithmetic afterwards.

Attribution
-----------
Each inner solve records why it moved the plan, two ways (see
`InnerNMPC._attribution`): the dual mass on each inequality block, and a
counterfactual rollout of the plan's own action. They answer different
questions and the difference matters -- measured at 03:00, a 1500 kWh plan
overcommits the bus by 43.6 kW against 4.5 kW for a 100 kWh plan, yet its dual
is 26x *smaller*, because the large pack can absorb what the small one cannot.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sfp.cli import REFERENCE_SIZING, build_plant, controllers_for  # noqa: E402
from sfp.economics import Economics  # noqa: E402
from sfp.report.metrics import compute_metrics  # noqa: E402
from sfp.report.plan_tracking import plan_tracking, summarise_tracking  # noqa: E402
from sfp.sim.simulator import SimulationConfig, simulate  # noqa: E402
from sfp.weather import pvgis  # noqa: E402
from sfp.weather.series import Site, WeatherSeries  # noqa: E402

RESULTS = HERE / "results"
SERIES = RESULTS / "series"

SITE = Site(37.3891, -5.9845, 10.0, name="Seville, ES")
PV_KWP = REFERENCE_SIZING["pv_kwp"]
BATTERIES = (100.0, 250.0, 500.0, 1000.0, 1500.0)
C_RATE = 0.5
CONTROLLERS = ("dispatch", "dispatch-nmpc")
SEASONS = {"winter": 15, "spring": 105, "summer": 172, "autumn": 288}
DAYS = 7.0
MACHINES = ("contactor", "calciner", "electrolyser", "sabatier")


def _weather(day: int, days: float) -> WeatherSeries:
    frame = pvgis.load_weather(SITE.latitude, SITE.longitude, SITE.altitude)
    return WeatherSeries(pvgis.slice_days(frame, day, int(days) + 1), SITE)


def _commitment(log: pd.DataFrame, dt_s: float, days: float) -> dict:
    """Does the plant actually shut machines down, and in what pattern?

    Commitment was the most delicate part of the controller -- the enable gate
    is a near-discontinuity, the rounding rule had to be biased toward
    committing, and an earlier version that never passed commitment at all left
    four shut-down machines drawing 26.4 kW round the clock. So this measures
    both that it works and whether it is doing anything worth the machinery.

    `night` is the thesis test. The project's premise is that the plant's answer
    to nightfall is chemical rather than electrical -- Sabatier running after
    dark off stored hydrogen and CO2 -- so a night duty cycle that collapses
    with pack size is the mechanism, not a curiosity.
    """
    out: dict[str, float] = {}
    night = log["pv_available_W"] < 1.0 if "pv_available_W" in log else None
    for key in MACHINES:
        col = f"enable_{key}"
        if col not in log:
            continue
        on = log[col].to_numpy() >= 0.5
        out[f"duty_{key}"] = float(on.mean())
        if night is not None and night.any():
            out[f"duty_night_{key}"] = float(on[night.to_numpy()].mean())
        starts = int(np.sum(on[1:] & ~on[:-1])) + int(on[0])
        out[f"starts_{key}"] = starts / days
        out[f"uptime_h_{key}"] = (float(on.sum()) * dt_s / 3600.0 / max(starts, 1))
        # Power drawn while NOT committed. Must be zero; this is the regression
        # check on the 26.4 kW failure.
        pcol = f"power.{key}"
        if pcol in log:
            p = np.abs(log[pcol].to_numpy())
            out[f"offdraw_kW_{key}"] = float(p[~on].max() / 1000.0) if (~on).any() else 0.0
            sp = log.get(f"setpoint_{key}")
            if sp is not None:
                idle = on & (sp.to_numpy() < 1e-3)
                out[f"idle_kWh_{key}"] = float(p[idle].sum() * dt_s / 3.6e6)
    if "state.calciner.temperature_K" in log:
        t_kiln = log["state.calciner.temperature_K"].to_numpy()
        out["kiln_hot_h"] = float((t_kiln > 1100.0).sum() * dt_s / 3600.0)
        out["kiln_T_max"] = float(t_kiln.max())
    return out


def _attribution(log: pd.DataFrame) -> dict:
    """Aggregate the per-solve attribution into a few numbers."""
    out: dict[str, float] = {}
    if "nmpc_edit" not in log:
        return out
    edit = log["nmpc_edit"].dropna()
    out["edit_mean"] = float(edit.mean()) if len(edit) else float("nan")
    out["edit_max"] = float(edit.max()) if len(edit) else float("nan")
    for col in log.columns:
        if col.startswith("bind_") or col.startswith("cf_"):
            v = log[col].dropna()
            if len(v):
                out[col] = float(v.mean())
                if col.startswith("cf_"):
                    out[f"{col}_frac"] = float((v > 1e-9).mean())
    return out


def one(battery_kwh: float, controller_name: str, season: str, day: int,
        days: float, econ: Economics) -> dict:
    weather = _weather(day, days)
    plant = build_plant(PV_KWP, battery_kwh, battery_kwh * C_RATE,
                        REFERENCE_SIZING["calciner_kw"])
    controller = controllers_for([controller_name])[0]
    config = SimulationConfig(days=days, start_day=day, dt_s=60.0,
                              control_interval_s=300.0, seed=0,
                              forecast_skill=0.75)
    started = time.perf_counter()
    result = simulate(plant, controller, weather, config, economics=econ)
    log = result.log
    m = compute_metrics(result, econ)

    row: dict = {
        "battery_kwh": battery_kwh, "controller": controller_name,
        "season": season, "days": days,
        "ch4_kg_per_day": m.ch4_kg_per_day, "lcom": m.lcom_eur_per_kg,
        "curtail_frac": m.curtailment_fraction,
        "night_share": m.night_production_fraction,
        "efc_per_day": m.battery_efc / days, "soc_min": m.soc_min,
        "soc_max": m.soc_max, "limiting": m.limiting_subsystem,
        "bus_trips": m.bus_trips, "bus_interventions": m.bus_interventions,
        "shed_kwh": m.shed_kwh, "unserved_kwh": m.unserved_kwh,
        "utilisation": m.utilisation, "capex_eur": m.capex_eur,
        "operating_margin_eur": m.operating_margin_eur,
        "wall_s": time.perf_counter() - started,
    }
    # plan versus realised -- the quantity the old grids got wrong
    s = summarise_tracking(plan_tracking(result, econ))
    if s is not None:
        row.update(planned_eur=s.planned_EUR, realised_eur=s.realised_EUR,
                   plan_bias=s.error_fraction, windows=s.windows,
                   divergence_rms=s.divergence_rms_mean)
    for key in ("nmpc_used", "nmpc_failed", "nmpc_iterations",
                "nmpc_solve_time_s", "nmpc_predicted_shed_kWh"):
        if key in log:
            v = log[key].dropna()
            if len(v):
                row[key] = float(v.mean())
    row.update(_commitment(log, config.dt_s, days))
    row.update(_attribution(log))

    # Keep the full trajectory for the figures that need one, rather than all
    # 40 runs: the commitment Gantt and the day-in-detail panel both want
    # summer, and both extremes of the axis.
    if season == "summer":
        SERIES.mkdir(parents=True, exist_ok=True)
        keep = [c for c in log.columns if c.startswith(("state.", "power.",
                "enable_", "setpoint_", "nmpc_", "bind_", "cf_"))
                or c in ("pv_available_W", "pv_delivered_W")]
        log[keep].to_csv(
            SERIES / f"{controller_name}_{battery_kwh:.0f}kWh_{season}.csv.gz",
            compression="gzip")
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=float, default=DAYS)
    p.add_argument("--seasons", nargs="+", default=list(SEASONS))
    p.add_argument("--batteries", nargs="+", type=float, default=list(BATTERIES))
    p.add_argument("--controllers", nargs="+", default=list(CONTROLLERS))
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    econ = Economics()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "battery_sizing.csv"
    print(f"controller price EUR {econ.p.methane_price_per_kg:.2f}/kg (fixed), "
          f"{PV_KWP:.0f} kWp, {args.days:.0f} d x {len(args.seasons)} seasons\n")

    rows, done = [], 0
    total = len(args.batteries) * len(args.controllers) * len(args.seasons)
    t0 = time.perf_counter()
    for kwh in args.batteries:
        for name in args.controllers:
            for season in args.seasons:
                rows.append(one(kwh, name, season, SEASONS[season],
                                args.days, econ))
                done += 1
                r = rows[-1]
                rate = (time.perf_counter() - t0) / done
                print(f"[{done:2d}/{total}] {kwh:6.0f} kWh {name:13s} {season:7s}"
                      f"  {r['ch4_kg_per_day']:6.1f} kg/d"
                      f"  trips {r['bus_trips']:4d}"
                      f"  bias {r.get('plan_bias', float('nan')):+7.1%}"
                      f"  {r['wall_s']:5.0f} s"
                      f"   ({(total - done) * rate / 60:.0f} min left)", flush=True)
                # written every cell: a sweep that dies at cell 68 should not
                # cost the first 67
                pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}  ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
