"""Experiment B: joint PV x battery sizing, closed loop, two sites.

Closed loop throughout, for the reason set out in `BATTERY_SIZING.md` §1: the
outer LP's planned profit is flat across a 15x range of battery capacity while
realised profit rises 2.5x, so a grid scored on plans measures a different shape
from the one the plant has.

Design decisions, and why
-------------------------
**One controller price, EUR 6.00.** Measured directly (`price_strategy.py`): a
controller at EUR 4 and one at EUR 8 differ by under 1 % on methane, EFC,
curtailment and LCOM, with commitment traces that overlap exactly. The price is
saturated across at least a 2x range, because the night-time shadow price of
electricity scales with the product price -- raising one lifts both sides of the
battery-wear comparison together, so the cycling decision does not move. There
is a threshold below roughly EUR 4 where the optimiser stops cycling, but no
slope above it. Sweeping the controller price would therefore have bought 36
extra runs and no new behaviour.

**Capex is not simulated.** Capital cost never enters the controller's
objective, verified in the earlier capex study where plans came back
bit-identical at 2200 and 1100 EUR/kW. The four multipliers are exact arithmetic
over these runs.

**`dispatch-nmpc` only.** `BATTERY_SIZING.md` §3.1 established that the outer
layer alone is systematically wrong for sizing, so re-running it here would
double the cost to re-demonstrate a settled point.

**Two seasons, weighted.** Summer and winter only, combined as
`0.63*summer + 0.37*winter`. A straight mean understates annual production by
3-11 %, and -- the part that matters -- the understatement *grows with pack
size*, tilting the grid by 7.3 points in the same direction as the error that
inverted the old conclusions. The weighting zeroes the level error (+0.31 %) and
leaves 6.2 points of residual tilt, which is an order of magnitude smaller than
the 43-point spread it replaces but is **not zero**: it under-credits large
packs, so any conclusion resting on a difference narrower than ~6 % is inside
the noise. Weights were fitted on the four-season Seville data in
`results/battery_sizing.csv`.
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
sys.path.insert(0, str(HERE))

from battery_sizing import _attribution, _commitment  # noqa: E402
from sfp.cli import REFERENCE_SIZING, build_plant, controllers_for  # noqa: E402
from sfp.economics import Economics  # noqa: E402
from sfp.report.metrics import compute_metrics  # noqa: E402
from sfp.sim.simulator import SimulationConfig, simulate  # noqa: E402
from sfp.weather import pvgis  # noqa: E402
from sfp.weather.series import Site, WeatherSeries  # noqa: E402

RESULTS = HERE / "results"
SERIES = RESULTS / "series"

SITES = {
    "Seville": Site(37.3891, -5.9845, 10.0, name="Seville, ES"),
    "London": Site(51.5074, -0.1278, 11.0, name="London, UK"),
}
PV_KWP = (700.0, 1100.0, 1500.0)
#: Reaches 2500 because the battery axis in experiment A still had net profit
#: rising at 1500 kWh -- the optimum sat on the grid edge and was not located.
BATTERIES = (500.0, 1500.0, 2500.0)
SEASONS = {"summer": 172, "winter": 15}
#: See the module docstring. Fitted on the four-season data, not assumed.
SEASON_WEIGHT = {"summer": 0.63, "winter": 0.37}
PRICE = 6.0
C_RATE = 0.5
DAYS = 7.0


def one(site_name: str, site: Site, pv_kwp: float, battery_kwh: float,
        season: str, days: float) -> dict:
    frame = pvgis.load_weather(site.latitude, site.longitude, site.altitude)
    day = SEASONS[season]
    weather = WeatherSeries(pvgis.slice_days(frame, day, int(days) + 1), site)
    plant = build_plant(pv_kwp, battery_kwh, battery_kwh * C_RATE,
                        REFERENCE_SIZING["calciner_kw"])
    econ = Economics()
    econ.p.methane_price_per_kg = PRICE
    config = SimulationConfig(days=days, start_day=day, dt_s=60.0,
                              control_interval_s=300.0, seed=0,
                              forecast_skill=0.75)
    started = time.perf_counter()
    result = simulate(plant, controllers_for(["dispatch-nmpc"])[0], weather,
                      config, economics=econ)
    log = result.log
    m = compute_metrics(result, econ)
    capex = econ.capex_full(plant)

    row: dict = {
        "site": site_name, "pv_kwp": pv_kwp, "battery_kwh": battery_kwh,
        "season": season, "weight": SEASON_WEIGHT[season], "days": days,
        "price": PRICE,
        "ch4_kg_per_day": m.ch4_kg_per_day, "lcom": m.lcom_eur_per_kg,
        "curtail_frac": m.curtailment_fraction,
        "night_share": m.night_production_fraction,
        "efc_per_day": m.battery_efc / days, "soc_min": m.soc_min,
        "soc_max": m.soc_max, "limiting": m.limiting_subsystem,
        "bus_trips": m.bus_trips, "bus_interventions": m.bus_interventions,
        "shed_kwh": m.shed_kwh, "unserved_kwh": m.unserved_kwh,
        "utilisation": m.utilisation,
        "operating_margin_eur": m.operating_margin_eur,
        # kept separately so the capex multipliers can be applied to the
        # process chain alone, which is the only part with a learning curve
        "capex_eur": capex.total, "capex_pv_eur": capex.pv,
        "capex_battery_eur": capex.battery, "capex_process_eur": capex.process,
        "wall_s": time.perf_counter() - started,
    }
    for key in ("nmpc_used", "nmpc_failed", "nmpc_iterations",
                "nmpc_solve_time_s", "nmpc_predicted_shed_kWh"):
        if key in log:
            v = log[key].dropna()
            if len(v):
                row[key] = float(v.mean())
    row.update(_commitment(log, config.dt_s, days))
    row.update(_attribution(log))

    # Trajectories only for the summer runs at each site: enough for the
    # timeline and commitment figures without storing 36 of them.
    if season == "summer":
        SERIES.mkdir(parents=True, exist_ok=True)
        keep = [c for c in log.columns
                if c.startswith(("state.", "power.", "enable_", "setpoint_",
                                 "nmpc_", "bind_", "cf_"))
                or c in ("pv_available_W", "pv_delivered_W",
                         "battery_charge_W", "battery_discharge_W")]
        log[keep].to_csv(
            SERIES / f"joint_{site_name}_{pv_kwp:.0f}kWp_"
                     f"{battery_kwh:.0f}kWh_{season}.csv.gz", compression="gzip")
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=float, default=DAYS)
    p.add_argument("--sites", nargs="+", default=list(SITES))
    p.add_argument("--pv", nargs="+", type=float, default=list(PV_KWP))
    p.add_argument("--batteries", nargs="+", type=float, default=list(BATTERIES))
    p.add_argument("--seasons", nargs="+", default=list(SEASONS))
    p.add_argument("--fresh", action="store_true",
                   help="ignore any existing CSV and start over")
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "joint_sizing.csv"

    # Resume from the per-cell CSV. An 8.8 h run is long enough to be killed
    # part way through (this one was, at cell 23 of 36), and without this a
    # restart re-runs every completed cell to reach the missing ones.
    rows: list[dict] = []
    finished: set = set()
    if out.exists() and not args.fresh:
        prior = pd.read_csv(out)
        rows = prior.to_dict("records")
        finished = set(zip(prior["site"], prior["pv_kwp"].astype(float),
                           prior["battery_kwh"].astype(float), prior["season"]))
        print(f"resuming: {len(finished)} cells already in {out.name}",
              flush=True)

    total = (len(args.sites) * len(args.pv) * len(args.batteries)
             * len(args.seasons))
    print(f"joint sizing: {total} runs, controller EUR {PRICE:.2f}/kg, "
          f"{args.days:.0f} d, dispatch-nmpc\n", flush=True)

    done, t0 = len(finished), time.perf_counter()
    for site_name in args.sites:
        for pv in args.pv:
            for kwh in args.batteries:
                for season in args.seasons:
                    if (site_name, float(pv), float(kwh), season) in finished:
                        continue
                    rows.append(one(site_name, SITES[site_name], pv, kwh,
                                    season, args.days))
                    done += 1
                    r = rows[-1]
                    rate = (time.perf_counter() - t0) / done
                    print(f"[{done:2d}/{total}] {site_name:8s} {pv:6.0f} kWp "
                          f"{kwh:6.0f} kWh {season:7s}"
                          f"  {r['ch4_kg_per_day']:6.1f} kg/d"
                          f"  trips {r['bus_trips']:5d}"
                          f"  used {r.get('nmpc_used', float('nan')):.3f}"
                          f"  {r['wall_s']:5.0f} s"
                          f"   ({(total - done) * rate / 60:.0f} min left)",
                          flush=True)
                    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}  ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
