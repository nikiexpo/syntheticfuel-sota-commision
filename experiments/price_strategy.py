"""Does the controller's methane price change its strategy between 6 and 8 EUR?

The joint-sizing experiment costs twice as much if it does, because the price
then has to be a *controller* price rather than a market price applied
arithmetically afterwards. The argument for running both is that a controller
which undervalues methane will not cycle the battery hard enough to maximise
profit, so the strategy shift would be invisible if the price were only applied
at reporting time. This measures whether that shift is real and how large it is.

One configuration, both prices, identical weather and seed, so any difference is
the price and nothing else.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sfp.cli import REFERENCE_SIZING, build_plant, controllers_for  # noqa: E402
from sfp.economics import Economics  # noqa: E402
from sfp.report.metrics import compute_metrics  # noqa: E402
from sfp.report.plots import COLOURS  # noqa: E402
from sfp.sim.simulator import SimulationConfig, simulate  # noqa: E402
from sfp.weather import pvgis  # noqa: E402
from sfp.weather.series import Site, WeatherSeries  # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
SITE = Site(37.3891, -5.9845, 10.0, name="Seville, ES")
#: Deliberately wide. A first attempt at 6 vs 8 moved every metric by
#: under 1 % -- the controller is saturated in that range. 4.00 sits near
#: the battery-wear crossover (wear 0.0565 EUR/kWh against a night-time
#: shadow price of ~0.043), which is where the optimiser stops cycling the
#: pack at all, so this pair brackets a real change of regime rather than
#: a marginal one.
PRICES = (4.0, 8.0)
STYLE = {4.0: ("#4C72B0", "-"), 8.0: ("#C1453B", "-")}


def run(price: float, pv_kwp: float, battery_kwh: float, day: int,
        days: float) -> tuple[pd.DataFrame, object]:
    frame = pvgis.load_weather(SITE.latitude, SITE.longitude, SITE.altitude)
    weather = WeatherSeries(pvgis.slice_days(frame, day, int(days) + 1), SITE)
    plant = build_plant(pv_kwp, battery_kwh, battery_kwh * 0.5,
                        REFERENCE_SIZING["calciner_kw"])
    econ = Economics()
    econ.p.methane_price_per_kg = price
    config = SimulationConfig(days=days, start_day=day, dt_s=60.0,
                              control_interval_s=300.0, seed=0,
                              forecast_skill=0.75)
    t0 = time.perf_counter()
    result = simulate(plant, controllers_for(["dispatch-nmpc"])[0], weather,
                      config, economics=econ)
    print(f"  EUR {price:.2f}/kg done in {time.perf_counter() - t0:.0f} s",
          flush=True)
    return result.log, compute_metrics(result, econ)


def figure(logs: dict, mets: dict, pv_kwp: float, battery_kwh: float) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(3, 2, figsize=(14, 8.4), constrained_layout=True)
    a = ax.ravel()

    for price, log in logs.items():
        colour, ls = STYLE[price]
        t = pd.to_datetime(log.index)
        lab = f"EUR {price:.0f}/kg"
        dt_h = 1.0 / 60.0

        a[0].plot(t, log["state.battery.soc"] if "state.battery.soc" in log
                  else log["battery_soc"], color=colour, ls=ls, lw=1.1, label=lab)
        # Cumulative energy pushed through the pack -- the direct measure of
        # how hard the controller is willing to work it, and a sharper test
        # than SoC: shallow frequent cycling and deep daily cycling trace
        # similar SoC envelopes but differ greatly in throughput.
        #
        # Taken from the `efc` state rather than `power.battery`, which the
        # simulator left identically zero until this was found (see
        # `sfp/sim/simulator.py`). One EFC is a full charge plus a full
        # discharge, hence the factor of two.
        efc = (log["state.battery.efc"] if "state.battery.efc" in log
               else log["battery_efc"]).to_numpy()
        a[1].plot(t, 2.0 * (efc - efc[0]) * battery_kwh, color=colour, ls=ls,
                  lw=1.3, label=lab)
        curt = (log["pv_available_W"] - log["pv_delivered_W"]).clip(lower=0)
        a[2].plot(t, np.cumsum(curt.to_numpy()) * dt_h / 1e3, color=colour,
                  ls=ls, lw=1.3, label=lab)
        ch4 = (log["state.sabatier.ch4_kg"] if "state.sabatier.ch4_kg" in log
               else log["ch4_total_kg"])
        a[3].plot(t, ch4, color=colour, ls=ls, lw=1.3, label=lab)
        # how much of the plant is committed, hour by hour
        keys = ("contactor", "calciner", "electrolyser", "sabatier")
        com = sum((log[f"enable_{k}"] >= 0.5).astype(int) for k in keys)
        a[4].plot(t, com.rolling(60, min_periods=1).mean(), color=colour, ls=ls,
                  lw=1.1, label=lab)

    for axis, title, ylab in (
            (a[0], "Battery state of charge", "SoC"),
            (a[1], "Cumulative battery throughput", "kWh"),
            (a[2], "Cumulative curtailed energy", "kWh"),
            (a[3], "Cumulative methane", "kg"),
            (a[4], "Machines committed (1 h mean)", "count")):
        axis.set_title(title)
        axis.set_ylabel(ylab)
        axis.legend(frameon=False, fontsize=8)
        axis.xaxis.set_major_locator(mdates.DayLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    a[0].set_ylim(0, 1)

    # --- the numbers, as a difference bar chart
    keys = [("ch4_kg_per_day", "methane\nkg/day"), ("battery_efc", "battery\nEFC"),
            ("night_production_fraction", "night\nshare"),
            ("curtailment_fraction", "curtailed"),
            ("lcom_eur_per_kg", "LCOM\nEUR/kg"), ("bus_trips", "bus\ntrips")]
    lo, hi = mets[PRICES[0]], mets[PRICES[1]]
    delta = []
    for k, _ in keys:
        v0, v1 = getattr(lo, k), getattr(hi, k)
        delta.append(100.0 * (v1 / v0 - 1.0) if v0 else 0.0)
    bars = a[5].bar(range(len(keys)), delta,
                    color=["#C1453B" if d > 0 else "#4C72B0" for d in delta])
    a[5].axhline(0, color=COLOURS["grid"], lw=0.8)
    a[5].set_xticks(range(len(keys)))
    a[5].set_xticklabels([n for _, n in keys], fontsize=7.5)
    a[5].set_ylabel(f"EUR {PRICES[1]:.0f} vs EUR {PRICES[0]:.0f}, %")
    a[5].set_title("Strategy difference")
    for b, d in zip(bars, delta):
        a[5].text(b.get_x() + b.get_width() / 2,
                  d + (1 if d >= 0 else -1) * 0.4,
                  f"{d:+.1f}%", ha="center",
                  va="bottom" if d >= 0 else "top", fontsize=7.5)

    fig.suptitle(f"Controller methane price: does it change the strategy?   "
                 f"({pv_kwp:.0f} kWp, {battery_kwh:.0f} kWh, Seville, summer, 7 d)",
                 fontweight="bold")
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "price_strategy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pv", type=float, default=1100.0)
    p.add_argument("--battery", type=float, default=1500.0)
    p.add_argument("--day", type=int, default=172)
    p.add_argument("--days", type=float, default=7.0)
    a = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    logs, mets = {}, {}
    print(f"{a.pv:.0f} kWp, {a.battery:.0f} kWh, Seville, {a.days:.0f} days\n")
    for price in PRICES:
        logs[price], mets[price] = run(price, a.pv, a.battery, a.day, a.days)

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for price, m in mets.items():
        rows.append({"price": price, "ch4_kg_per_day": m.ch4_kg_per_day,
                     "battery_efc": m.battery_efc, "soc_min": m.soc_min,
                     "night_share": m.night_production_fraction,
                     "curtail_frac": m.curtailment_fraction,
                     "lcom": m.lcom_eur_per_kg, "bus_trips": m.bus_trips,
                     "shed_kwh": m.shed_kwh})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "price_strategy.csv", index=False)
    print()
    print(df.round(3).to_string(index=False))
    lo, hi = mets[PRICES[0]], mets[PRICES[1]]
    print(f"\nmethane  {100 * (hi.ch4_kg_per_day / lo.ch4_kg_per_day - 1):+.2f} %"
          f"   EFC {100 * (hi.battery_efc / lo.battery_efc - 1):+.2f} %"
          f"   curtail {100 * (hi.curtailment_fraction / max(lo.curtailment_fraction, 1e-9) - 1):+.2f} %")
    print("wrote", figure(logs, mets, a.pv, a.battery))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
