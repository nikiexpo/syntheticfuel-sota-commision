"""Experiment 1: profitability across methane price and battery size.

A 5x5 grid per site, evaluated on **dispatch plans rather than closed-loop
simulations**. A 240 h plan costs about twelve seconds; the equivalent
simulation costs tens of minutes, so a grid of this size is affordable one way
and not the other.

Why that substitution is defensible
-----------------------------------
Measured on the reference plant at Seville, comparing what the dispatch LP
predicted against what the plant actually delivered:

    3 days   plan EUR  953.9   plant EUR  946.6   -0.8 %
    7 days   plan EUR 2468.3   plant EUR 2325.8   -5.8 %

Individual replan windows scatter -- the plant leads or lags the plan by an
interval -- but the error averages out rather than accumulating, and inventory
divergence *falls* with horizon length (6.6 % of capacity at three days, 3.6 %
at seven). See `sfp/report/plan_tracking.py`.

So a plan is a good predictor of realised operating profit in aggregate, which
is what a relative comparison needs. What has *not* been established is that the
bias stays constant across the swept axes, which is why every number here is
reported as a relative surface and why a few cells should be spot-checked
against full simulations before anything load-bearing rests on them.

What is actually plotted
------------------------
Net profitability, EUR/day:

    operating profit from the plan   (methane, less battery wear, sorbent,
                                      water and start-ups)
    less annualised capital          (PV + battery + process, at the capital
                                      recovery factor plus fixed opex)

Including capital is not optional. Operating profit alone rises monotonically
with battery size -- a bigger buffer is never operationally worse -- so a grid
without a capital charge has no interior optimum and says nothing.

Design choices worth knowing
----------------------------
**Battery power scales with capacity at C/2** rather than being held fixed. Each
pack is then the same technology at a different size, which is what a sizing
study means; holding power constant would confound the energy axis with a power
axis that gets relatively cheaper as the pack grows.

**Four seasonal windows per cell.** A single summer window would flatter the
high-latitude sites enormously. Winter, spring, summer and autumn are sampled
and averaged, which is crude but at least symmetric.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _plotting import heatmap_panels  # noqa: E402
from sfp.cli import REFERENCE_SIZING, build_plant  # noqa: E402
from sfp.control.base import ControlContext  # noqa: E402
from sfp.control.dispatch import EconomicDispatch  # noqa: E402
from sfp.control.dispatch_model import zi  # noqa: E402
from sfp.economics import Economics  # noqa: E402
from sfp.weather import pvgis  # noqa: E402
from sfp.weather.series import Site, WeatherSeries  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RESULTS = HERE / "results"

#: Three sites chosen for genuinely different solar regimes, not just latitude.
SITES: dict[str, Site] = {
    "Seville": Site(37.3891, -5.9845, 10.0, name="Seville, ES"),
    "London": Site(51.5074, -0.1278, 11.0, name="London, UK"),
    "Gothenburg": Site(57.7089, 11.9746, 10.0, name="Gothenburg, SE"),
}

#: Day-of-year for each seasonal window. Solstices and equinoxes, near enough.
SEASONS: dict[str, int] = {
    "winter": 15, "spring": 105, "summer": 172, "autumn": 288,
}

#: Grid axes. The price range spans the market value of green methane at the low
#: end (about EUR 1.80/kg) to the plant's own levelised cost at the high end
#: (about EUR 6/kg); EUR 3.00 is the current default and lands on a grid line.
PRICES = np.linspace(2.0, 6.0, 5)
#: Battery capacities, kWh. The reference 1500 kWh lands on a grid line.
BATTERIES = np.linspace(500.0, 2500.0, 5)

#: Discharge rate as a fraction of capacity per hour.
C_RATE = 0.5

PLAN_HOURS = 240


def _weather(site: Site, day: int, days: int = 11) -> WeatherSeries:
    frame = pvgis.load_weather(site.latitude, site.longitude, site.altitude)
    return WeatherSeries(pvgis.slice_days(frame, day, days), site)


def _plan_profit_per_day(plant, econ: Economics, weather: WeatherSeries,
                         site: Site) -> tuple[float, float, float, float]:
    """Solve one plan.

    Returns `(operating EUR/day, hours, EFC/day, mean DoD stress)`. The last two
    are what make the battery's *replacement* schedule a function of how hard the
    plan actually works it, rather than an assumption.
    """
    controller = EconomicDispatch(horizon_hours=PLAN_HOURS)
    controller.reset(ControlContext(plant=plant, site=site, dt_s=300.0,
                                    forecast=weather, economics=econ))
    state = {key: sub.initial_state() for key, sub in plant}
    try:
        controller.act(0.0, state, {}, weather)
    except Exception as exc:  # one bad cell must not cost the whole sweep
        print(f"    !! cell failed: {type(exc).__name__}: {exc}", flush=True)
        return float("nan"), 0.0, float("nan"), float("nan")
    plan = controller.plan
    if plan is None or len(plan.stage_profit_EUR) == 0:
        return float("nan"), 0.0, float("nan"), float("nan")

    hours = plan.horizon_s / 3600.0
    days = hours / 24.0
    operating = float(np.sum(plan.stage_profit_EUR)) / days

    battery = plant["battery"]
    throughput = plan.battery_W.sum(axis=1) * 3600.0          # J per interval
    efc_per_day = float(throughput.sum() / (2.0 * battery.nominal_energy_J) / days)
    stress = battery.mean_stress_over(plan.states[:-1, zi("soc")], throughput)
    return operating, hours, efc_per_day, stress


def run(sites: dict[str, Site], quick: bool = False) -> pd.DataFrame:
    econ_base = Economics()
    crf = econ_base.capital_recovery_factor()
    fixed = float(econ_base.p.fixed_opex_fraction)
    seasons = {"summer": SEASONS["summer"]} if quick else SEASONS

    rows = []
    total = len(sites) * len(PRICES) * len(BATTERIES) * len(seasons)
    done = 0
    for site_name, site in sites.items():
        for kwh in BATTERIES:
            plant = build_plant(
                REFERENCE_SIZING["pv_kwp"], float(kwh), float(kwh * C_RATE),
                REFERENCE_SIZING["calciner_kw"],
            )
            capex = econ_base.capex_full(plant)
            first_pack_per_day = capex.total * (crf + fixed) / 365.0
            battery = plant["battery"]
            for price in PRICES:
                econ = Economics()
                econ.p.methane_price_per_kg = float(price)
                per_season, per_efc, per_stress = [], [], []
                for season, day in seasons.items():
                    weather = _weather(site, day)
                    operating, hours, efc, stress = _plan_profit_per_day(
                        plant, econ, weather, site)
                    per_season.append(operating)
                    per_efc.append(efc)
                    per_stress.append(stress)
                    done += 1
                    print(f"  [{done:3d}/{total}] {site_name:11s} "
                          f"{kwh:6.0f} kWh  EUR {price:.2f}/kg  {season:7s} "
                          f"-> operating EUR {operating:7.1f}/day  "
                          f"{efc:.2f} EFC/day", flush=True)
                operating = float(np.nanmean(per_season))
                efc_day = float(np.nanmean(per_efc))
                stress = float(np.nanmean(per_stress))

                # Replacements the annuity does not cover. The pack is amortised
                # over the project's 25 years and does not last them; only the
                # calendar-attributable share is added, because cycling is
                # already paid for through the operating objective.
                repl_pv = battery.uncharged_replacement_PV_EUR(
                    project_years=float(econ_base.p.project_lifetime_years),
                    discount_rate=float(econ_base.p.discount_rate),
                    efc_per_year=efc_day * 365.0, mean_stress=stress,
                )
                capital_per_day = first_pack_per_day + repl_pv * crf / 365.0
                rows.append({
                    "site": site_name,
                    "battery_kwh": float(kwh),
                    "price_EUR_per_kg": float(price),
                    "operating_EUR_per_day": operating,
                    "capital_EUR_per_day": capital_per_day,
                    "first_pack_EUR_per_day": first_pack_per_day,
                    "replacement_EUR_per_day": repl_pv * crf / 365.0,
                    "net_EUR_per_day": operating - capital_per_day,
                    "capex_EUR": capex.total,
                    "efc_per_day": efc_day,
                    "dod_stress": stress,
                    "battery_life_years": battery.life_years(
                        efc_day * 365.0, stress),
                    "seasons": len(per_season),
                    "season_min": float(np.nanmin(per_season)),
                    "season_max": float(np.nanmax(per_season)),
                })
    return pd.DataFrame(rows)


def add_payback(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive discounted payback years for every cell.

    Preferred over the break-even *price* this experiment reported first. Price
    is one of the two swept axes, so solving for the price at which profit
    reaches zero collapses the axis the grid exists to display, and it is
    undefined wherever the zero lies outside the swept range -- which it did at
    two of three sites, where `np.interp` silently clamped to the grid edge and
    returned 6.00 as though it were a result.

    Payback is defined at every cell, takes price as given, and says how long
    rather than merely whether.

    Three adjustments turn the annuitised figures into a cash flow:

    * **Capital is an outlay at t = 0**, not an annual charge, so the capital
      recovery factor is removed and the capex enters whole.
    * **Fixed opex stays annual**, because it is.
    * **The wear accrual is added back and replacements enter as lumps.** The
      controller's operating profit already subtracts
      `cost_per_kWh_delivered` on every kWh moved, which pre-pays a replacement
      that has not happened. A cash-flow model buys the pack when it dies.
    """
    econ = Economics()
    fixed = float(econ.p.fixed_opex_fraction)

    paybacks, cash = [], []
    for _, row in frame.iterrows():
        kwh = float(row["battery_kwh"])
        plant = build_plant(REFERENCE_SIZING["pv_kwp"], kwh, kwh * C_RATE,
                            REFERENCE_SIZING["calciner_kw"])
        battery = plant["battery"]

        wear_per_day = battery.cost_per_efc_EUR() * float(row["efc_per_day"])
        opex_per_day = float(row["capex_EUR"]) * fixed / 365.0
        cash_per_day = float(row["operating_EUR_per_day"]) + wear_per_day - opex_per_day

        paybacks.append(econ.discounted_payback_years(
            float(row["capex_EUR"]), cash_per_day * 365.0,
            replacement_EUR=battery.capex_EUR(),
            replacement_interval_years=float(row["battery_life_years"]),
        ))
        cash.append(cash_per_day)

    out = frame.copy()
    out["cash_EUR_per_day"] = cash
    out["payback_years"] = paybacks
    return out


def plot(frame: pd.DataFrame, sites: list[str]) -> None:
    from matplotlib.colors import TwoSlopeNorm

    rows = [f"{b:.0f}" for b in BATTERIES]
    cols = [f"{p:.2f}" for p in PRICES]
    horizon = float(Economics().p.project_lifetime_years)

    def grid(site, column):
        sub = frame[frame["site"] == site]
        return sub.pivot(index="battery_kwh", columns="price_EUR_per_kg",
                         values=column).to_numpy()

    nets = {s: grid(s, "net_EUR_per_day") for s in sites}
    lo = min(g.min() for g in nets.values())
    hi = max(g.max() for g in nets.values())
    heatmap_panels(
        FIGURES / "relative_profitability_absolute.png", nets,
        row_labels=rows, col_labels=cols,
        title="Net profitability: methane price against battery size "
              "(dispatch plans, seasonally averaged)",
        xlabel="methane price, EUR/kg", ylabel="battery capacity, kWh",
        cbar_label="net profitability, EUR/day "
                   "(operating less annualised capital)",
        cmap="RdYlGn", fmt=".0f",
        norm=TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi) if lo < 0 < hi else None,
        vmin=lo, vmax=hi,
    )

    # Difference from the reference cell, not a ratio: net profitability is
    # negative almost everywhere here, and dividing by a negative reference
    # inverts the ordering so a cell that loses less money reads as worse.
    i_ref = int(np.argmin(np.abs(BATTERIES - 1500.0)))
    j_ref = int(np.argmin(np.abs(PRICES - 3.0)))
    rel = {s: nets[s] - nets[s][i_ref, j_ref] for s in sites}
    span = max(abs(g).max() for g in rel.values())
    heatmap_panels(
        FIGURES / "relative_profitability_relative.png", rel,
        row_labels=rows, col_labels=cols,
        title="Profitability relative to the reference sizing "
              "(1500 kWh, EUR 3.00/kg)",
        xlabel="methane price, EUR/kg", ylabel="battery capacity, kWh",
        cbar_label="EUR/day relative to the reference cell",
        cmap="RdYlGn", fmt="+.0f", vmin=-span, vmax=span,
    )

    if "payback_years" in frame:
        heatmap_panels(
            FIGURES / "relative_profitability_payback.png",
            {s: grid(s, "payback_years") for s in sites},
            row_labels=rows, col_labels=cols,
            title="Time to break even: discounted payback at 7 %, "
                  "with battery replacement as it falls due",
            xlabel="methane price, EUR/kg", ylabel="battery capacity, kWh",
            cbar_label=f"discounted payback, years "
                       f"(dark red = not repaid within {horizon:.0f} yr)",
            cmap="viridis_r", fmt=".1f", vmin=0.0, vmax=horizon,
            mask_invalid=True,
        )

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="summer window only, for a fast smoke test")
    parser.add_argument("--sites", nargs="+", default=list(SITES))
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    sites = {k: SITES[k] for k in args.sites}
    frame = run(sites, quick=args.quick)

    frame = add_payback(frame)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / ("relative_profitability_quick.csv" if args.quick
                     else "relative_profitability.csv")
    frame.to_csv(out, index=False)
    plot(frame, list(sites))

    print(f"\nwrote {out}")
    print(f"wrote {FIGURES}/relative_profitability_*.png\n")
    for site in sites:
        sub = frame[frame["site"] == site]
        best = sub.loc[sub["net_EUR_per_day"].idxmax()]
        repaid = sub[np.isfinite(sub["payback_years"])]
        fastest = (f"{repaid['payback_years'].min():.1f} yr at "
                   f"{repaid.loc[repaid['payback_years'].idxmin(), 'battery_kwh']:.0f} kWh @ "
                   f"EUR {repaid.loc[repaid['payback_years'].idxmin(), 'price_EUR_per_kg']:.2f}/kg"
                   if len(repaid) else "never repaid at any cell")
        print(f"{site:11s} best net {best['net_EUR_per_day']:+.0f} EUR/day at "
              f"{best['battery_kwh']:.0f} kWh @ EUR {best['price_EUR_per_kg']:.2f}/kg"
              f"   |  fastest payback: {fastest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
