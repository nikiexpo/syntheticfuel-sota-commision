"""Experiment 2: profitability across methane price and PV array size.

The natural successor to `relative_profitability.py`, and probably the more
informative of the two. PV is a larger capital item than the battery at every
sizing tested, and it is the binding resource at two of the three sites, so the
sizing question really lives on this axis.

Same method throughout: 240 h dispatch plans rather than closed-loop
simulations, four seasonal windows per cell, discounted payback as the headline
metric. See `RELATIVE_PROFITABILITY.md` sections 1 and 4 for the justification
and the caveats, all of which carry over unchanged.

Why the battery is held at 500 kWh
----------------------------------
Not at the reference 1500 kWh. Experiment 1 found net profitability falling
monotonically with capacity at every site and every price, with the optimum at
or below the smallest pack tested. Sweeping PV at 1500 kWh would study a plant
already known to be mis-sized, and would drag a fixed ~EUR 80/day penalty
through every cell of this grid.

The consequence is that the two experiments are **not** directly comparable
cell-for-cell: this one is a better plant at the same PV. Where they share a
point -- 1100 kWp -- the difference is the battery, and it is worth about
EUR 46-68/day depending on site.

Why not precompile
------------------
Measured at 240 h: matrix assembly is 0.017 s against 1.05 s in the solver, so
about 1.5 % of the work. Unlike the inner NMPC -- where compiling the CasADi
expression graph was worth roughly 4x -- an LP has no graph to compile. The cost
is branch and bound over the kiln's 24 integer hours, and `scipy.optimize.milp`
exposes no warm start. Fixing `y` from an adjacent cell would work, since PV
size barely moves the kiln's temperature trajectory, but it is an approximation
and would need validating against full solves before any number rested on it.
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

from relative_profitability import (  # noqa: E402
    C_RATE, PRICES, SEASONS, SITES, _plan_profit_per_day, _weather,
)
from _plotting import heatmap_panels  # noqa: E402
from sfp.cli import REFERENCE_SIZING, build_plant  # noqa: E402
from sfp.economics import Economics  # noqa: E402
from sfp.weather.series import Site  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RESULTS = HERE / "results"

#: Array sizes, kWp. Spans clearly undersized to clearly oversized, and the
#: reference 1100 kWp lands on a grid line.
ARRAYS = np.linspace(300.0, 1900.0, 5)

#: Held fixed. See the module docstring.
BATTERY_KWH = 500.0

PLAN_HOURS = 240


def run(sites: dict[str, Site], quick: bool = False) -> pd.DataFrame:
    econ_base = Economics()
    crf = econ_base.capital_recovery_factor()
    fixed = float(econ_base.p.fixed_opex_fraction)
    seasons = {"summer": SEASONS["summer"]} if quick else SEASONS

    rows = []
    total = len(sites) * len(PRICES) * len(ARRAYS) * len(seasons)
    done = 0
    for site_name, site in sites.items():
        for kwp in ARRAYS:
            plant = build_plant(float(kwp), BATTERY_KWH, BATTERY_KWH * C_RATE,
                                REFERENCE_SIZING["calciner_kw"])
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
                          f"{kwp:6.0f} kWp  EUR {price:.2f}/kg  {season:7s} "
                          f"-> operating EUR {operating:7.1f}/day  "
                          f"{efc:.2f} EFC/day", flush=True)
                operating = float(np.nanmean(per_season))
                efc_day = float(np.nanmean(per_efc))
                stress = float(np.nanmean(per_stress))

                repl_pv = battery.uncharged_replacement_PV_EUR(
                    project_years=float(econ_base.p.project_lifetime_years),
                    discount_rate=float(econ_base.p.discount_rate),
                    efc_per_year=efc_day * 365.0, mean_stress=stress,
                )
                rows.append({
                    "site": site_name,
                    "pv_kwp": float(kwp),
                    "battery_kwh": BATTERY_KWH,
                    "price_EUR_per_kg": float(price),
                    "operating_EUR_per_day": operating,
                    "capital_EUR_per_day": first_pack_per_day + repl_pv * crf / 365.0,
                    "first_pack_EUR_per_day": first_pack_per_day,
                    "replacement_EUR_per_day": repl_pv * crf / 365.0,
                    "net_EUR_per_day": operating - first_pack_per_day
                                       - repl_pv * crf / 365.0,
                    "capex_EUR": capex.total,
                    "capex_pv_EUR": capex.pv,
                    "capex_process_EUR": capex.process,
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
    """Discounted payback per cell. Same cash-flow treatment as experiment 1."""
    econ = Economics()
    fixed = float(econ.p.fixed_opex_fraction)

    paybacks, cash = [], []
    for _, row in frame.iterrows():
        plant = build_plant(float(row["pv_kwp"]), BATTERY_KWH,
                            BATTERY_KWH * C_RATE, REFERENCE_SIZING["calciner_kw"])
        battery = plant["battery"]
        wear_per_day = battery.cost_per_efc_EUR() * float(row["efc_per_day"])
        opex_per_day = float(row["capex_EUR"]) * fixed / 365.0
        cash_per_day = (float(row["operating_EUR_per_day"]) + wear_per_day
                        - opex_per_day)
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

    rows = [f"{a:.0f}" for a in ARRAYS]
    cols = [f"{p:.2f}" for p in PRICES]
    horizon = float(Economics().p.project_lifetime_years)

    def grid(site, column):
        sub = frame[frame["site"] == site]
        return sub.pivot(index="pv_kwp", columns="price_EUR_per_kg",
                         values=column).to_numpy()

    nets = {s: grid(s, "net_EUR_per_day") for s in sites}
    lo = min(g.min() for g in nets.values())
    hi = max(g.max() for g in nets.values())
    heatmap_panels(
        FIGURES / "pv_sizing_absolute.png", nets,
        row_labels=rows, col_labels=cols,
        title=f"Net profitability: methane price against PV array "
              f"({BATTERY_KWH:.0f} kWh battery, dispatch plans, seasonally averaged)",
        xlabel="methane price, EUR/kg", ylabel="PV array, kWp",
        cbar_label="net profitability, EUR/day "
                   "(operating less annualised capital)",
        cmap="RdYlGn", fmt=".0f",
        norm=TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi) if lo < 0 < hi else None,
        vmin=lo, vmax=hi,
    )

    if "payback_years" in frame:
        heatmap_panels(
            FIGURES / "pv_sizing_payback.png",
            {s: grid(s, "payback_years") for s in sites},
            row_labels=rows, col_labels=cols,
            title="Time to break even: discounted payback at 7 %, "
                  "with battery replacement as it falls due",
            xlabel="methane price, EUR/kg", ylabel="PV array, kWp",
            cbar_label=f"discounted payback, years "
                       f"(dark red = not repaid within {horizon:.0f} yr)",
            cmap="viridis_r", fmt=".1f", vmin=0.0, vmax=horizon,
            mask_invalid=True,
        )

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--sites", nargs="+", default=list(SITES))
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    sites = {k: SITES[k] for k in args.sites}
    frame = add_payback(run(sites, quick=args.quick))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / ("pv_sizing_quick.csv" if args.quick else "pv_sizing.csv")
    frame.to_csv(out, index=False)
    plot(frame, list(sites))

    print(f"\nwrote {out}\nwrote {FIGURES}/pv_sizing_*.png\n")
    for site in sites:
        sub = frame[frame["site"] == site]
        best = sub.loc[sub["net_EUR_per_day"].idxmax()]
        repaid = sub[np.isfinite(sub["payback_years"])]
        fastest = (
            f"{repaid['payback_years'].min():.1f} yr at "
            f"{repaid.loc[repaid['payback_years'].idxmin(), 'pv_kwp']:.0f} kWp @ "
            f"EUR {repaid.loc[repaid['payback_years'].idxmin(), 'price_EUR_per_kg']:.2f}/kg"
            if len(repaid) else "never repaid at any cell")
        print(f"{site:11s} best net {best['net_EUR_per_day']:+.0f} EUR/day at "
              f"{best['pv_kwp']:.0f} kWp @ EUR {best['price_EUR_per_kg']:.2f}/kg"
              f"   |  fastest payback: {fastest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
