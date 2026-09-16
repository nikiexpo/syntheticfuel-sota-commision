"""Experiment 3: profitability against process-chain capital cost.

The process chain -- electrolyser stack and BoP, air contactor, calciner,
Sabatier reactor -- is the largest capital item in the plant at every sizing
tested, larger than the array, and it is priced by the single weakest number in
the model: `process_capex_per_kw = 2200`, `provenance: assumed`, applied to one
aggregate kW rating.

It is also the item with the most credible path to falling. The electrolyser is
62 % of that rating, and electrolyser capex is where published learning curves
are steepest. PV modules fell roughly an order of magnitude over two decades on
exactly this mechanism.

**No new solves.** Capital cost never enters the controller's objective, so the
dispatch plans are bit-identical at every multiplier -- verified directly:
Gothenburg at 1100 kWp returns 760.6586 EUR/day operating and 0.8573 EFC/day at
both 2200 and 1100 EUR/kW. The sweep is therefore exact arithmetic over the
plans already solved for `pv_sizing.py`, and costs seconds rather than the
ninety minutes that grid took.

Fixed at the reference point of the other two experiments: 1100 kWp array,
500 kWh battery.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _plotting import heatmap_panels  # noqa: E402
from relative_profitability import C_RATE, PRICES  # noqa: E402
from sfp.cli import REFERENCE_SIZING, build_plant  # noqa: E402
from sfp.economics import Economics  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RESULTS = HERE / "results"
SOURCE = RESULTS / "pv_sizing.csv"

#: Fraction of the assumed 2200 EUR/kW. 0.25 is an aggressive but not absurd
#: floor for a mass-produced electrolyser on a 2035 horizon; 1.00 is today's
#: assumption.
MULTIPLIERS = np.arange(0.25, 1.001, 0.125)

PV_KWP = 1100.0
BATTERY_KWH = 500.0


def run() -> pd.DataFrame:
    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found -- run experiments/pv_sizing.py first")

    econ = Economics()
    crf = econ.capital_recovery_factor()
    fixed = float(econ.p.fixed_opex_fraction)
    plant = build_plant(PV_KWP, BATTERY_KWH, BATTERY_KWH * C_RATE,
                        REFERENCE_SIZING["calciner_kw"])
    battery = plant["battery"]

    source = pd.read_csv(SOURCE)
    source = source[source["pv_kwp"] == PV_KWP]
    if source.empty:
        raise SystemExit(f"no rows at {PV_KWP:.0f} kWp in {SOURCE}")

    rows = []
    for _, r in source.iterrows():
        base_total = float(r["capex_EUR"])
        process = float(r["capex_process_EUR"])
        repl_pv = battery.uncharged_replacement_PV_EUR(
            project_years=float(econ.p.project_lifetime_years),
            discount_rate=float(econ.p.discount_rate),
            efc_per_year=float(r["efc_per_day"]) * 365.0,
            mean_stress=float(r["dod_stress"]),
        )
        wear_per_day = battery.cost_per_efc_EUR() * float(r["efc_per_day"])

        for m in MULTIPLIERS:
            capex = base_total - process * (1.0 - float(m))
            first_pack = capex * (crf + fixed) / 365.0
            operating = float(r["operating_EUR_per_day"])
            cash_per_day = operating + wear_per_day - capex * fixed / 365.0
            rows.append({
                "site": r["site"],
                "process_capex_multiplier": float(m),
                "process_capex_per_kw": 2200.0 * float(m),
                "price_EUR_per_kg": float(r["price_EUR_per_kg"]),
                "operating_EUR_per_day": operating,
                "capex_EUR": capex,
                "capital_EUR_per_day": first_pack + repl_pv * crf / 365.0,
                "net_EUR_per_day": operating - first_pack - repl_pv * crf / 365.0,
                "payback_years": econ.discounted_payback_years(
                    capex, cash_per_day * 365.0,
                    replacement_EUR=battery.capex_EUR(),
                    replacement_interval_years=float(r["battery_life_years"]),
                ),
            })
    return pd.DataFrame(rows)


def plot(frame: pd.DataFrame) -> None:
    from matplotlib.colors import TwoSlopeNorm

    sites = list(dict.fromkeys(frame["site"]))
    rows = [f"{m:.2f}" for m in MULTIPLIERS]
    cols = [f"{p:.2f}" for p in PRICES]

    def grid(site, column):
        sub = frame[frame["site"] == site]
        return sub.pivot(index="process_capex_multiplier",
                         columns="price_EUR_per_kg", values=column).to_numpy()

    nets = {s: grid(s, "net_EUR_per_day") for s in sites}
    lo = min(g.min() for g in nets.values())
    hi = max(g.max() for g in nets.values())
    heatmap_panels(
        FIGURES / "process_capex_absolute.png", nets,
        row_labels=rows, col_labels=cols,
        title="Net profitability against process-chain capital cost "
              f"({PV_KWP:.0f} kWp, {BATTERY_KWH:.0f} kWh)",
        xlabel="methane price, EUR/kg",
        ylabel="process capex, fraction of EUR 2200/kW",
        cbar_label="net profitability, EUR/day",
        cmap="RdYlGn", fmt=".0f",
        norm=TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi) if lo < 0 < hi else None,
        vmin=lo, vmax=hi,
    )

    horizon = float(Economics().p.project_lifetime_years)
    heatmap_panels(
        FIGURES / "process_capex_payback.png",
        {s: grid(s, "payback_years") for s in sites},
        row_labels=rows, col_labels=cols,
        title="Time to break even against process-chain capital cost",
        xlabel="methane price, EUR/kg",
        ylabel="process capex, fraction of EUR 2200/kW",
        cbar_label=f"discounted payback, years "
                   f"(dark red = not repaid within {horizon:.0f} yr)",
        cmap="viridis_r", fmt=".1f", vmin=0.0, vmax=horizon,
        mask_invalid=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    frame = run()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "process_capex.csv"
    frame.to_csv(out, index=False)
    plot(frame)
    print(f"wrote {out}\nwrote {FIGURES}/process_capex_*.png\n")

    for site in dict.fromkeys(frame["site"]):
        sub = frame[frame["site"] == site]
        repaid = sub[np.isfinite(sub["payback_years"])]
        if repaid.empty:
            print(f"{site:11s} never repays at any multiplier or price")
            continue
        # the cheapest process cost that repays at the market-ish price
        best = repaid.loc[repaid["payback_years"].idxmin()]
        at3 = repaid[repaid["price_EUR_per_kg"] == 3.0]
        line = (f"needs multiplier <= {at3['process_capex_multiplier'].max():.2f} "
                f"at EUR 3.00/kg" if not at3.empty
                else "does not repay at EUR 3.00/kg at any multiplier")
        print(f"{site:11s} fastest {best['payback_years']:.1f} yr at "
              f"x{best['process_capex_multiplier']:.2f}, "
              f"EUR {best['price_EUR_per_kg']:.2f}/kg   |  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
