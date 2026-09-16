"""Experiment 4: the joint PV x battery sizing grid, Seville only.

Experiments 1 and 2 each swept one design axis with the other pinned -- battery
at 1100 kWp, PV at 500 kWh -- and `PV_SIZING.md` section 3 flagged the obvious
hole: the two axes interact, because a larger array makes storage more useful
and a larger store makes an oversized array less wasteful. Neither one-dimensional
grid can locate a joint optimum that sits off its own line.

This grid closes that. **Seville only**: the cloudy sites cost three times the
solve budget to re-confirm a conclusion both of the earlier grids already
returned for them (cheapest battery, and no cell that repays), and the
interesting interaction -- curtailment at large arrays being recovered by
storage -- is strongest where there is surplus to store.

Axes
----
**PV** reuses experiment 2's five array sizes exactly, so every row has a
directly comparable predecessor.

**Battery** does not reuse experiment 1's. That grid ran 500-2500 kWh and found
net profitability falling monotonically, optimum pinned at the smallest size
tested and therefore *not located*. This axis runs 100-1500 kWh instead, so the
optimum has somewhere to be. 500 and 1500 kWh are retained as grid lines because
they are the two anchors the earlier experiments used.

**Price** becomes the panel dimension rather than a grid axis. Experiment 2
found the optimal array moving by a factor of three or four across the price
range, so a single-price sizing grid would be answering a much narrower question
than it appears to. Three panels at EUR 3.00, 4.50 and 6.00/kg.

Cross-checks
------------
Two cells have known answers and must reproduce them:

* (1100 kWp, 500 kWh) at EUR 3.00 and 6.00 -- against `pv_sizing.csv`
* (1100 kWp, 1500 kWh) at EUR 3.00 and 6.00 -- against
  `relative_profitability.csv`

`--check` prints both comparisons against whatever this run produced. They
should agree to the last decimal: same plant, same weather, same solver.

Method is unchanged from experiments 1 and 2 -- 240 h dispatch plans rather than
closed-loop simulations, four seasonal windows per cell, discounted payback at
7 % with battery replacement bought when it falls due. See
`RELATIVE_PROFITABILITY.md` sections 1 and 4 for the justification and the
caveats, which carry over in full.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _plotting import heatmap_panels  # noqa: E402
from relative_profitability import (  # noqa: E402
    C_RATE, SEASONS, SITES, _plan_profit_per_day, _weather,
)
from sfp.cli import REFERENCE_SIZING, build_plant  # noqa: E402
from sfp.economics import Economics  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RESULTS = HERE / "results"

SITE_NAME = "Seville"

#: Array sizes, kWp. Identical to experiment 2's axis.
ARRAYS = np.array([300.0, 700.0, 1100.0, 1500.0, 1900.0])

#: Capacities, kWh. Extends below experiment 1's floor so the optimum it could
#: not locate has somewhere to land.
BATTERIES = np.array([100.0, 250.0, 500.0, 1000.0, 1500.0])

#: One panel per price. See the module docstring.
PANEL_PRICES = np.array([3.0, 4.5, 6.0])

#: Process-chain capital cost as a fraction of the assumed EUR 2200/kW. A
#: near-term forward assumption rather than today's number: experiment 3 swept
#: 0.25-1.00 and found that even a free process chain does not repay at
#: EUR 3.00/kg, so a sizing grid run at 1.00 spends most of its cells in
#: territory where nothing is viable and the optimum is hard to read.
#:
#: This is free to assume here. Capital cost never enters the controller's
#: objective, so the multiplier moves the economics and not a single dispatch
#: decision -- verified directly in experiment 3, where the plans came back
#: bit-identical at 2200 and 1100 EUR/kW. It rescales the surface; it cannot
#: move the optimum's location for a reason the physics does not support.
PROCESS_CAPEX_MULTIPLIER = 0.75

PLAN_HOURS = 240


def run(quick: bool = False,
        process_multiplier: float = PROCESS_CAPEX_MULTIPLIER) -> pd.DataFrame:
    econ_base = Economics()
    crf = econ_base.capital_recovery_factor()
    fixed = float(econ_base.p.fixed_opex_fraction)
    horizon = float(econ_base.p.project_lifetime_years)
    discount = float(econ_base.p.discount_rate)
    seasons = {"summer": SEASONS["summer"]} if quick else SEASONS
    site = SITES[SITE_NAME]

    rows = []
    total = len(ARRAYS) * len(BATTERIES) * len(PANEL_PRICES) * len(seasons)
    done = 0
    started = time.time()
    for kwp in ARRAYS:
        for kwh in BATTERIES:
            plant = build_plant(float(kwp), float(kwh), float(kwh * C_RATE),
                                REFERENCE_SIZING["calciner_kw"])
            capex = econ_base.capex_full(plant)
            battery = plant["battery"]
            # Only the process chain is discounted; the array and the pack are
            # priced at today's numbers, which are already market-observed.
            capex_total = (capex.pv + capex.battery
                           + capex.process * process_multiplier)
            first_pack_per_day = capex_total * (crf + fixed) / 365.0
            opex_per_day = capex_total * fixed / 365.0

            for price in PANEL_PRICES:
                econ = Economics()
                econ.p.methane_price_per_kg = float(price)
                per_season, per_efc, per_stress = [], [], []
                for season, day in seasons.items():
                    weather = _weather(site, day)
                    operating, _hours, efc, stress = _plan_profit_per_day(
                        plant, econ, weather, site)
                    per_season.append(operating)
                    per_efc.append(efc)
                    per_stress.append(stress)
                    done += 1
                    rate = (time.time() - started) / done
                    print(f"  [{done:3d}/{total}] {kwp:6.0f} kWp "
                          f"{kwh:6.0f} kWh  EUR {price:.2f}/kg  {season:7s} "
                          f"-> operating EUR {operating:7.1f}/day  "
                          f"{efc:.2f} EFC/day   "
                          f"({rate:.1f} s/solve, "
                          f"{(total - done) * rate / 60.0:.0f} min left)",
                          flush=True)

                operating = float(np.nanmean(per_season))
                efc_day = float(np.nanmean(per_efc))
                stress = float(np.nanmean(per_stress))

                repl_pv = battery.uncharged_replacement_PV_EUR(
                    project_years=horizon, discount_rate=discount,
                    efc_per_year=efc_day * 365.0, mean_stress=stress,
                )
                life = battery.life_years(efc_day * 365.0, stress)

                # Cash flow, not the annuity: capex whole at t = 0, fixed opex
                # annual, the wear accrual added back so replacements can enter
                # as lumps when the pack actually dies. Same treatment as
                # experiments 1 and 2.
                wear_per_day = battery.cost_per_efc_EUR() * efc_day
                cash_per_day = operating + wear_per_day - opex_per_day

                rows.append({
                    "site": SITE_NAME,
                    "pv_kwp": float(kwp),
                    "battery_kwh": float(kwh),
                    "price_EUR_per_kg": float(price),
                    "operating_EUR_per_day": operating,
                    "capital_EUR_per_day": first_pack_per_day
                                           + repl_pv * crf / 365.0,
                    "first_pack_EUR_per_day": first_pack_per_day,
                    "replacement_EUR_per_day": repl_pv * crf / 365.0,
                    "net_EUR_per_day": operating - first_pack_per_day
                                       - repl_pv * crf / 365.0,
                    "cash_EUR_per_day": cash_per_day,
                    "payback_years": econ_base.discounted_payback_years(
                        capex_total, cash_per_day * 365.0,
                        replacement_EUR=battery.capex_EUR(),
                        replacement_interval_years=life,
                    ),
                    "capex_EUR": capex_total,
                    "capex_pv_EUR": capex.pv,
                    "capex_battery_EUR": capex.battery,
                    "capex_process_EUR": capex.process * process_multiplier,
                    "capex_process_full_EUR": capex.process,
                    "process_capex_multiplier": float(process_multiplier),
                    "efc_per_day": efc_day,
                    "dod_stress": stress,
                    "battery_life_years": life,
                    "seasons": len(per_season),
                    "season_min": float(np.nanmin(per_season)),
                    "season_max": float(np.nanmax(per_season)),
                })
    return pd.DataFrame(rows)


def _grid(frame: pd.DataFrame, price: float, column: str) -> np.ndarray:
    sub = frame[frame["price_EUR_per_kg"] == price]
    return sub.pivot(index="pv_kwp", columns="battery_kwh",
                     values=column).to_numpy()


def plot(frame: pd.DataFrame) -> None:
    from matplotlib.colors import TwoSlopeNorm

    rows = [f"{a:.0f}" for a in ARRAYS]
    cols = [f"{b:.0f}" for b in BATTERIES]
    panels = [f"EUR {p:.2f}/kg" for p in PANEL_PRICES]
    mult = float(frame["process_capex_multiplier"].iloc[0])
    basis = (f"process capex at {mult:.2f}x of EUR 2200/kW"
             if mult != 1.0 else "process capex at EUR 2200/kW")

    nets = {name: _grid(frame, float(p), "net_EUR_per_day")
            for name, p in zip(panels, PANEL_PRICES)}
    best = {name: np.unravel_index(int(np.nanargmax(g)), g.shape)
            for name, g in nets.items()}
    lo = min(np.nanmin(g) for g in nets.values())
    hi = max(np.nanmax(g) for g in nets.values())
    heatmap_panels(
        FIGURES / "joint_sizing_absolute.png", nets,
        row_labels=rows, col_labels=cols,
        title=f"Joint sizing at {SITE_NAME}: net profitability over PV array "
              f"and battery capacity\n(dispatch plans, seasonally averaged; "
              f"{basis})",
        xlabel="battery capacity, kWh", ylabel="PV array, kWp",
        cbar_label="net profitability, EUR/day "
                   "(operating less annualised capital)",
        cmap="RdYlGn", fmt=".0f",
        norm=TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi) if lo < 0 < hi else None,
        vmin=lo, vmax=hi, highlight=best,
    )

    horizon = float(Economics().p.project_lifetime_years)
    paybacks = {name: _grid(frame, float(p), "payback_years")
                for name, p in zip(panels, PANEL_PRICES)}
    fastest = {}
    for name, g in paybacks.items():
        finite = np.isfinite(g)
        if finite.any():
            masked = np.where(finite, g, np.inf)
            fastest[name] = np.unravel_index(int(np.argmin(masked)), g.shape)
    heatmap_panels(
        FIGURES / "joint_sizing_payback.png", paybacks,
        row_labels=rows, col_labels=cols,
        title=f"Joint sizing at {SITE_NAME}: time to break even\n"
              f"(discounted at 7 %, battery replaced as it falls due; {basis})",
        xlabel="battery capacity, kWh", ylabel="PV array, kWp",
        cbar_label=f"discounted payback, years "
                   f"(dark red = not repaid within {horizon:.0f} yr)",
        cmap="viridis_r", fmt=".1f", vmin=0.0, vmax=horizon,
        mask_invalid=True, highlight=fastest,
    )

    # Battery utilisation, which is what makes the interaction visible: the
    # economics alone cannot distinguish a pack that is too small to help from
    # one that is large enough but has no surplus to absorb.
    heatmap_panels(
        FIGURES / "joint_sizing_cycling.png",
        {name: _grid(frame, float(p), "efc_per_day")
         for name, p in zip(panels, PANEL_PRICES)},
        row_labels=rows, col_labels=cols,
        title=f"Joint sizing at {SITE_NAME}: how hard the plan works the pack",
        xlabel="battery capacity, kWh", ylabel="PV array, kWp",
        cbar_label="equivalent full cycles per day",
        cmap="magma", fmt=".2f",
    )


def check(frame: pd.DataFrame) -> None:
    """Reproduce two cells whose answers the earlier grids already fixed."""
    sources = [
        ("pv_sizing.csv", 500.0, {"pv_kwp": 1100.0}),
        ("relative_profitability.csv", 1500.0, {"battery_kwh": 1500.0}),
    ]
    print("\ncross-check against the one-dimensional grids")
    print(f"  {'source':28s} {'cell':22s} {'here':>10s} {'there':>10s} {'diff':>9s}")
    for filename, kwh, filt in sources:
        path = RESULTS / filename
        if not path.exists():
            print(f"  {filename:28s} not found -- skipped")
            continue
        other = pd.read_csv(path)
        other = other[other["site"] == SITE_NAME]
        for key, value in filt.items():
            other = other[other[key] == value]
        for price in (3.0, 6.0):
            mine = frame[(frame["pv_kwp"] == 1100.0)
                         & (frame["battery_kwh"] == kwh)
                         & (frame["price_EUR_per_kg"] == price)]
            theirs = other[other["price_EUR_per_kg"] == price]
            if mine.empty or theirs.empty:
                continue
            a = float(mine["operating_EUR_per_day"].iloc[0])
            b = float(theirs["operating_EUR_per_day"].iloc[0])
            cell = f"1100 kWp {kwh:.0f} kWh @{price:.0f}"
            print(f"  {filename:28s} {cell:22s} {a:10.4f} {b:10.4f} "
                  f"{a - b:+9.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="summer window only, for a fast smoke test")
    parser.add_argument("--check", action="store_true",
                        help="compare shared cells against experiments 1 and 2")
    parser.add_argument("--process-capex", type=float,
                        default=PROCESS_CAPEX_MULTIPLIER,
                        help="process-chain capex as a fraction of EUR 2200/kW")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    frame = run(quick=args.quick, process_multiplier=args.process_capex)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / ("joint_sizing_quick.csv" if args.quick
                     else "joint_sizing.csv")
    frame.to_csv(out, index=False)
    plot(frame)
    print(f"\nwrote {out}\nwrote {FIGURES}/joint_sizing_*.png")
    print(f"{SITE_NAME}, process capex at {args.process_capex:.2f}x "
          f"(EUR {2200.0 * args.process_capex:.0f}/kW)")

    for price in PANEL_PRICES:
        sub = frame[frame["price_EUR_per_kg"] == price]
        best = sub.loc[sub["net_EUR_per_day"].idxmax()]
        repaid = sub[np.isfinite(sub["payback_years"])]
        if len(repaid):
            f = repaid.loc[repaid["payback_years"].idxmin()]
            line = (f"fastest payback {f['payback_years']:.1f} yr at "
                    f"{f['pv_kwp']:.0f} kWp / {f['battery_kwh']:.0f} kWh")
        else:
            line = "never repaid at any cell"
        print(f"EUR {price:.2f}/kg   best net {best['net_EUR_per_day']:+.0f} "
              f"EUR/day at {best['pv_kwp']:.0f} kWp / "
              f"{best['battery_kwh']:.0f} kWh   |  {line}")

    if args.check and not args.quick:
        check(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
