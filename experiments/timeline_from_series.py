"""Rebuild the standard six-panel operating timeline from a stored trajectory.

`sfp.report.plots.timeline` needs a live `SimulationResult`; the sweep only kept
the log. Everything the standard figure shows is recoverable from the saved
columns except one thing:

**Bus trips are not in the stored series.** `sfp/report/plots.py` shades the SoC
panel where `bus_tripped > 0`, and that column was not among those kept. The
panel is otherwise identical; the shading is simply absent, and the trip counts
live in `results/battery_sizing.csv`.

Everything else is reconstructed rather than approximated: curtailment is
available minus delivered, temperatures come from the state vector, and the
buffer fills are computed with the same sorbent capacity model the plant uses,
so the CaCO3 loading accounts for Grasa-Abanades deactivation rather than being
a fraction of the nominal inventory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sfp.cli import REFERENCE_SIZING, build_plant  # noqa: E402
from sfp.report.plots import COLOURS, SUBSYSTEM_COLOURS  # noqa: E402

SERIES = HERE / "results" / "series"
FIGURES = HERE / "figures"
STACK = ("sabatier", "contactor", "electrolyser", "calciner", "gas")


def timeline(controller: str, battery_kwh: float, season: str = "summer") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = SERIES / f"{controller}_{battery_kwh:.0f}kWh_{season}.csv.gz"
    if not src.exists():
        raise SystemExit(f"{src} not found")
    log = pd.read_csv(src, index_col=0)
    log.index = pd.to_datetime(log.index)
    t = log.index

    plant = build_plant(REFERENCE_SIZING["pv_kwp"], battery_kwh,
                        battery_kwh * 0.5, REFERENCE_SIZING["calciner_kw"])
    battery, solids, gas = plant["battery"], plant["solids"], plant["gas"]

    # Two columns, column-major: the electrical story down the left (resource,
    # loads, state of charge) and the chemical one down the right (buffers,
    # temperatures, product). Reading across a row then pairs them -- what the
    # power was doing against what the chemistry was doing at the same hour.
    fig, grid = plt.subplots(3, 2, figsize=(15.5, 8.6), sharex=True)
    axes = [grid[0, 0], grid[1, 0], grid[2, 0],
            grid[0, 1], grid[1, 1], grid[2, 1]]

    # --- 1. solar resource ------------------------------------------------
    ax = axes[0]
    avail = log["pv_available_W"] / 1e3
    used = log["pv_delivered_W"] / 1e3
    ax.fill_between(t, 0, avail, color=COLOURS["solar"], alpha=0.35, label="available")
    ax.plot(t, used, color=COLOURS["solar"], lw=1.0, label="used")
    if (avail - used).max() > 1.0:
        ax.fill_between(t, used, avail, color=COLOURS["curtailed"], alpha=0.55,
                        label="curtailed")
    ax.set_ylabel("PV  [kW]")
    ax.set_title(f"{controller}  --  Seville, ES   "
                 f"({battery_kwh:.0f} kWh pack, {season}, 7 days)")
    ax.legend(ncol=3, loc="upper right")

    # --- 2. load by subsystem --------------------------------------------
    ax = axes[1]
    keys = [k for k in STACK if f"power.{k}" in log]
    ax.stackplot(t, *[log[f"power.{k}"] / 1e3 for k in keys], labels=keys,
                 colors=[SUBSYSTEM_COLOURS[k] for k in keys], alpha=0.85)
    # `power.battery` is signed on the plant's convention: positive is power
    # taken from the bus, so charging is positive and discharge negative.
    pb = log["power.battery"] / 1e3
    ax.plot(t, pb.clip(lower=0), color=COLOURS["battery"], lw=0.9, label="battery charge")
    ax.plot(t, pb.clip(upper=0), color=COLOURS["battery"], lw=0.9, ls="--",
            label="battery discharge")
    ax.axhline(0.0, color=COLOURS["grid"], lw=0.8)
    ax.set_ylabel("load  [kW]")
    ax.legend(ncol=4, loc="upper right", fontsize=7)

    # --- 3. battery -------------------------------------------------------
    ax = axes[2]
    ax.plot(t, log["state.battery.soc"], color=COLOURS["battery"], lw=1.2)
    ax.axhline(battery.p.soc_min, color=COLOURS["warn"], lw=0.8, ls=":", label="limits")
    ax.axhline(battery.p.soc_max, color=COLOURS["warn"], lw=0.8, ls=":")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("battery SoC")
    ax.legend(ncol=2, loc="upper right", fontsize=7.5)

    # --- 4. the buffers ---------------------------------------------------
    ax = axes[3]
    # Capacity falls as the sorbent deactivates, so loading is against the
    # *present* capacity, not the nominal inventory.
    cap = solids.p.n_total_mol * solids.max_conversion(
        log["state.solids.cycle_number"].to_numpy())
    ax.plot(t, log["state.solids.n_caco3"].to_numpy() / np.maximum(cap, 1.0),
            color=COLOURS["warn"], lw=1.3, label="CaCO$_3$ loading")
    ax.plot(t, log["state.gas.n_h2"] / gas.p.h2_capacity_mol,
            color=SUBSYSTEM_COLOURS["electrolyser"], lw=1.1, label="H$_2$ tank")
    ax.plot(t, log["state.gas.n_co2"] / gas.p.co2_capacity_mol,
            color=SUBSYSTEM_COLOURS["sabatier"], lw=1.1, label="CO$_2$ tank")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("buffer fill")
    ax.legend(ncol=3, loc="upper right", fontsize=7.5)

    # --- 5. temperatures --------------------------------------------------
    ax = axes[4]
    ax.plot(t, log["state.calciner.temperature_K"] - 273.15,
            color=SUBSYSTEM_COLOURS["calciner"], lw=1.2, label="kiln")
    thr = plant["calciner"].threshold_temperature_K() - 273.15
    ax.axhline(thr, color=SUBSYSTEM_COLOURS["calciner"], lw=0.8, ls=":",
               label=f"calcination threshold ({thr:.0f} $^\\circ$C)")
    ax.plot(t, log["state.sabatier.temperature_K"] - 273.15,
            color=SUBSYSTEM_COLOURS["sabatier"], lw=1.2, label="reactor")
    ax.set_ylabel("temperature  [$^\\circ$C]")
    ax.legend(ncol=3, loc="upper right", fontsize=7.5)

    # --- 6. cumulative product -------------------------------------------
    ax = axes[5]
    ch4 = log["state.sabatier.ch4_kg"]
    ax.plot(t, ch4, color=COLOURS["process"], lw=1.4)
    # `cos_zenith` was not stored; darkness is taken from the resource itself.
    night = log["pv_available_W"].to_numpy() < 1.0
    ax.fill_between(t, 0, ch4.max() * 1.05, where=night, color=COLOURS["text"],
                    alpha=0.06, step="mid", label="night")
    ax.set_ylabel("CH$_4$  [kg]")
    ax.legend(loc="upper left", fontsize=7.5)

    # Both columns carry their own date axis: with `sharex` only the bottom of
    # each column shows ticks, and a two-column layout needs both.
    import matplotlib.dates as mdates
    for a in (grid[2, 0], grid[2, 1]):
        a.set_xlabel("time (UTC)")
        a.xaxis.set_major_locator(mdates.DayLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    hours = (t[-1] - t[0]).total_seconds() / 3600.0
    night_share = float(np.diff(ch4.to_numpy(), prepend=ch4.iloc[0])[night].sum()
                        / max(ch4.iloc[-1] - ch4.iloc[0], 1e-9))
    fig.text(0.5, 0.005,
             f"{ch4.iloc[-1]:.0f} kg CH$_4$ over {hours / 24:.0f} days "
             f"({ch4.iloc[-1] / (hours / 24):.1f} kg/day)   "
             f"night {night_share:.0%}   "
             f"curtailed {(avail - used).sum() / max(avail.sum(), 1e-9):.0%}   "
             f"SoC {log['state.battery.soc'].min():.2f}-"
             f"{log['state.battery.soc'].max():.2f}   "
             f"(trips not stored in the series -- see battery_sizing.csv)",
             ha="center", fontsize=9, color=COLOURS["text"])

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"timeline_{controller}_{battery_kwh:.0f}kWh_{season}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controller", default="dispatch-nmpc")
    p.add_argument("--battery", type=float, default=1500.0)
    p.add_argument("--season", default="summer")
    p.add_argument("--all", action="store_true", help="every stored trajectory")
    a = p.parse_args()
    jobs = ([(c, b) for c in ("dispatch", "dispatch-nmpc")
             for b in (100.0, 250.0, 500.0, 1000.0, 1500.0)]
            if a.all else [(a.controller, a.battery)])
    for controller, kwh in jobs:
        print("wrote", timeline(controller, kwh, a.season))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
