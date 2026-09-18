"""Figures for experiment A. Reads `results/battery_sizing.csv` and the summer
trajectories in `results/series/`; writes to `figures/`.

Six figures in one argument: the outer layer's economics are wrong, here is how
wrong, here is why, here is what it looks like, here is the corrected answer,
and here is the commitment behaviour underneath it all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sfp.economics import Economics  # noqa: E402

RESULTS = HERE / "results"
SERIES = RESULTS / "series"
FIGURES = HERE / "figures"

MACHINES = ("contactor", "calciner", "electrolyser", "sabatier")
BLOCKS = ("bus_balance", "state_box", "band_ceiling", "rate_limit")
BLOCK_LABEL = {"bus_balance": "bus balance", "state_box": "state box",
               "band_ceiling": "plan band", "rate_limit": "slew limit"}
MARKET_PRICES = (3.0, 5.5, 8.0)


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.grid": False, "font.size": 9,
                         "axes.titleweight": "bold", "figure.dpi": 150})
    FIGURES.mkdir(parents=True, exist_ok=True)
    return plt


def load() -> pd.DataFrame:
    f = RESULTS / "battery_sizing.csv"
    if not f.exists():
        raise SystemExit(f"{f} not found -- run battery_sizing.py first")
    return pd.read_csv(f)


def seasonal_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Average the four seasonal windows, keeping counts as sums per day."""
    num = [c for c in df.select_dtypes(include=[np.number]).columns
           if c != "battery_kwh"]
    return df.groupby(["controller", "battery_kwh"])[num].mean().reset_index()


def net_per_day(row, price: float, econ: Economics) -> float:
    """Net EUR/day at an arbitrary *market* price.

    The controller's own price is fixed, so production does not depend on this
    -- which is exactly why the price axis costs nothing. Operating cost is
    recovered from the margin recorded at the controller's price and re-priced.
    """
    days = float(row["days"])
    ctrl_price = float(econ.p.methane_price_per_kg)
    revenue_ctrl = row["ch4_kg_per_day"] * days * ctrl_price
    cost = revenue_ctrl - row["operating_margin_eur"]
    margin = (row["ch4_kg_per_day"] * days * price - cost) / days
    crf = econ.capital_recovery_factor()
    fixed = float(econ.p.fixed_opex_fraction)
    return margin - row["capex_eur"] * (crf + fixed) / 365.0


# --- 1. the finding -------------------------------------------------------
def fig_plan_vs_realised(df, plt):
    d = seasonal_mean(df[df.controller == "dispatch"]).sort_values("battery_kwh")
    if "planned_eur" not in d:
        return
    days = float(df["days"].iloc[0])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    x = d["battery_kwh"]
    ax[0].plot(x, d["planned_eur"] / days, "o-", color="#1f77b4",
               label="planned by the dispatch LP")
    ax[0].plot(x, d["realised_eur"] / days, "s-", color="#d62728",
               label="realised by the plant")
    ax[0].fill_between(x, d["planned_eur"] / days, d["realised_eur"] / days,
                       color="#d62728", alpha=0.15)
    ax[0].set_xlabel("battery capacity, kWh")
    ax[0].set_ylabel("operating profit, EUR/day")
    ax[0].set_title("The plan over-credits the plant")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(x, 100 * d["plan_bias"], "o-", color="#d62728")
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].set_xlabel("battery capacity, kWh")
    ax[1].set_ylabel("realised vs planned, %")
    ax[1].set_title("...and worse the smaller the store")
    for a in ax:
        a.set_xscale("log")
        a.set_xticks(x); a.set_xticklabels([f"{v:.0f}" for v in x])
    fig.suptitle("Experiment A.1 — plan fidelity is not common-mode along the "
                 "sizing axis", fontweight="bold")
    fig.savefig(FIGURES / "A1_plan_vs_realised.png")
    plt.close(fig)


# --- 2. what the safety layer buys ---------------------------------------
def fig_safety(df, plt):
    g = seasonal_mean(df).sort_values("battery_kwh")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    w = 0.38
    sizes = sorted(g["battery_kwh"].unique())
    idx = np.arange(len(sizes))
    for off, (name, colour) in zip((-w / 2, w / 2),
                                   (("dispatch", "#d62728"),
                                    ("dispatch-nmpc", "#2ca02c"))):
        s = g[g.controller == name].set_index("battery_kwh").reindex(sizes)
        ax[0].bar(idx + off, s["bus_trips"], w, label=name, color=colour)
        ax[1].bar(idx + off, s["shed_kwh"], w, label=name, color=colour)
        ax[2].plot(idx, s["ch4_kg_per_day"], "o-", color=colour, label=name)
    for a, t, yl in ((ax[0], "Undervoltage trips", "trips per 7 d"),
                     (ax[1], "Load shed by the bus", "kWh per 7 d"),
                     (ax[2], "Methane", "kg/day")):
        a.set_xticks(idx); a.set_xticklabels([f"{v:.0f}" for v in sizes])
        a.set_xlabel("battery capacity, kWh"); a.set_ylabel(yl); a.set_title(t)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle("Experiment A.2 — what the inner safety layer buys",
                 fontweight="bold")
    fig.savefig(FIGURES / "A2_safety.png")
    plt.close(fig)


# --- 3. why the filter edits ---------------------------------------------
def fig_attribution(df, plt):
    g = seasonal_mean(df[df.controller == "dispatch-nmpc"]).sort_values("battery_kwh")
    cols = [f"bind_{b}" for b in BLOCKS if f"bind_{b}" in g]
    if not cols:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    sizes = g["battery_kwh"].to_numpy()
    idx = np.arange(len(sizes))
    mass = g[cols].to_numpy()
    share = mass / np.maximum(mass.sum(axis=1, keepdims=True), 1e-12)
    bottom = np.zeros(len(sizes))
    for j, c in enumerate(cols):
        ax[0].bar(idx, share[:, j], 0.7, bottom=bottom,
                  label=BLOCK_LABEL.get(c[5:], c[5:]))
        bottom += share[:, j]
    ax[0].set_ylabel("share of dual mass"); ax[0].set_title("Which constraint acts")
    ax[0].legend(frameon=False, fontsize=8, ncol=2)

    ax[1].plot(idx, g["edit_mean"], "o-", color="#9467bd")
    ax[1].set_ylabel(r"$\|u_0-\bar u_0\|^2$, scaled")
    ax[1].set_title("How hard the filter edits")
    ax[1].set_yscale("log")

    if "cf_bus_deficit_W" in g:
        ax[2].plot(idx, g["cf_bus_deficit_W"] / 1000.0, "o-", color="#ff7f0e",
                   label="plan's bus deficit")
        ax[2].set_ylabel("kW")
        ax[2].set_title("What the plan would have done")
        ax[2].legend(frameon=False, fontsize=8)
    for a in ax:
        a.set_xticks(idx); a.set_xticklabels([f"{v:.0f}" for v in sizes])
        a.set_xlabel("battery capacity, kWh")
    fig.suptitle("Experiment A.3 — why the filter moves the plan "
                 "(duals, and the plan's own counterfactual)", fontweight="bold")
    fig.savefig(FIGURES / "A3_attribution.png")
    plt.close(fig)


# --- 4. one day in detail -------------------------------------------------
def fig_detail(plt, kwh: float = 250.0, day: int = 2):
    f = SERIES / f"dispatch-nmpc_{kwh:.0f}kWh_summer.csv.gz"
    if not f.exists():
        return
    log = pd.read_csv(f, index_col=0)
    n = len(log)
    per_day = n // 7
    sl = slice(day * per_day, (day + 1) * per_day)
    log = log.iloc[sl]
    h = np.arange(len(log)) / 60.0

    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True,
                           constrained_layout=True)
    if "pv_available_W" in log:
        ax[0].fill_between(h, log["pv_available_W"] / 1000.0, color="#ffd27f",
                           label="available")
        ax[0].plot(h, log["pv_delivered_W"] / 1000.0, color="#d95f02", lw=1,
                   label="delivered")
    ax[0].set_ylabel("PV, kW"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title(f"{kwh:.0f} kWh pack, summer, day {day + 1}")

    if "state.battery.soc" in log:
        ax[1].plot(h, log["state.battery.soc"], color="#1f77b4")
        ax[1].set_ylabel("SoC"); ax[1].set_ylim(0, 1)

    for key, colour in zip(MACHINES, ("#4c72b0", "#dd8452", "#55a868", "#c44e52")):
        c = f"setpoint_{key}"
        if c in log:
            ax[2].plot(h, log[c], lw=1, color=colour, label=key)
    ax[2].set_ylabel("setpoint applied"); ax[2].legend(frameon=False, fontsize=7, ncol=4)

    if "nmpc_edit" in log:
        e = log["nmpc_edit"].ffill()
        ax[3].plot(h, e, color="#9467bd", lw=1)
        ax[3].set_yscale("log"); ax[3].set_ylabel(r"edit $\|u_0-\bar u_0\|^2$")
    ax[3].set_xlabel("hour of day")
    fig.suptitle("Experiment A.4 — the filter at work over one day",
                 fontweight="bold")
    fig.savefig(FIGURES / "A4_detail.png")
    plt.close(fig)


# --- 5. corrected economics ----------------------------------------------
def fig_economics(df, plt):
    econ = Economics()
    g = seasonal_mean(df).sort_values("battery_kwh")
    # seasonal_mean drops non-numeric columns; days survives
    fig, ax = plt.subplots(1, len(MARKET_PRICES), figsize=(12, 3.8),
                           sharey=True, constrained_layout=True)
    for a, price in zip(np.atleast_1d(ax), MARKET_PRICES):
        for name, colour in (("dispatch", "#d62728"),
                             ("dispatch-nmpc", "#2ca02c")):
            s = g[g.controller == name].sort_values("battery_kwh")
            net = [net_per_day(r, price, econ) for _, r in s.iterrows()]
            a.plot(s["battery_kwh"], net, "o-", color=colour, label=name)
        a.axhline(0, color="0.6", lw=0.8)
        a.set_xscale("log")
        a.set_xticks(g["battery_kwh"].unique())
        a.set_xticklabels([f"{v:.0f}" for v in sorted(g['battery_kwh'].unique())])
        a.set_xlabel("battery capacity, kWh")
        a.set_title(f"market price EUR {price:.2f}/kg")
    np.atleast_1d(ax)[0].set_ylabel("net EUR/day")
    np.atleast_1d(ax)[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Experiment A.5 — corrected economics "
                 "(controller price fixed; market price is arithmetic)",
                 fontweight="bold")
    fig.savefig(FIGURES / "A5_economics.png")
    plt.close(fig)


# --- 6. does it actually shut down ---------------------------------------
def fig_commitment(plt, sizes=(100.0, 1500.0), days: int = 3):
    files = [(k, SERIES / f"dispatch-nmpc_{k:.0f}kWh_summer.csv.gz") for k in sizes]
    files = [(k, f) for k, f in files if f.exists()]
    if not files:
        return
    fig, ax = plt.subplots(len(files), 1, figsize=(10, 3.0 * len(files)),
                           sharex=True, constrained_layout=True)
    for a, (kwh, f) in zip(np.atleast_1d(ax), files):
        log = pd.read_csv(f, index_col=0).iloc[:days * 1440]
        h = np.arange(len(log)) / 60.0
        if "pv_available_W" in log:
            pv = log["pv_available_W"].to_numpy()
            a.fill_between(h, 0, len(MACHINES) * (pv / max(pv.max(), 1.0)),
                           color="#ffe9b0", zorder=0, label="PV available")
        for i, key in enumerate(MACHINES):
            c = f"enable_{key}"
            if c not in log:
                continue
            on = log[c].to_numpy() >= 0.5
            a.fill_between(h, i + 0.1, i + 0.9, where=on, step="mid",
                           color="#2c7fb8", zorder=2)
        a.set_yticks(np.arange(len(MACHINES)) + 0.5)
        a.set_yticklabels(MACHINES)
        a.set_ylim(0, len(MACHINES))
        a.set_title(f"{kwh:.0f} kWh pack")
        for d in range(1, days):
            a.axvline(24 * d, color="0.7", lw=0.8, zorder=3)
    np.atleast_1d(ax)[-1].set_xlabel("hour")
    fig.suptitle("Experiment A.6 — commitment: what actually runs, and when",
                 fontweight="bold")
    fig.savefig(FIGURES / "A6_commitment.png")
    plt.close(fig)


SEASON_STYLE = {"winter": ("#3b6ea5", "o-"), "spring": ("#4daf4a", "s-"),
                "summer": ("#e6a11e", "^-"), "autumn": ("#a4562c", "v-")}


# --- 7. trips by season ---------------------------------------------------
def fig_seasonal_trips(df, plt):
    """Why the seasonal mean of the trip count describes no season.

    Averaging four windows is defensible for production, which varies smoothly.
    It is misleading for trips, which do not: three seasons collapse to nothing
    once the store is adequate, and winter does not collapse at all. A single
    averaged number reports neither behaviour.
    """
    from matplotlib.ticker import FixedLocator, FixedFormatter
    sizes = sorted(df["battery_kwh"].unique())
    idx = np.arange(len(sizes))
    names = ["dispatch", "dispatch-nmpc"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True,
                           constrained_layout=True)
    for a, name in zip(ax, names):
        sub = df[df.controller == name]
        for season, (colour, marker) in SEASON_STYLE.items():
            s = sub[sub.season == season].set_index("battery_kwh").reindex(sizes)
            a.plot(idx, s["bus_trips"], marker, color=colour, label=season,
                   markersize=5, lw=1.6)
        a.set_xticks(idx)
        a.set_xticklabels([f"{v:.0f}" for v in sizes])
        a.set_xlabel("battery capacity, kWh")
        a.set_title(name)
        # symlog: the interesting part of this plot is a collapse to zero, and
        # a log axis cannot show zero at all.
        a.set_yscale("symlog", linthresh=10)
        a.yaxis.set_major_locator(FixedLocator([0, 10, 100, 1000, 5000]))
        a.yaxis.set_major_formatter(FixedFormatter(["0", "10", "100", "1k", "5k"]))
        a.grid(axis="y", alpha=0.25, lw=0.6)
    ax[0].set_ylabel("undervoltage trips per 7 days")
    ax[0].legend(frameon=False, fontsize=8, title="season", title_fontsize=8)
    fig.suptitle("Experiment A.7 — trips by season: the sizing result is a "
                 "winter result", fontweight="bold")
    fig.savefig(FIGURES / "A7_seasonal_trips.png")
    plt.close(fig)


def main() -> int:
    plt = _setup()
    df = load()
    print(f"{len(df)} rows, controllers {sorted(df.controller.unique())}, "
          f"sizes {sorted(df.battery_kwh.unique())}")
    for fn, args in ((fig_plan_vs_realised, (df, plt)), (fig_safety, (df, plt)),
                     (fig_attribution, (df, plt)), (fig_detail, (plt,)),
                     (fig_economics, (df, plt)), (fig_commitment, (plt,)),
                     (fig_seasonal_trips, (df, plt))):
        try:
            fn(*args)
            print(f"  ok  {fn.__name__}")
        except Exception as exc:
            print(f"  --  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"wrote {FIGURES}/A*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
