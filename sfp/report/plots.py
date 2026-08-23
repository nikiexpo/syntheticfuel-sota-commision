"""Figures for the report and the submission video.

One multi-panel timeline per run, plus a strategy-comparison chart. The brief
asks submissions to "show the system responding dynamically rather than only
presenting final results", so these are built to be read as a sequence -- the
same axes, the same colours, the same y-limits across strategies, so two runs can
be put side by side and the difference is the strategy rather than the scaling.

Colour is used semantically and consistently:
    solar      amber      what is available
    process    teal       what is being converted
    battery    violet     what is being stored
    curtailed  grey       what is being thrown away
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sfp.report.metrics import RunMetrics  # noqa: E402
from sfp.sim.simulator import SimulationResult  # noqa: E402

COLOURS = {
    "solar": "#E8A33D",
    "process": "#2A9D8F",
    "battery": "#7B6CD9",
    "curtailed": "#9AA0A6",
    "grid": "#D9DCE0",
    "text": "#22252A",
    "warn": "#C1453B",
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.edgecolor": COLOURS["grid"],
        "axes.labelcolor": COLOURS["text"],
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": COLOURS["grid"],
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.color": COLOURS["text"],
        "ytick.color": COLOURS["text"],
    }
)


SUBSYSTEM_COLOURS = {
    "contactor": "#4FA3B8",
    "calciner": "#C1453B",
    "electrolyser": "#5B8FF9",
    "sabatier": "#2A9D8F",
    "gas": "#9AA0A6",
}


def timeline(
    result: SimulationResult,
    metrics: RunMetrics | None = None,
    path: Path | None = None,
) -> Path:
    """Six-panel operating timeline for one run.

    Two panels exist specifically to show the thing the architecture is about:
    the buffer levels, and the kiln/reactor temperatures. Reading them together
    is how you see the chemical battery charging by day and discharging at night
    -- solids loading rising while the sun is up, the H2 tank filling, then the
    reactor drawing both down through the dark hours with the kiln coasting.
    """
    log = result.log
    t = log.index

    fig, axes = plt.subplots(6, 1, figsize=(11, 13.5), sharex=True)

    # --- 1. solar resource ------------------------------------------------
    ax = axes[0]
    ax.fill_between(t, 0, log["pv_available_W"] / 1e3, color=COLOURS["solar"], alpha=0.35,
                    label="available")
    ax.plot(t, log["pv_used_W"] / 1e3, color=COLOURS["solar"], lw=1.0, label="used")
    if log["pv_curtailed_W"].max() > 1.0:
        ax.fill_between(t, log["pv_used_W"] / 1e3, log["pv_available_W"] / 1e3,
                        color=COLOURS["curtailed"], alpha=0.55, label="curtailed")
    ax.set_ylabel("PV  [kW]")
    ax.set_title(f"{result.controller_name}  --  {result.site.name}  ({result.weather_provenance} weather)")
    ax.legend(ncol=3, loc="upper right")

    # --- 2. load by subsystem (stacked) -----------------------------------
    ax = axes[1]
    stack_keys = [k for k in ("sabatier", "contactor", "electrolyser", "calciner", "gas")
                  if f"power.{k}" in log]
    if stack_keys:
        ax.stackplot(
            t,
            *[log[f"power.{k}"] / 1e3 for k in stack_keys],
            labels=stack_keys,
            colors=[SUBSYSTEM_COLOURS[k] for k in stack_keys],
            alpha=0.85,
        )
    ax.plot(t, -log["battery_discharge_W"] / 1e3, color=COLOURS["battery"], lw=0.9, ls="--",
            label="battery discharge")
    ax.plot(t, log["battery_charge_W"] / 1e3, color=COLOURS["battery"], lw=0.9,
            label="battery charge")
    ax.axhline(0.0, color=COLOURS["grid"], lw=0.8)
    ax.set_ylabel("load  [kW]")
    ax.legend(ncol=4, loc="upper right", fontsize=7)

    # --- 3. battery -------------------------------------------------------
    ax = axes[2]
    ax.plot(t, log["battery_soc"], color=COLOURS["battery"], lw=1.2)
    battery = result.plant["battery"]
    ax.axhline(battery.p.soc_min, color=COLOURS["warn"], lw=0.8, ls=":", label="limits")
    ax.axhline(battery.p.soc_max, color=COLOURS["warn"], lw=0.8, ls=":")
    if "bus_tripped" in log and log["bus_tripped"].sum() > 0:
        ax.fill_between(t, 0, 1, where=log["bus_tripped"] > 0, color=COLOURS["warn"],
                        alpha=0.12, step="mid", label="undervoltage trip")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("battery SoC")
    ax.legend(ncol=2, loc="upper right", fontsize=7.5)

    # --- 4. the buffers ---------------------------------------------------
    ax = axes[3]
    if "solids_loading" in log:
        ax.plot(t, log["solids_loading"], color=COLOURS["warn"], lw=1.3,
                label="CaCO$_3$ loading")
    if "gas_h2_fill" in log:
        ax.plot(t, log["gas_h2_fill"], color=SUBSYSTEM_COLOURS["electrolyser"], lw=1.1,
                label="H$_2$ tank")
    if "gas_co2_fill" in log:
        ax.plot(t, log["gas_co2_fill"], color=SUBSYSTEM_COLOURS["sabatier"], lw=1.1,
                label="CO$_2$ tank")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("buffer fill")
    ax.legend(ncol=3, loc="upper right", fontsize=7.5)

    # --- 5. temperatures --------------------------------------------------
    ax = axes[4]
    if "calciner_temperature_C" in log:
        ax.plot(t, log["calciner_temperature_C"], color=SUBSYSTEM_COLOURS["calciner"], lw=1.2,
                label="kiln")
        threshold = result.plant["calciner"].threshold_temperature_K() - 273.15
        ax.axhline(threshold, color=SUBSYSTEM_COLOURS["calciner"], lw=0.8, ls=":",
                   label=f"calcination threshold ({threshold:.0f} $^\\circ$C)")
    if "sabatier_temperature_C" in log:
        ax.plot(t, log["sabatier_temperature_C"], color=SUBSYSTEM_COLOURS["sabatier"], lw=1.2,
                label="reactor")
    ax.set_ylabel("temperature  [$^\\circ$C]")
    ax.legend(ncol=3, loc="upper right", fontsize=7.5)

    # --- 6. cumulative product -------------------------------------------
    ax = axes[5]
    ax.plot(t, log["ch4_total_kg"], color=COLOURS["process"], lw=1.4)
    if "cos_zenith" in log:
        ax.fill_between(t, 0, log["ch4_total_kg"].max() * 1.05,
                        where=log["cos_zenith"] <= 0.0, color=COLOURS["text"], alpha=0.06,
                        step="mid", label="night")
        ax.legend(loc="upper left", fontsize=7.5)
    ax.set_ylabel("CH$_4$  [kg]")
    ax.set_xlabel("time (UTC)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    if metrics is not None:
        summary = (
            f"{metrics.ch4_kg:.0f} kg CH$_4$   "
            f"night {metrics.night_production_fraction:.0%}   "
            f"utilisation {metrics.utilisation:.0%}   "
            f"curtailed {metrics.curtailment_fraction:.0%}   "
            f"{metrics.sorbent_cycles:.2f} sorbent cycles   "
            f"LCOM {metrics.lcom_eur_per_kg:.2f} EUR/kg   "
            f"limiting: {metrics.limiting_subsystem}"
        )
        fig.text(0.5, 0.005, summary, ha="center", fontsize=9, color=COLOURS["text"])

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = path or Path("out") / f"timeline_{result.controller_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def comparison(metrics: list[RunMetrics], path: Path | None = None) -> Path:
    """Bar chart comparing strategies on the metrics that matter."""
    names = [m.controller for m in metrics]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))

    panels = [
        ("CH$_4$ produced [kg]", [m.ch4_kg for m in metrics], COLOURS["process"], False),
        ("LCOM [EUR/kg]", [m.lcom_eur_per_kg for m in metrics], COLOURS["solar"], True),
        ("night-time share [%]", [100 * m.night_production_fraction for m in metrics], COLOURS["battery"], False),
        ("sorbent cycles", [m.sorbent_cycles for m in metrics], COLOURS["warn"], True),
    ]

    for ax, (title, values, colour, lower_better) in zip(axes, panels):
        bars = ax.bar(names, values, color=colour, alpha=0.85)
        ax.set_title(title + ("  (lower better)" if lower_better else "  (higher better)"),
                     fontsize=8.5)
        ax.tick_params(axis="x", rotation=20)
        finite = [v for v in values if np.isfinite(v)]
        if finite:
            ax.set_ylim(0, max(finite) * 1.25)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.1f}",
                        ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = path or Path("out") / "comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def limiting_subsystem(metrics: list[RunMetrics], path: Path | None = None) -> Path:
    """Stacked bar of what was binding, per strategy.

    The most actionable plot in the report: it says which component to buy more
    of, and it shows that the answer changes with the control strategy.
    """
    frame = pd.DataFrame({m.controller: m.limiting_distribution for m in metrics}).fillna(0.0).T
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bottom = np.zeros(len(frame))
    palette = [COLOURS["solar"], COLOURS["process"], COLOURS["battery"],
               COLOURS["curtailed"], COLOURS["warn"], "#5B8FF9", "#B8C0C8"]
    for i, column in enumerate(frame.columns):
        values = frame[column].to_numpy() * 100.0
        ax.barh(frame.index, values, left=bottom, label=column,
                color=palette[i % len(palette)], alpha=0.9)
        bottom += values
    ax.set_xlabel("share of run time [%]")
    ax.set_xlim(0, 100)
    ax.set_title("Limiting constraint over the run")
    ax.legend(ncol=3, loc="lower right", fontsize=7.5)
    fig.tight_layout()
    path = path or Path("out") / "limiting.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
