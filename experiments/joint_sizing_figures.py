"""Analysis and figures for experiment B (joint PV x battery sizing).

Reads `results/joint_sizing.csv`, combines the two seasonal windows with the
fitted weights, recovers the battery duty stress from the stored summer
trajectories, and writes `results/joint_payback.csv` plus five figures.

Battery stress
--------------
`life_years` needs a duty-weighted mean depth-of-discharge stress, which sets
when replacement lumps fall due. It is **measured from the summer trajectory and
applied to both seasons**, because winter trajectories were not stored.

That approximation is defensible and its direction is known. Stress is a
property of how deeply the controller cycles the pack, not of how much energy it
moves, and the measured values (0.78-1.06) are far from the 0.5 placeholder used
in the first pass. Winter cycles shallower, so using the summer figure slightly
*overstates* stress, shortening life and lengthening payback -- the conservative
direction.

It matters unevenly: over a 15x range of stress, payback at a 500 kWh pack moves
0.5 years, and at 2500 kWh it moves up to 7.75. Since one of the open questions
is whether Seville's optimum sits at 1500 or 2500 kWh, a placeholder there would
have been deciding the answer.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sfp.cli import REFERENCE_SIZING, build_plant  # noqa: E402
from sfp.economics import Economics  # noqa: E402

RESULTS = HERE / "results"
SERIES = RESULTS / "series"
FIGURES = HERE / "figures"

SITES = ("Seville", "London")
PV = (700.0, 1100.0, 1500.0)
BATT = (500.0, 1500.0, 2500.0)
PANEL_MULT = (1.00, 0.50, 0.25, 0.10)
SITE_COLOUR = {"Seville": "#E8A33D", "London": "#4C72B0"}
HORIZON = 25.0


def measured_stress() -> dict:
    """Duty-weighted DoD stress per configuration, from the summer runs."""
    out = {}
    for f in sorted(SERIES.glob("joint_*summer.csv.gz")):
        m = re.search(r"joint_(\w+?)_(\d+)kWp_(\d+)kWh", f.name)
        if not m:
            continue
        site, pv, kwh = m.group(1), float(m.group(2)), float(m.group(3))
        d = pd.read_csv(f, index_col=0)
        bat = build_plant(pv, kwh, kwh * 0.5,
                          REFERENCE_SIZING["calciner_kw"])["battery"]
        soc = d["state.battery.soc"].to_numpy()
        efc = d["state.battery.efc"].to_numpy()
        throughput = np.diff(efc, prepend=efc[0])
        out[(site, pv, kwh)] = float(bat.mean_stress_over(soc[:-1], throughput[1:]))
    return out


def weighted() -> pd.DataFrame:
    """Combine the seasonal windows with the fitted weights."""
    d = pd.read_csv(RESULTS / "joint_sizing.csv")
    num = [c for c in d.select_dtypes(include=[np.number]).columns
           if c not in ("pv_kwp", "battery_kwh", "weight")]

    def combine(g):
        w = g["weight"].to_numpy()
        w = w / w.sum()
        return pd.Series({c: float(np.dot(w, g[c].to_numpy())) for c in num})

    g = (d.groupby(["site", "pv_kwp", "battery_kwh"])
          .apply(combine, include_groups=False).reset_index())
    # keep the seasonal detail too, for B4
    g.attrs["seasonal"] = d
    return g


def payback_table(g: pd.DataFrame, stress: dict,
                  multipliers=np.linspace(0.0, 1.0, 41)) -> pd.DataFrame:
    econ = Economics()
    fixed = float(econ.p.fixed_opex_fraction)
    rows = []
    for _, r in g.iterrows():
        plant = build_plant(r.pv_kwp, r.battery_kwh, r.battery_kwh * 0.5,
                            REFERENCE_SIZING["calciner_kw"])
        bat = plant["battery"]
        ms = stress.get((r.site, r.pv_kwp, r.battery_kwh), 1.0)
        life = bat.life_years(r.efc_per_day * 365.0, ms)
        # The wear accrual is added back and replacements enter as lumps: the
        # operating objective already pre-pays a replacement that has not
        # happened, and a cash-flow model buys the pack when it dies.
        wear_day = bat.cost_per_efc_EUR() * r.efc_per_day
        op_day = r.operating_margin_eur / r.days
        for m in multipliers:
            capex = (r.capex_pv_eur + r.capex_battery_eur
                     + r.capex_process_eur * m)
            cash_day = op_day + wear_day - capex * fixed / 365.0
            rows.append(dict(
                site=r.site, pv_kwp=r.pv_kwp, battery_kwh=r.battery_kwh,
                mult=round(float(m), 4), capex=capex, stress=ms,
                battery_life_yr=life, cash_day=cash_day,
                ch4_kg_per_day=r.ch4_kg_per_day,
                payback=econ.discounted_payback_years(
                    capex, cash_day * 365.0, replacement_EUR=bat.capex_EUR(),
                    replacement_interval_years=life)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def _grid(sub, column):
    return (sub.pivot(index="pv_kwp", columns="battery_kwh", values=column)
               .reindex(index=PV, columns=BATT).to_numpy())


def _heat(ax, grid, fmt, cmap, vmin, vmax, mask_invalid=False,
          invalid="never", invalid_colour="#3b0a0a", highlight=None):
    import matplotlib.pyplot as plt
    shown = np.ma.masked_invalid(grid) if mask_invalid else grid
    cm = plt.get_cmap(cmap).copy()
    if mask_invalid:
        cm.set_bad(invalid_colour)
    im = ax.imshow(shown, origin="lower", aspect="auto", cmap=cm,
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(BATT)), [f"{b:.0f}" for b in BATT])
    ax.set_yticks(range(len(PV)), [f"{p:.0f}" for p in PV])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if not np.isfinite(v):
                ax.text(j, i, invalid, ha="center", va="center", fontsize=8,
                        fontweight="bold", color="white")
                continue
            rgba = im.cmap(im.norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                    fontweight="bold", color="black" if lum > 0.55 else "white")
    if highlight:
        from matplotlib.patches import Rectangle
        for (i, j), colour in highlight:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                   edgecolor=colour, lw=2.5, zorder=5))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    return im


def fig_b1(p, plt):
    """The payback grid, landscape: capex across, site down.

    Four columns by two rows rather than the reverse, so the figure is
    landscape and sits in a report column without rotation. Reading left to
    right is what a cost decline buys; reading down is what the site costs.
    """
    fig, axes = plt.subplots(len(SITES), len(PANEL_MULT),
                             figsize=(13.2, 6.2), constrained_layout=True)
    im = None
    for i, site in enumerate(SITES):
        for j, m in enumerate(PANEL_MULT):
            sub = p[(p.site == site) & (np.isclose(p["mult"], m))]
            ax = axes[i, j]
            im = _heat(ax, _grid(sub, "payback"), ".1f", "viridis_r", 0.0,
                       HORIZON, mask_invalid=True)
            if i == 0:
                ax.set_title(f"process capex  x{m:.2f}", fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{site}\n\nPV array, kWp", fontweight="bold")
            if i == len(SITES) - 1:
                ax.set_xlabel("battery, kWh")
    fig.colorbar(im, ax=axes, label=f"discounted payback, years "
                                    f"(dark red = not repaid within {HORIZON:.0f} yr)",
                 shrink=0.75)
    fig.suptitle("B1 — Time to profitability across sizing, site and "
                 "process-chain cost", fontweight="bold")
    fig.savefig(FIGURES / "B1_payback_grid.png", dpi=150)
    plt.close(fig)


def fig_b2(p, plt):
    """The capex frontier -- the lead figure."""
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.axhspan(HORIZON, HORIZON * 1.6, color="#3b0a0a", alpha=0.10, zorder=0)
    ax.text(0.02, HORIZON * 1.06, "never repays within the 25-year project",
            fontsize=8, color="#7a1f1f")
    label_dy = {"Seville": -2.6, "London": +2.6}
    for site in SITES:
        s = p[p.site == site].groupby("mult")["payback"].min().reset_index()
        finite = s[np.isfinite(s.payback)]
        if not len(finite):
            continue
        ax.plot(finite["mult"], finite["payback"], "-", lw=2.4,
                color=SITE_COLOUR[site], label=site, zorder=3)
        cross_m = finite["mult"].max()
        cross_y = finite.loc[finite["mult"].idxmax(), "payback"]
        ax.plot([cross_m], [cross_y], "o", ms=7, color=SITE_COLOUR[site],
                zorder=4, clip_on=False)
        ax.annotate(f"repays below  x{cross_m:.2f}", xy=(cross_m, cross_y),
                    xytext=(cross_m - 0.03, cross_y + 1.6), fontsize=8.5,
                    ha="right", fontweight="bold", color=SITE_COLOUR[site])
        # The y-intercept is the point of the figure: what remains when the
        # process chain is free. Offset away from each line so neither label
        # sits on top of the curve it describes.
        floor = finite.loc[finite["mult"].idxmin(), "payback"]
        ax.plot([0.0], [floor], "s", ms=6, color=SITE_COLOUR[site], zorder=4,
                clip_on=False)
        ax.annotate(f"{floor:.1f} yr with the process chain FREE",
                    xy=(0.0, floor), xytext=(0.035, floor + label_dy[site]),
                    fontsize=8.5, color=SITE_COLOUR[site], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", lw=0.9,
                                    color=SITE_COLOUR[site], alpha=0.7))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, HORIZON * 1.35)
    ax.set_xlabel("process-chain capex, fraction of EUR 2200/kW")
    ax.set_ylabel("discounted payback of the best cell, years")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("B2 — What it would take: payback of the best plant against "
                 "process-chain cost", fontweight="bold")
    fig.savefig(FIGURES / "B2_capex_frontier.png", dpi=150)
    plt.close(fig)


def fig_b3(p, g, plt):
    """Production surface, with the production and payback optima marked."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
    lo = g.ch4_kg_per_day.min()
    hi = g.ch4_kg_per_day.max()
    im = None
    for ax, site in zip(axes, SITES):
        sub = g[g.site == site]
        grid = _grid(sub, "ch4_kg_per_day")
        best_prod = np.unravel_index(int(np.nanargmax(grid)), grid.shape)
        pay = p[(p.site == site) & (np.isclose(p["mult"], 0.10))]
        pgrid = _grid(pay, "payback")
        marks = [(best_prod, "#111111")]
        if np.isfinite(pgrid).any():
            masked = np.where(np.isfinite(pgrid), pgrid, np.inf)
            marks.append((np.unravel_index(int(np.argmin(masked)), pgrid.shape),
                          "#C1453B"))
        im = _heat(ax, grid, ".1f", "viridis", lo, hi, highlight=marks)
        ax.set_title(site, fontweight="bold")
        ax.set_xlabel("battery, kWh")
        if ax is axes[0]:
            ax.set_ylabel("PV array, kWp")
    fig.colorbar(im, ax=axes, label="methane, kg/day (seasonally weighted)")
    fig.suptitle("B3 — Production surface.  black = most methane,  "
                 "red = fastest payback (at capex x0.10)", fontweight="bold")
    fig.savefig(FIGURES / "B3_production.png", dpi=150)
    plt.close(fig)


def fig_b4(seasonal, plt):
    """Summer against winter: the resource story."""
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), constrained_layout=True)
    lo = seasonal.ch4_kg_per_day.min()
    hi = seasonal.ch4_kg_per_day.max()
    im = None
    for i, site in enumerate(SITES):
        for j, season in enumerate(("summer", "winter")):
            sub = seasonal[(seasonal.site == site) & (seasonal.season == season)]
            ax = axes[i, j]
            im = _heat(ax, _grid(sub, "ch4_kg_per_day"), ".0f", "viridis", lo, hi)
            ax.set_title(f"{site} — {season}", fontweight="bold")
            if j == 0:
                ax.set_ylabel("PV array, kWp")
            if i == 1:
                ax.set_xlabel("battery, kWh")
    fig.colorbar(im, ax=axes, label="methane, kg/day", shrink=0.7)
    fig.suptitle("B4 — Seasonal shifts in methane production", fontweight="bold")
    fig.savefig(FIGURES / "B4_seasonal.png", dpi=150)
    plt.close(fig)


def fig_b5(p, plt):
    """Every cell's payback against capex -- ordering *and* how tight it is.

    This replaces a plot of the argmin cell, which implied a migration the data
    does not support: at x0.25 the two best Seville cells differ by 0.34 years
    (3.5 %), well inside the 6.2-point tilt left by the two-season weighting.
    Tracking the winner would have drawn a confident line through noise.

    Plotting the whole family separates what is robust from what is not. The
    three **battery** families do not overlap -- 500 kWh sits below 1500, which
    sits below 2500, at every capex and both sites -- so that ordering is a real
    result. The three **array** sizes inside each family are within a year of
    one another, so which array is "best" is not determined by this grid.
    """
    batt_colour = {500.0: "#2A9D8F", 1500.0: "#E8A33D", 2500.0: "#C1453B"}
    pv_style = {700.0: ":", 1100.0: "-", 1500.0: "--"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True,
                             constrained_layout=True)
    for ax, site in zip(axes, SITES):
        ax.axhspan(HORIZON, HORIZON * 1.4, color="#3b0a0a", alpha=0.10, zorder=0)
        for kwh in BATT:
            for pv in PV:
                s = p[(p.site == site) & (p.battery_kwh == kwh)
                      & (p.pv_kwp == pv)].sort_values("mult")
                fin = s[np.isfinite(s.payback)]
                if not len(fin):
                    continue
                ax.plot(fin["mult"], fin["payback"], pv_style[pv], lw=1.6,
                        color=batt_colour[kwh], zorder=3,
                        label=f"{kwh:.0f} kWh" if pv == 1100.0 else None)
        ax.set_title(site, fontweight="bold")
        ax.set_xlabel("process-chain capex, fraction of EUR 2200/kW")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, HORIZON * 1.25)
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("discounted payback, years")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=2, label=f"{k:.0f} kWh")
               for k, c in batt_colour.items()]
    handles += [Line2D([], [], color="0.4", ls=st, lw=1.6, label=f"{pv:.0f} kWp")
                for pv, st in pv_style.items()]
    axes[1].legend(handles=handles, frameon=False, fontsize=8, ncol=2,
                   loc="upper left")
    fig.suptitle("B5 — All nine configurations: battery size separates cleanly, "
                 "array size does not", fontweight="bold")
    fig.savefig(FIGURES / "B5_all_cells.png", dpi=150)
    plt.close(fig)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.grid": False, "font.size": 9,
                         "axes.titleweight": "bold"})
    FIGURES.mkdir(parents=True, exist_ok=True)

    stress = measured_stress()
    print(f"measured stress for {len(stress)} configurations, "
          f"range {min(stress.values()):.3f}-{max(stress.values()):.3f}")
    g = weighted()
    seasonal = g.attrs["seasonal"]
    p = payback_table(g, stress)
    p.to_csv(RESULTS / "joint_payback.csv", index=False)
    g.to_csv(RESULTS / "joint_sizing_weighted.csv", index=False)

    for fn, args in ((fig_b1, (p, plt)), (fig_b2, (p, plt)),
                     (fig_b3, (p, g, plt)), (fig_b4, (seasonal, plt)),
                     (fig_b5, (p, plt))):
        try:
            fn(*args)
            print(f"  ok  {fn.__name__}")
        except Exception as exc:
            print(f"  --  {fn.__name__}: {type(exc).__name__}: {exc}")

    print("\nbest cell per site, by multiplier:")
    for site in SITES:
        for m in PANEL_MULT + (0.0,):
            s = p[(p.site == site) & (np.isclose(p["mult"], m))]
            fin = s[np.isfinite(s.payback)]
            if len(fin):
                b = fin.loc[fin.payback.idxmin()]
                print(f"  {site:8s} x{m:.2f}  {b.payback:5.1f} yr at "
                      f"{b.pv_kwp:.0f} kWp / {b.battery_kwh:.0f} kWh")
            else:
                print(f"  {site:8s} x{m:.2f}  never")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
