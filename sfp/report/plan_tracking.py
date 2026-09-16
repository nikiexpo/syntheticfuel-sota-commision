"""How faithfully the plant follows the dispatch layer's plan.

This exists to answer one question, and the answer decides what a parameter
sweep is allowed to be made of.

Sweeping profitability across battery sizes, methane prices, sites and seasons
is affordable on *plans* -- a 240 h LP costs well under a second -- and is not
affordable on closed-loop simulations, which cost minutes to hours each. Using
plans as a stand-in is only legitimate if the plan is a faithful predictor of
what the plant actually does, and "faithful" is a number, not an assertion.

Two errors, answering different questions
-----------------------------------------
**Physical divergence.** Where the buffers actually are against where the plan
said they would be, normalised by each buffer's own capacity. This is the LP's
linearised model measured against the real plant.

**Profit error.** Realised minus predicted operating profit over the same
window. This is the one a sweep rests on, and it is *not* implied by the first:
inventories can drift a long way while the economics still land, and a small
drift caused by a mis-timed commitment can cost real money.

What is compared
----------------
Only the marginal terms the dispatch layer actually optimises -- methane
revenue, battery wear, sorbent deactivation, water, and start-ups. Capital is
sunk and appears in neither. The realised side is reconstructed from the plant's
own logged states, so it is what happened, not what any controller believed.

A caveat worth carrying into the write-up: the planned figure covers the window
the *superseded* plan was responsible for, which is the replan interval, not the
plan's whole horizon. A plan is only ever accountable for the part of itself
that was implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sfp.economics import Economics

#: Controllable subsystems whose start-ups the dispatch layer charges for.
_STARTERS = ("contactor", "calciner", "electrolyser", "sabatier")


@dataclass
class PlanTrackingSummary:
    """Aggregate fidelity of plan against outturn."""

    windows: int
    planned_EUR: float
    realised_EUR: float
    error_EUR: float
    error_fraction: float
    #: Mean |error| per window as a fraction of the *mean* window's planned
    #: profit, not of each window's own. Normalising per window looks natural
    #: and is useless: a window whose plan nets near zero -- dawn, or a
    #: commitment that pays for itself exactly -- divides a finite error by
    #: almost nothing and reports hundreds of per cent. The aggregate scale is
    #: the meaningful denominator, and it is the one a sweep would use.
    mean_abs_error_EUR: float
    mean_abs_error_normalised: float
    divergence_rms_mean: float
    divergence_rms_max: float

    def describe(self) -> str:
        return (
            f"{self.windows} replan windows: plan predicted "
            f"EUR {self.planned_EUR:.1f}, plant delivered "
            f"EUR {self.realised_EUR:.1f} "
            f"({self.error_fraction * 100:+.1f} % aggregate). "
            f"Per window, mean |error| EUR {self.mean_abs_error_EUR:.1f} "
            f"= {self.mean_abs_error_normalised * 100:.0f} % of a mean window. "
            f"Inventory divergence RMS mean {self.divergence_rms_mean * 100:.2f} % "
            f"of capacity, worst {self.divergence_rms_max * 100:.2f} %."
        )


def _starts_in(window: pd.DataFrame) -> float:
    """Rising edges on the enable channels, i.e. start-ups actually incurred."""
    total = 0.0
    for key in _STARTERS:
        col = f"enable_{key}"
        if col not in window:
            continue
        on = (window[col].to_numpy(dtype=float) >= 0.5).astype(int)
        if len(on) > 1:
            total += float(np.count_nonzero(np.diff(on) > 0))
    return total


def plan_tracking(result, economics: Economics | None = None,
                  start_cost_EUR: float | None = None) -> pd.DataFrame:
    """One row per replan window, comparing planned against realised profit.

    Returns an empty frame for a controller that publishes no plan diagnostics,
    which is the honest answer for the rule-based and greedy baselines: they do
    not predict anything, so there is nothing to score them against.
    """
    log = result.log
    economics = economics or Economics()
    if "plan_replan_index" not in log or "plan_profit_window_EUR" not in log:
        return pd.DataFrame()

    dt = result.dt_s
    plant = result.plant
    c_efc = plant["battery"].cost_per_efc_EUR()
    # Start-ups are charged per event. The dispatch layer uses a per-subsystem
    # table; a single representative figure is used here because the log records
    # that a machine started, not what the layer charged for it.
    c_start = (float(economics.p.startup_cost_EUR) if start_cost_EUR is None
               else float(start_cost_EUR))

    # The log is indexed by timestamp, so elapsed time comes from a positional
    # counter rather than from the index itself.
    log = log.assign(_step=np.arange(len(log)))

    rows = []
    for idx, window in log.groupby("plan_replan_index"):
        if len(window) < 2 or not np.isfinite(window["plan_profit_window_EUR"].iloc[0]):
            continue
        planned = float(window["plan_profit_window_EUR"].iloc[0])
        if not np.isfinite(planned):
            continue

        def delta(col: str) -> float:
            if col not in window:
                return 0.0
            v = window[col].to_numpy(dtype=float)
            return float(v[-1] - v[0])

        ch4 = delta("state.sabatier.ch4_kg")
        efc = delta("state.battery.efc")
        water = delta("state.water.consumed_kg")
        cycles = delta("state.solids.cycle_number")
        starts = _starts_in(window)

        realised = economics.marginal_objective_EUR(
            ch4_kg=ch4, battery_efc=efc, battery_cost_per_efc=c_efc,
            starts=starts, water_kg=water,
        )
        # Sorbent deactivation is not in `marginal_objective_EUR`, so it is added
        # here using the same coefficient the dispatch layer charged: the value
        # of the capacity lost by advancing the mean cycle number by one, which
        # is `n_tot` times the solids model's per-mole figure.
        solids = plant["solids"]
        n0 = float(window.get("state.solids.cycle_number",
                              pd.Series([0.0])).iloc[0])
        c_cycle = float(solids.p.n_total_mol) * float(
            solids.marginal_deactivation_cost_per_mol(
                np.array([0.0, 0.0, n0]), float(solids.p.sorbent_cost_per_mol)))
        realised -= cycles * c_cycle

        rows.append({
            "replan": int(idx),
            "t_start_h": float(window["_step"].iloc[0] * dt / 3600.0),
            "hours": float(len(window) * dt / 3600.0),
            "planned_EUR": planned,
            "realised_EUR": realised,
            "error_EUR": realised - planned,
            "error_fraction": (realised - planned) / planned if planned else np.nan,
            "ch4_kg": ch4,
            "starts": starts,
            "divergence_rms": float(window.get(
                "plan_divergence_rms", pd.Series([np.nan])).iloc[0]),
            **{
                f"div_{name}": float(window.get(
                    f"plan_divergence_{name}", pd.Series([np.nan])).iloc[0])
                for name in ("soc", "n_h2", "n_co2", "n_caco3")
            },
        })

    return pd.DataFrame(rows)


def summarise_tracking(frame: pd.DataFrame) -> PlanTrackingSummary | None:
    """Reduce the per-window frame to the figures a write-up would quote."""
    if frame.empty:
        return None
    planned = float(frame["planned_EUR"].sum())
    realised = float(frame["realised_EUR"].sum())
    mean_abs = float(frame["error_EUR"].abs().mean())
    mean_window = abs(planned) / max(len(frame), 1)
    div = frame["divergence_rms"].dropna()
    return PlanTrackingSummary(
        windows=len(frame),
        planned_EUR=planned,
        realised_EUR=realised,
        error_EUR=realised - planned,
        error_fraction=(realised - planned) / planned if planned else float("nan"),
        mean_abs_error_EUR=mean_abs,
        mean_abs_error_normalised=mean_abs / mean_window if mean_window else float("nan"),
        divergence_rms_mean=float(div.mean()) if len(div) else float("nan"),
        divergence_rms_max=float(div.max()) if len(div) else float("nan"),
    )
