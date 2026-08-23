"""PLACEHOLDER lumped power-to-methane block (milestone M0 only).

This stands in for the real chain -- air contactor, calciner, electrolyser, H2
buffer, Sabatier -- so that the surrounding machinery (weather, dispatch,
metrics, baselines, reporting) can be built and tested end to end first. It is
deleted in M1, not extended.

It is not a strawman, though. It carries the two features that make the
scheduling problem non-trivial in the first place:

    minimum load    the chain cannot idle at 5 %; below `min_load_fraction` it
                    is off, so the controller faces a genuine commitment decision
    warm-up state   production lags power, and a cold start costs energy, so
                    stopping at every passing cloud is expensive

What it deliberately does *not* have is the thing the real model exists to
capture: separate buffers with separate time constants. Here methane appears the
instant power is applied. In the real plant CO2 and H2 accumulate in inventories
that decouple the subsystems in time, which is where the actual optimisation
value lives. The gap between this block's results and M1's is a useful measure of
how much the buffering is worth.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sfp.models import mathx as mx
from sfp.models.base import Signal, Subsystem, VectorSpec
from sfp.units import J_PER_KWH


class AggregateProcess(Subsystem):
    """Lumped power-to-methane chain with warm-up dynamics and a minimum load."""

    name = "process"

    @property
    def states(self) -> VectorSpec:
        return VectorSpec(
            (
                Signal("warmth", "-", "thermal readiness, 0 cold to 1 at temperature",
                       lower=0.0, upper=1.0),
                Signal("ch4_kg", "kg", "cumulative methane produced", lower=0.0),
                Signal("starts", "-", "cumulative start-up count (continuous proxy)", lower=0.0),
            )
        )

    @property
    def inputs(self) -> VectorSpec:
        return VectorSpec(
            (
                Signal(
                    "load_fraction",
                    "-",
                    "commanded electrical load as a fraction of rating",
                    lower=0.0,
                    upper=1.0,
                ),
                Signal(
                    "enable",
                    "-",
                    "1 = energised (parasitics live, plant held warm), 0 = dark and cooling",
                    lower=0.0,
                    upper=1.0,
                ),
            )
        )

    # --- ratings ----------------------------------------------------------
    @property
    def rated_power_W(self) -> float:
        return self.p.rated_power_kw * 1e3

    @property
    def min_power_W(self) -> float:
        return self.rated_power_W * self.p.min_load_fraction

    @property
    def standby_power_W(self) -> float:
        return self.p.standby_power_kw * 1e3

    # --- physics ----------------------------------------------------------
    def _enable_gate(self, u):
        """1 when the block is energised, 0 when it is dark.

        A de-energised plant draws nothing at all -- no controls, no fans, no
        purge -- and coasts down in temperature. Making this an explicit input
        rather than an implicit consequence of the power balance matters: whether
        to hold the plant warm through the night, paying parasitic load to avoid
        a cold start in the morning, is a genuine economic decision. In M1 it
        becomes the central one, because holding a 900 degC kiln warm overnight
        is enormously expensive and letting it cool is worse.
        """
        return mx.smooth_step(mx.clip(u[1], 0.0, 1.0) - 0.5, width=0.05)

    def _running_gate(self, load):
        """1 when the block is producing, 0 when it is not.

        The threshold sits at half the minimum turndown, several transition
        widths away from both 0 and `min_load_fraction`, so the gate saturates
        properly at each end. Placing it a *single* width from zero would leave
        the gate at tanh(-1)/2 + 1/2 = 0.12 while the plant is stopped, which
        silently charges a stopped plant ~12 % of its warm-up power forever.
        """
        threshold = 0.5 * self.p.min_load_fraction
        return mx.smooth_step(load - threshold, width=0.02 * self.p.min_load_fraction)

    def _effective_load(self, u):
        """Commanded load fraction, snapped to 0 below the minimum load.

        Smooth rather than a hard branch so the same expression is usable inside
        an NLP: the gate is a tanh centred on the minimum load. A load command
        means nothing while the block is de-energised.
        """
        commanded = mx.clip(u[0], 0.0, 1.0)
        # Centre the gate *below* the minimum load, not on it, so that commanding
        # exactly `min_load_fraction` saturates the gate at 1 rather than at 0.5.
        # A gate centred on its own threshold halves the very setpoint it is
        # meant to admit.
        centre = 0.6 * self.p.min_load_fraction
        gate = mx.smooth_step(commanded - centre, width=0.05 * self.p.min_load_fraction)
        return commanded * gate * self._enable_gate(u)

    def specific_energy_J_per_kg(self, load_fraction):
        """Electricity per kg of methane, rising at part load.

        Linear interpolation of the part-load penalty between minimum load and
        rated. Stands in for the electrolyser polarisation curve, whose real
        shape -- efficiency peaking below rated -- lands in M1.
        """
        nominal = self.p.specific_energy_kwh_per_kg * J_PER_KWH
        span = mx.fmax(1.0 - self.p.min_load_fraction, 1e-6)
        excess = mx.clip((1.0 - load_fraction) / span, 0.0, 1.0)
        return nominal * (1.0 + self.p.part_load_penalty * excess)

    def production_rate_kg_s(self, x, u):
        """Methane production, kg/s. Scales with both power and thermal readiness."""
        warmth = mx.clip(x[0], 0.0, 1.0)
        load = self._effective_load(u)
        power_W = load * self.rated_power_W
        specific = self.specific_energy_J_per_kg(load)
        return warmth * power_W / specific

    def rhs(self, t, x, u, w: Mapping[str, Any]):
        warmth = mx.clip(x[0], 0.0, 1.0)
        load = self._effective_load(u)

        # first-order warm-up towards the commanded load, first-order decay when idle
        running = self._running_gate(load)
        target = running * 1.0
        tau = mx.if_else(
            running > 0.5,
            self.p.warmup_time_s,
            self.p.cooldown_time_s,
        )
        d_warmth = (target - warmth) / tau

        d_ch4 = self.production_rate_kg_s(x, u)

        # continuous proxy for start-up count: accumulates while warming under power
        d_starts = running * mx.fmax(1.0 - warmth, 0.0) / self.p.warmup_time_s

        return mx.vertcat(d_warmth, d_ch4, d_starts)

    def outputs(self, t, x, u, w: Mapping[str, Any]) -> dict[str, Any]:
        warmth = mx.clip(x[0], 0.0, 1.0)
        load = self._effective_load(u)
        running = self._running_gate(load)
        enabled = self._enable_gate(u)

        process_W = load * self.rated_power_W
        # parasitics exist only while energised -- a dark plant draws nothing
        standby_W = self.standby_power_W * enabled
        warmup_W = (
            running
            * mx.fmax(1.0 - warmth, 0.0)
            * self.p.startup_energy_kwh
            * J_PER_KWH
            / self.p.warmup_time_s
        )
        total_W = process_W + standby_W + warmup_W

        rate_kg_s = self.production_rate_kg_s(x, u)

        return {
            "power_electrical_W": total_W,
            "process_power_W": process_W,
            "process_standby_W": standby_W,
            "process_warmup_W": warmup_W,
            "process_load_fraction": load,
            "process_enabled": enabled,
            "process_warmth": warmth,
            "ch4_rate_kg_s": rate_kg_s,
            "ch4_total_kg": x[1],
            "process_running": running,
            "process_starts": x[2],
        }

    def initial_state(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    # --- helpers for the dispatcher --------------------------------------
    def power_for_load(self, x, load_fraction: float, enable: float = 1.0) -> float:
        """Electrical draw at a given commanded load, W. Used to size a dispatch."""
        u = np.array([load_fraction, enable], dtype=float)
        return float(self.outputs(0.0, x, u, {})["power_electrical_W"])

    def parasitic_power_W(self, x, enable: float = 1.0) -> float:
        """Draw with the load at zero but the plant energised -- the floor a
        dispatch must clear before any production is possible."""
        return self.power_for_load(x, 0.0, enable)

    def load_for_power(self, x, power_W: float, enable: float = 1.0) -> float:
        """Invert `power_for_load`: the load fraction that draws roughly `power_W`.

        Analytic inverse of the affine part; warm-up and standby are treated as
        a fixed offset because neither depends on the commanded load. Returns 0
        when the remainder cannot clear the minimum turndown.
        """
        overhead = self.parasitic_power_W(x, enable)
        usable = max(power_W - overhead, 0.0)
        load = usable / self.rated_power_W
        if load < self.p.min_load_fraction:
            return 0.0
        return float(min(load, 1.0))
