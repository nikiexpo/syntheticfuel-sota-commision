"""The closed-loop simulation harness.

One loop, one set of physics, swappable controllers. Everything that
distinguishes a run -- the strategy, the weather, the injected fault -- is an
argument; the harness itself never changes. That is what makes the baseline
comparison in the report an honest one.

Two clocks, deliberately different:

    dt_s                the plant integration step (default 60 s)
    control_interval_s  how often the controller is allowed to act (default 300 s)

Real supervisory control does not run at the plant's timescale, and pretending
it does hides exactly the lag that makes intermittency hard. Between controller
calls the last setpoint is held, and the bus keeps the plant feasible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sfp.control.base import ControlContext, Controller
from sfp.sim import bus as bus_mod
from sfp.sim.plant import Plant
from sfp.weather.series import Site, WeatherSeries


@dataclass
class SimulationConfig:
    """Everything that defines a run except the controller and the plant."""

    days: float = 10.0
    start_day: int = 172  # 21 June by default
    dt_s: float = 60.0
    control_interval_s: float = 300.0
    seed: int = 0
    mismatch_scale: float = 1.0
    forecast_skill: float = 0.75
    progress: bool = False

    @property
    def duration_s(self) -> float:
        return self.days * 86400.0

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_s / self.dt_s))


@dataclass
class SimulationResult:
    """A completed run: the full log plus everything needed to interpret it."""

    log: pd.DataFrame
    config: SimulationConfig
    site: Site
    controller_name: str
    controller_description: str = ""
    weather_provenance: str = "unknown"
    weather_source: str = ""
    plant: Plant | None = None
    wall_time_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dt_s(self) -> float:
        return self.config.dt_s

    def total(self, column: str) -> float:
        """Time-integral of a rate column, in <units>*s."""
        return float(self.log[column].sum() * self.dt_s)

    def energy_kwh(self, column: str) -> float:
        """Integral of a power column (W) in kWh."""
        return float(self.log[column].sum() * self.dt_s / 3.6e6)


def _weather_row(dense: pd.DataFrame, i: int) -> dict[str, float]:
    row = dense.iloc[i]
    return {
        "poa_global": float(row["poa_global"]),
        "ghi": float(row["ghi"]),
        "ghi_clear": float(row["ghi_clear"]),
        "clearsky_index": float(row["clearsky_index"]),
        "temp_air": float(row["temp_air"]),
        "relative_humidity": float(row["relative_humidity"]),
        "wind_speed": float(row["wind_speed"]),
        "pressure": float(row["pressure"]),
        "cos_zenith": float(row["cos_zenith"]),
    }


def simulate(
    plant: Plant,
    controller: Controller,
    weather: WeatherSeries,
    config: SimulationConfig | None = None,
    *,
    economics: Any = None,
    fault_schedule: Any = None,
) -> SimulationResult:
    """Run one closed-loop simulation and return the full log.

    `fault_schedule` is accepted now and honoured from M2; passing one before
    then raises, rather than silently ignoring it and producing a fault-free
    result that looks like a fault response.
    """
    if fault_schedule is not None:
        raise NotImplementedError(
            "fault injection lands in milestone M2; refusing to run and report a "
            "fault-free result as if a fault had been injected"
        )

    config = config or SimulationConfig()
    started = time.perf_counter()

    dense = weather.densify(config.dt_s, config.duration_s)
    n_steps = min(config.n_steps, len(dense))

    pv = plant["pv"]

    forecast = weather.forecast(skill=config.forecast_skill, seed=config.seed)
    context = ControlContext(
        plant=plant,
        site=weather.site,
        dt_s=config.control_interval_s,
        horizon_s=config.duration_s,
        forecast=forecast,
        economics=economics,
        # The truth is offered here for exactly one consumer: the perfect-
        # foresight oracle, which is a bound rather than a controller. Any real
        # strategy that reached for it would be cheating, and the plant/model
        # split at M5 is what makes that distinction enforceable.
        metadata={"weather_provenance": weather.provenance, "truth": weather},
    )
    controller.reset(context)

    x = plant.initial_state()
    request = bus_mod.Request.all_off()
    last_inputs = bus_mod._inputs_from(plant, {}, {})
    last_inputs["pv"] = np.array([0.0])
    last_inputs["battery"] = np.array([0.0, 0.0])
    control_every = max(1, int(round(config.control_interval_s / config.dt_s)))

    records: list[dict[str, float]] = []

    for i in range(n_steps):
        t = i * config.dt_s
        w = _weather_row(dense, i)
        states = plant.split(x)

        # --- PV availability at this instant -----------------------------
        pv_out = pv.outputs(t, np.zeros(0), np.array([0.0]), w)
        pv_available = float(pv_out["pv_available_W"])
        pv_clipped = float(pv_out["pv_clipped_W"])

        # --- supervisory control (slower clock) --------------------------
        if i % control_every == 0:
            measurement = dict(w)
            measurement.update(
                {
                    "pv_available_W": pv_available,
                    "battery_soc": float(states["battery"][0]),
                    "time_s": t,
                }
            )
            # the controller also sees the plant's derived quantities (buffer
            # levels, temperatures, sorbent loading) as they stand right now
            measurement.update(plant.evaluate(t, x, last_inputs, w)[0])
            request = controller.act(t, states, measurement, forecast)

        # --- regulatory layer: make it feasible --------------------------
        dispatch = bus_mod.reconcile(
            request,
            plant=plant,
            state=x,
            weather=w,
            pv_available_W=pv_available,
            pv_clipped_W=pv_clipped,
            dt_s=config.dt_s,
            t=t,
        )

        curtail_fraction = (
            dispatch.pv_curtailed_W / pv_available if pv_available > 1e-9 else 0.0
        )
        u = bus_mod._inputs_from(plant, dispatch.setpoints, dispatch.enables)
        u["pv"] = np.array([curtail_fraction])
        u["battery"] = np.array([dispatch.battery_charge_W, dispatch.battery_discharge_W])
        last_inputs = u

        # --- log ----------------------------------------------------------
        record = {"time_s": t}
        record.update(w)
        # Reuse the evaluation the bus already performed at this exact state and
        # dispatch rather than repeating the most expensive call in the loop.
        # Those outputs come first because the dispatch, applied next, is
        # authoritative wherever the two overlap.
        record.update(dispatch.plant_outputs)
        record.update(dispatch.as_dict())
        # The bus evaluates the plant *before* it knows how much PV will be
        # curtailed, so it passes a curtail fraction of zero. Every PV-side
        # quantity in `plant_outputs` is therefore provisional and must be
        # replaced with the settled dispatch values.
        record["pv_delivered_W"] = dispatch.pv_used_W
        record["pv_power_W"] = -dispatch.pv_used_W
        record["power.pv"] = -dispatch.pv_used_W
        record.update({f"state.{n}": v for n, v in zip(plant.state_names(), x)})
        record.update(controller.diagnostics())
        records.append(record)

        # --- advance ------------------------------------------------------
        x = plant.step(t, x, u, w, config.dt_s)

        if config.progress and i % max(1, n_steps // 20) == 0:
            print(f"  {100.0 * i / n_steps:5.1f}%  t={t / 86400.0:5.2f} d", flush=True)

    log = pd.DataFrame.from_records(records)
    log.index = dense.index[:n_steps]
    log.index.name = "time"

    return SimulationResult(
        log=log,
        config=config,
        site=weather.site,
        controller_name=controller.name,
        controller_description=controller.description,
        weather_provenance=weather.provenance,
        weather_source=weather.source,
        plant=plant,
        wall_time_s=time.perf_counter() - started,
        metadata={"n_steps": n_steps},
    )
