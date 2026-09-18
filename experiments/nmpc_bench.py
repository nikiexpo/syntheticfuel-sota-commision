"""A/B harness for inner-NMPC solver changes, on states that are actually busy.

Why replay rather than a short simulation
-----------------------------------------
Every timing number in this project so far has been distorted one of two ways.

**Idle windows.** A quarter-day run from midnight is 99.8 % night: nothing is
committed, `capped` is empty, the problems are trivial, and the filter reported
98.6 % success -- which then fell to 78 % over three days. Solve cost tracks how
much of the plant is *running*, not elapsed time, so a benchmark that averages
over darkness measures almost nothing.

**Feedback divergence.** Two solver variants in closed loop stop seeing the same
states after the first difference, so their timings are not comparable.

This harness fixes both. It replays a **fixed sequence of consecutive states**,
sampled from a real `dispatch` trajectory and filtered to intervals where at
least `MIN_COMMITTED` machines are running. Every variant sees byte-identical
inputs, so a difference in iterations is a difference in the solver and nothing
else. Consecutive states are kept in order, because that is what makes a warm
start meaningful -- a shuffled set would test cold starts wearing a disguise.

The states come from a `dispatch`-only run, so they are not exactly what
`dispatch-nmpc` would visit. That is deliberate: a fixed reference trajectory is
a controlled experiment, and the alternative -- each variant generating its own
states -- is the thing being avoided.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfp.cli import REFERENCE_SIZING, build_plant, controllers_for  # noqa: E402
from sfp.control.base import ControlContext  # noqa: E402
from sfp.control.dispatch_nmpc import DispatchNMPCController  # noqa: E402
from sfp.economics import Economics  # noqa: E402
from sfp.sim.simulator import SimulationConfig, simulate  # noqa: E402
from sfp.weather import pvgis  # noqa: E402
from sfp.weather.series import Site, WeatherSeries  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "results" / "nmpc_bench_states.npz"

SITE = Site(37.3891, -5.9845, 10.0, name="Seville, ES")
START_DAY = 172
#: Only sample intervals with at least this many machines committed. Three of
#: four is the regime where the expensive solves live -- measured at 24 s
#: against under a second at night.
MIN_COMMITTED = 3
#: Spacing between replayed states, matching the NMPC's own resolve interval so
#: the sequence is a genuine receding-horizon succession.
SAMPLE_DT_S = 600.0


def _weather(days: int = 2) -> WeatherSeries:
    frame = pvgis.load_weather(SITE.latitude, SITE.longitude, SITE.altitude)
    return WeatherSeries(pvgis.slice_days(frame, START_DAY, days), SITE)


def generate(n_samples: int, source_days: float = 0.75) -> dict:
    """Run `dispatch` once and keep the busy intervals."""
    print(f"generating reference trajectory ({source_days} d of `dispatch`) ...",
          flush=True)
    weather = _weather()
    plant = build_plant(REFERENCE_SIZING["pv_kwp"], REFERENCE_SIZING["battery_kwh"],
                        REFERENCE_SIZING["battery_kw"],
                        REFERENCE_SIZING["calciner_kw"])
    config = SimulationConfig(days=source_days, start_day=START_DAY, dt_s=60.0,
                              control_interval_s=300.0, seed=0,
                              forecast_skill=0.75)
    started = time.perf_counter()
    result = simulate(plant, controllers_for(["dispatch"])[0], weather, config,
                      economics=Economics())
    log = result.log
    print(f"  {time.perf_counter() - started:.0f} s", flush=True)

    keys = ("contactor", "calciner", "electrolyser", "sabatier")
    committed = sum((log[f"enable_{k}"] >= 0.5).astype(int) for k in keys)
    state_cols = [c for c in log.columns if c.startswith("state.")]

    step = int(SAMPLE_DT_S / config.dt_s)
    rows, times, counts = [], [], []
    for i in range(0, len(log), step):
        if committed.iloc[i] < MIN_COMMITTED:
            continue
        rows.append(log[state_cols].iloc[i].to_numpy(dtype=float))
        times.append(i * config.dt_s)
        counts.append(int(committed.iloc[i]))
        if len(rows) >= n_samples:
            break

    if not rows:
        raise SystemExit(
            f"no interval had {MIN_COMMITTED}+ machines committed in "
            f"{source_days} d -- lower MIN_COMMITTED or lengthen the run")
    print(f"  kept {len(rows)} busy samples, "
          f"t = {times[0] / 3600:.1f}-{times[-1] / 3600:.1f} h, "
          f"committed {min(counts)}-{max(counts)}", flush=True)
    return {"x": np.array(rows), "t": np.array(times, dtype=float),
            "committed": np.array(counts), "names": np.array(
                [c[len("state."):] for c in state_cols])}


def load(n_samples: int, regen: bool) -> dict:
    if CACHE.exists() and not regen:
        d = dict(np.load(CACHE, allow_pickle=False))
        if len(d["t"]) >= n_samples:
            return {k: (v[:n_samples] if k != "names" else v) for k, v in d.items()}
    d = generate(n_samples)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **d)
    return d


def replay(label: str, data: dict, *, horizon_s: float = 3600.0,
           dt_s: float = 600.0, **nmpc_kwargs) -> dict:
    """Push the fixed state sequence through one NMPC configuration.

    `resolve_interval_s` is pinned at the replay spacing rather than defaulting
    to `dt_s`, so every configuration performs exactly one solve per sample.
    Without that, halving `dt_s` would silently change the solve *cadence* as
    well as the problem, and the comparison would measure two things at once.
    """
    weather = _weather()
    plant = build_plant(REFERENCE_SIZING["pv_kwp"], REFERENCE_SIZING["battery_kwh"],
                        REFERENCE_SIZING["battery_kw"],
                        REFERENCE_SIZING["calciner_kw"])
    ctrl = DispatchNMPCController(objective="filter", nmpc_horizon_s=horizon_s,
                                  nmpc_dt_s=dt_s,
                                  resolve_interval_s=SAMPLE_DT_S)
    ctrl.reset(ControlContext(plant=plant, site=SITE, dt_s=300.0,
                              forecast=weather, economics=Economics()))
    for key, value in nmpc_kwargs.items():
        setattr(ctrl.nmpc, key, value)

    # Time the backend directly. The controller's own `nmpc_solve_time_s`
    # diagnostic is not recorded on the failure path and the diagnostics dict is
    # never cleared between steps, so a failed solve silently carries forward
    # the previous success's time -- biasing the mean toward the cheap solves.
    backend = ctrl.nmpc.backend
    inner = backend.solve
    times: list[float] = []

    viol: list[float] = []
    sizes: list[int] = []

    def timed(nlp, **kw):
        t0 = time.perf_counter()
        out = inner(nlp, **kw)
        times.append(time.perf_counter() - t0)
        sizes.append(nlp.n_x)
        # Primal infeasibility of the returned point. `success` only reports
        # what the solver believes; this is the constraint violation actually
        # handed to the plant, which is what "converged" has to mean here.
        try:
            g = np.asarray(nlp.functions().g(out.x), dtype=float).ravel()
            viol.append(float(np.max(np.maximum(
                np.maximum(nlp.lbg - g, g - nlp.ubg), 0.0))))
        except Exception:
            viol.append(float("nan"))
        return out
    backend.solve = timed

    iters, ok, statuses = [], [], {}
    for j, (t, x) in enumerate(zip(data["t"], data["x"])):
        ctrl._next_solve_s = -np.inf          # force a solve at every sample
        ctrl._held = None
        state = plant.split(np.asarray(x, dtype=float))
        ctrl.act(float(t), state, {}, weather)
        d = ctrl.diagnostics()
        used = float(d.get("nmpc_used", 0.0))
        ok.append(used >= 0.5)
        iters.append(float(d.get("nmpc_iterations", np.nan)))
        print(f"    [{j + 1:2d}/{len(data['t'])}] t={t / 3600:5.2f} h  "
              f"{'ok ' if used >= 0.5 else 'FAIL'}  "
              f"iters={iters[-1]:6.0f}  {times[-1]:7.3f} s", flush=True)
    for k, v in ctrl._failure_reasons.items():
        statuses[k] = v

    return {"label": label, "iters": np.array(iters), "time": np.array(times),
            "ok": np.array(ok), "reasons": statuses,
            "viol": np.array(viol), "n_x": sizes[0] if sizes else 0}


def report(runs: list[dict]) -> None:
    print(f"\n{'variant':24s} {'ok':>7} {'iters mean':>11} {'iters max':>10} "
          f"{'s mean':>8} {'s max':>8} {'s total':>9}")
    base = None
    for r in runs:
        n = len(r["ok"])
        line = (f"{r['label']:26s} {r['n_x']:5d} {r['ok'].sum():3d}/{n:<3d} "
                f"{np.nanmean(r['iters']):11.1f} {np.nanmax(r['iters']):10.0f} "
                f"{r['time'].mean():8.3f} {r['time'].max():8.3f} "
                f"{r['time'].sum():9.1f} {np.nanmax(r['viol']):10.2e}")
        if base is None:
            base = r
        else:
            dt = 100.0 * (r["time"].sum() / base["time"].sum() - 1.0)
            di = 100.0 * (np.nanmean(r["iters"]) / np.nanmean(base["iters"]) - 1.0)
            line += f"   ({dt:+.0f}% time, {di:+.0f}% iters)"
        print(line)
        if r["reasons"]:
            print(f"{'':26s}   {r['reasons']}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--samples", type=int, default=12)
    p.add_argument("--regen", action="store_true")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    data = load(args.samples, args.regen)
    print(f"\nreplaying {len(data['t'])} busy states "
          f"({data['committed'].min()}-{data['committed'].max()} machines committed)\n")

    #: `warm_shift` is off everywhere here. The shift assumes the previous
    #: solve's one-step prediction approximates the next measured state, and at
    #: dt = 600 s that prediction is wrong by 40 K on the electrolyser
    #: temperature -- so it would confound a comparison whose whole point is the
    #: integration error.
    configs = [
        ("1 h / 600 s (current)", dict(horizon_s=3600.0, dt_s=600.0)),
        ("15 min / 60 s", dict(horizon_s=900.0, dt_s=60.0)),
        ("15 min / 60 s, slew/10", dict(horizon_s=900.0, dt_s=60.0,
                                        rate_limit=0.034)),
    ]
    runs = []
    for label, kwargs in configs:
        print(f"  {label}")
        runs.append(replay(label, data, warm_shift=False, **kwargs))
    report(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
