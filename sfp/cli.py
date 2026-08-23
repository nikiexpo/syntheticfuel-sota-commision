"""Command-line entry point.

    python -m sfp.cli run --lat 37.39 --lon -5.99 --days 10
    python -m sfp.cli run --site seville --pv 1200 --battery 2000 --compare
    python -m sfp.cli assumptions
    python -m sfp.cli sites

`run` executes one closed-loop simulation per strategy and emits the full set of
metrics the challenge brief asks for, plus figures. Everything that defines a run
is a flag, so a result can be reproduced from the command that produced it -- the
command is printed into the report header for exactly that reason.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from sfp.economics import Economics
from sfp.models.battery import Battery
from sfp.models.buffers import GasBuffer, WaterTank
from sfp.models.calciner import Calciner
from sfp.models.contactor import AirContactor
from sfp.models.electrolyser import Electrolyser
from sfp.models.pv import PVArray
from sfp.models.sabatier import SabatierReactor
from sfp.models.solids import SolidsInventory
from sfp.params import load_params
from sfp.report import assumptions as assumptions_mod
from sfp.report import plots
from sfp.report.metrics import comparison_table, compute_metrics, improvement
from sfp.sim.plant import Plant
from sfp.sim.simulator import SimulationConfig, simulate
from sfp.weather import pvgis
from sfp.weather.series import Site, WeatherSeries

# A few reference European sites spanning the solar gradient that matters for
# siting: a Nordic winter-limited site, a temperate one, and a southern one.
REFERENCE_SITES = {
    "seville": Site(37.3891, -5.9845, 10.0, name="Seville, ES"),
    "madrid": Site(40.4168, -3.7038, 650.0, name="Madrid, ES"),
    "toulouse": Site(43.6047, 1.4442, 150.0, name="Toulouse, FR"),
    "munich": Site(48.1351, 11.5820, 520.0, name="Munich, DE"),
    "amsterdam": Site(52.3676, 4.9041, 0.0, name="Amsterdam, NL"),
    "gothenburg": Site(57.7089, 11.9746, 10.0, name="Gothenburg, SE"),
}


def build_plant(
    pv_kwp: float,
    battery_kwh: float,
    battery_kw: float,
    calciner_kw: float = 150.0,
    electrolyser_scale: float = 1.0,
) -> Plant:
    """Assemble the reference plant at a given sizing.

    **Registration order is load-bearing.** It is the phase-1 evaluation order,
    and it follows the material flow: the sorbent inventory publishes its loading
    before the contactor tapers against it, the calciner publishes its CO2 rate
    before the gas buffer accumulates it, and the buffers publish their levels
    before the reactor throttles against them. `Plant` validates this at
    construction, so a wrong order raises rather than silently producing zeros.
    """
    return Plant(
        {
            "pv": PVArray(load_params("pv").override(capacity_kwp=pv_kwp)),
            "battery": Battery(
                load_params("battery").override(capacity_kwh=battery_kwh, power_kw=battery_kw)
            ),
            "solids": SolidsInventory(load_params("solids")),
            "contactor": AirContactor(load_params("contactor")),
            "calciner": Calciner(
                load_params("calciner").override(heater_power_rated_kw=calciner_kw)
            ),
            "electrolyser": Electrolyser(
                load_params("electrolyser").override(
                    n_cells=105.0 * electrolyser_scale
                )
            ),
            "gas": GasBuffer(load_params("buffers")),
            "water": WaterTank(load_params("buffers")),
            "sabatier": SabatierReactor(load_params("sabatier")),
        }
    )


def resolve_site(args: argparse.Namespace) -> Site:
    if args.site:
        key = args.site.lower()
        if key not in REFERENCE_SITES:
            raise SystemExit(
                f"unknown site {args.site!r}; known: {', '.join(sorted(REFERENCE_SITES))}"
            )
        return REFERENCE_SITES[key]
    if args.lat is None or args.lon is None:
        raise SystemExit("give either --site NAME or both --lat and --lon")
    return Site(args.lat, args.lon, args.altitude, tilt_deg=args.tilt)


def controllers_for(names: list[str]):
    from sfp.control.baselines import GreedyController, RuleBasedController

    registry = {
        "greedy": GreedyController,
        "rule-based": RuleBasedController,
    }
    out = []
    for name in names:
        if name not in registry:
            raise SystemExit(
                f"unknown controller {name!r}; available: {', '.join(sorted(registry))}"
            )
        out.append(registry[name]())
    return out


def cmd_run(args: argparse.Namespace) -> int:
    site = resolve_site(args)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    weather_frame = pvgis.load_weather(
        site.latitude,
        site.longitude,
        site.altitude,
        allow_network=not args.offline,
        seed=args.seed,
    )
    window = pvgis.slice_days(weather_frame, args.start_day, int(args.days))
    weather = WeatherSeries(window, site)

    if weather.provenance == "invented":
        print(
            "  !! WARNING: running on SYNTHETIC weather. Results are illustrative,\n"
            "     not a statement about this site. Re-run with network access for PVGIS data.\n",
            file=sys.stderr,
        )

    config = SimulationConfig(
        days=args.days,
        start_day=args.start_day,
        dt_s=args.dt,
        control_interval_s=args.control_interval,
        seed=args.seed,
        forecast_skill=args.forecast_skill,
        progress=args.progress,
    )
    economics = Economics()

    names = args.controllers if args.compare else args.controllers[:1]
    results, metrics = [], []

    print(f"\nSite      {site.name}   ({site.latitude:.3f}, {site.longitude:.3f}, {site.altitude:.0f} m)")
    print(f"Array     {args.pv:.0f} kWp at {site.tilt_deg:.1f} deg tilt")
    print(f"Battery   {args.battery:.0f} kWh / {args.battery_power:.0f} kW")
    print(f"Calciner  {args.calciner:.0f} kW")
    print(f"Window    {args.days:.0f} days from day-of-year {args.start_day}")
    print(f"Weather   {weather.provenance}: {weather.source}\n")

    for controller in controllers_for(names):
        plant = build_plant(args.pv, args.battery, args.battery_power, args.calciner)
        result = simulate(plant, controller, weather, config, economics=economics)
        m = compute_metrics(result, economics)
        results.append(result)
        metrics.append(m)

        print(f"--- {m.controller} " + "-" * (54 - len(m.controller)))
        print(f"  methane            {m.ch4_kg:10.1f} kg   ({m.ch4_kg_per_day:.1f} kg/day)")
        print(f"  utilisation        {m.utilisation:10.1%}")
        print(f"  starts             {m.starts:10.1f}")
        print(f"  PV available       {m.pv_available_kwh:10.0f} kWh")
        print(f"  PV curtailed       {m.pv_curtailed_kwh:10.0f} kWh  ({m.curtailment_fraction:.1%})")
        print(f"  inverter clipping  {m.pv_clipped_kwh:10.0f} kWh")
        print(f"  specific energy    {m.specific_energy_kwh_per_kg:10.1f} kWh/kg")
        print(f"  system efficiency  {m.system_efficiency_lhv:10.1%} (LHV)")
        print(f"  battery cycles     {m.battery_efc:10.2f} EFC   SoC {m.soc_min:.2f}-{m.soc_max:.2f}")
        print(f"  limiting subsystem {m.limiting_subsystem:>10}")
        print(f"  LCOM               {m.lcom_eur_per_kg:10.2f} EUR/kg  ({m.lcom_eur_per_mwh:.0f} EUR/MWh)")
        print(f"  bus interventions  {m.bus_interventions:10d}   undervoltage trips {int(result.log['bus_tripped'].sum()):d}")
        print(f"  wall time          {m.wall_time_s:10.1f} s")

        if not args.no_plots:
            plots.timeline(result, m, outdir / f"timeline_{m.controller}.png")

    if len(metrics) > 1:
        table = comparison_table(metrics)
        print("\n=== strategy comparison " + "=" * 40)
        print(table.to_string(float_format=lambda v: f"{v:,.3f}"))

        baseline = next((m for m in metrics if m.controller == "greedy"), metrics[-1])
        for m in metrics:
            if m is baseline:
                continue
            delta = improvement(m, baseline)
            print(f"\n  {m.controller} vs {baseline.controller}:")
            print(f"    methane      {delta['ch4_kg']:+.1%}")
            print(f"    LCOM         {delta['lcom_eur_per_kg']:+.1%}  (negative is better)")
            print(f"    battery use  {delta['battery_efc']:+.1%}")

        if not args.no_plots:
            plots.comparison(metrics, outdir / "comparison.png")
            plots.limiting_subsystem(metrics, outdir / "limiting.png")
        table.to_csv(outdir / "comparison.csv")

    # Compressed CSV rather than parquet: no optional engine to install, and a
    # public repo should be readable with nothing but pandas.
    for result, m in zip(results, metrics):
        result.log.to_csv(outdir / f"log_{m.controller}.csv.gz", compression="gzip")

    if not args.no_plots:
        print(f"\nFigures and logs written to {outdir.resolve()}")
    return 0


def cmd_assumptions(args: argparse.Namespace) -> int:
    path = assumptions_mod.write()
    print(f"assumptions ledger written to {path}")
    return 0


def cmd_sites(args: argparse.Namespace) -> int:
    print("Reference sites (use with --site):\n")
    for key, site in REFERENCE_SITES.items():
        print(f"  {key:<12} {site.name:<18} {site.latitude:7.3f}, {site.longitude:8.3f}   "
              f"{site.altitude:5.0f} m   tilt {site.tilt_deg:.1f} deg")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sfp",
        description="Autonomous solar synthetic-fuel plant: digital twin, control and siting.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="simulate the plant and report")
    run.add_argument("--site", type=str, help="named reference site (see `sites`)")
    run.add_argument("--lat", type=float, help="latitude, degrees north")
    run.add_argument("--lon", type=float, help="longitude, degrees east")
    run.add_argument("--altitude", type=float, default=0.0, help="site altitude, m")
    run.add_argument("--tilt", type=float, default=-1.0, help="array tilt, deg (-1 = auto from latitude)")
    run.add_argument("--pv", type=float, default=500.0, help="array capacity, kWp")
    run.add_argument("--battery", type=float, default=1000.0, help="battery energy, kWh")
    run.add_argument("--battery-power", type=float, default=400.0, help="battery power, kW")
    run.add_argument("--calciner", type=float, default=150.0, help="calciner heater rating, kW")
    run.add_argument("--days", type=float, default=10.0, help="simulation length, days")
    run.add_argument("--start-day", type=int, default=172, help="day of year to start (172 = 21 June)")
    run.add_argument("--dt", type=float, default=60.0, help="plant integration step, s")
    run.add_argument("--control-interval", type=float, default=300.0, help="supervisory control period, s")
    run.add_argument("--forecast-skill", type=float, default=0.75, help="forecast correlation at zero lead time")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--controllers", nargs="+", default=["greedy", "rule-based"])
    run.add_argument("--compare", action="store_true", help="run every controller and compare")
    run.add_argument("--offline", action="store_true", help="never touch the network")
    run.add_argument("--no-plots", action="store_true")
    run.add_argument("--progress", action="store_true")
    run.add_argument("--out", type=str, default="out")
    run.set_defaults(func=cmd_run)

    assumptions = sub.add_parser("assumptions", help="regenerate docs/ASSUMPTIONS.md")
    assumptions.set_defaults(func=cmd_assumptions)

    sites = sub.add_parser("sites", help="list the reference sites")
    sites.set_defaults(func=cmd_sites)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
