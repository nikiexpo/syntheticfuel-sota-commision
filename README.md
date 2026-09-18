# Synthetic Fuel Production via Direct Air Capture under Abundant Solar Energy

Autonomous operation of a remote, solar-powered plant that converts air and water into
methane. Submission to the SoTA Energy Abundant System Challenge (Commission II, Rivan).

This repository is the digital twin, the control stack and the sizing studies. The
write-up introduces the work; this file is an index and a set of commands.

## Install

```bash
python -m venv venv && venv/Scripts/activate      # or: source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

```bash
# one closed-loop run, every controller, full report and figures into out/
python -m sfp.cli run --site seville --pv 1100 --battery 1500 --days 7 --compare

# the shipped controller only
python -m sfp.cli run --site seville --days 7 --controllers dispatch-nmpc

python -m sfp.cli sites                     # reference sites
python -m sfp.cli assumptions               # regenerate docs/ASSUMPTIONS.md
pytest -q                                   # 423 tests, offline
```

Experiments write to `experiments/results/` and `experiments/figures/`. Both are
committed, so the figures can be regenerated without re-running the simulations. Each
sweep rewrites its CSV after every cell and resumes from it, so an interrupted run keeps
what it has. Add `--days 1 --seasons summer` for a smoke test.

```bash
python experiments/battery_sizing.py          # experiment A   40 cells, ~5.4 h
python experiments/battery_sizing_figures.py  #                figures A1-A7

python experiments/joint_sizing.py            # experiment B   36 cells, ~8.8 h
python experiments/joint_sizing_figures.py    #                figures B1-B5

python experiments/price_strategy.py          # controller price sensitivity
python experiments/nmpc_bench.py              # inner-solver A/B on replayed states
python experiments/timeline_from_series.py --all   # timelines from stored logs
```

`docs/MODEL.tex` builds with `pdflatex docs/MODEL.tex`.

## Where things are

| | |
| --- | --- |
| `docs/DISPATCH_NMPC.md` | **the control architecture** — both optimisation problems in full, what passes between them, measured behaviour, open items |
| `docs/MODEL.tex` | every subsystem model: full nonlinear form, then the LP approximation |
| `docs/ASSUMPTIONS.md` | generated parameter ledger — 139 values, each tagged `supplied` / `measured` / `literature` / `assumed` / `invented` |
| `experiments/BATTERY_SIZING.md` | experiment A: closed-loop battery sizing, and what the safety layer does |
| `experiments/results/` | every CSV and trajectory the reports are built from |

| package | |
| --- | --- |
| `sfp/models/` | reduced-order subsystem models, plus provenance-tagged parameters in YAML |
| `sfp/sim/` | truth simulator, DC bus reconciliation, RK4 integration |
| `sfp/control/` | `dispatch.py` (outer MILP), `nmpc.py` (inner filter), `dispatch_nmpc.py` (the two together), `baselines/` |
| `sfp/solvers/` | NLP backend interface, IPOPT backend, LP/MILP seam |
| `sfp/weather/` | solar geometry, clear-sky, PVGIS client, forecast degradation |
| `sfp/report/` | metrics, plots, plan-fidelity, assumptions generator |
| `sfp/economics.py` | LCOM, capex, and the marginal objective the outer layer maximises |

## Controllers

`--controllers` takes any of:

| name | |
| --- | --- |
| `dispatch-nmpc` | **the submission.** Outer economic MILP (240 h, hourly) over an inner NMPC safety filter (5 min, 1 min) on the full 16-state model |
| `dispatch` | the outer MILP alone, for measuring what the inner layer adds |
| `greedy` | run everything flat out whenever there is power — the brief's baseline |
| `rule-based` | PLC-style: buffer bands, hysteresis, evening reserve, no forecast |
| `oracle` | perfect-foresight bound, for quoting regret |
| `planner`, `hierarchical` | the earlier NLP planner and its price-coordinated hierarchy, kept as comparators |

Every model is written once against `sfp/models/mathx.py`, which dispatches on numpy or
CasADi, so the simulator and the controller share one source of physics.

## Sources

| Subsystem | Source |
| --- | --- |
| Ca-looping carbonator / calciner | Cormos & Simon (2013), *Chem. Eng. Transactions* 35, 421 |
| Calcination equilibrium | Baker (1962), *J. Chem. Soc.* |
| Sabatier reactor | Moioli, Gallandat & Züttel (2019), *Chem. Eng. J.* 375; El Sibai et al. (2016) |
| PEM electrolyser | Görgün (2006), *Int. J. Hydrogen Energy* 31, 29 |
| DAC ambient dependence | Shakouri Kalfati & Abdulla (2025), *Carbon Capture Sci. Technol.* 17 |
| Sorbent deactivation | Grasa & Abanades (2006) |
| Solar position / clear-sky | Spencer (1971); Kasten & Young (1989); Ineichen & Perez (2002); Liu & Jordan (1960) |
| PV thermal model | Faiman (2008), as used by PVGIS |

Source PDFs are not committed (copyright); `resources/README.md` has the citation list.
