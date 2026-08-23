# Synthetic Fuel Production via Direct Air Capture under Abundant Solar Energy

Autonomous operation of a remote, solar-powered plant that converts air and water into
methane. Submission to the SoTA Energy Abundant System Challenge (Commission II, Rivan).

The plant must run itself through intermittent power, wrong forecasts and equipment
faults, with no one on site. This repository is the digital twin, the control stack and
the siting tool.

```bash
python -m sfp.cli run --site seville --pv 800 --battery 1500 --days 10 --compare
python -m sfp.cli assumptions      # regenerate docs/ASSUMPTIONS.md
python -m sfp.cli sites            # list reference European sites
pytest -q                          # 310 tests, runs offline
```

---

## The idea

Most work on renewables-plus-industry treats the battery as *the* buffer. Here the
battery is the least interesting one. This plant has three storage media with round-trip
efficiencies and time constants that differ by orders of magnitude, and the control
problem is co-scheduling them:

| Medium | Charge | Discharge | Time constant |
| --- | --- | --- | --- |
| Electrical — battery | PV surplus | any load | minutes |
| **Chemical — CaCO₃ inventory** | carbonation: cheap, exothermic, needs only fans and air | calcination: 900 °C, enormous | hours–days |
| Chemical — H₂ tank | electrolysis | Sabatier | hours |
| Thermal — kiln refractory | heater | radiative loss | hours |

Carbonate whenever air is available; calcine when power is abundant; run the Sabatier
reactor off stored H₂ and CO₂ at night, where it is nearly free electrically because it
is exothermic. **The plant's answer to "the sun went down" should mostly be chemical,
not electrical.**

## Architecture

Hierarchical economic MPC — *not* bi-level optimisation in the Stackelberg/MPEC sense.
Five layers, with a strict separation between the simulated plant and the controller's
model of it.

```
  PVGIS weather ──► [L0] TRUTH SIMULATOR  (1 min, perturbed params, sensor
                          + fault injection      noise, never seen by the controller)
                              │ noisy measurements
                              ▼
                    [L1] MHE ESTIMATOR ──► residuals ──► [FDI] detect / isolate / derate
                              │ states + slow params
                              ▼
     solar ensembles ──► [L2] ECONOMIC PLANNER   10 days @ 1 h, re-solved every 3 h
                              │                  multistage scenario tree
                              │ inventory targets, commitment schedule,
                              │ AND λ_t = shadow price of electricity
                              ▼
                    [L3] INNER NMPC            2 h @ 5 min, re-solved every 5 min
                              │ setpoints      nonlinear ROM, buys energy at λ_t
                              ▼
                    [L4] REGULATORY + INTERLOCKS   trips the NMPC cannot override
```

**Layer coupling by price.** The planner passes down not just targets but the dual
variable on its energy-balance constraint, λ_t. The inner layer then maximises product
value minus λ_t × energy consumed, subject to real dynamics and safety. This decouples
the layers cleanly, degrades gracefully when the plan is stale, and dissolves the
multiscale problem — the inner layer never needs the outer layer's time grid. The plant
runs on an internal electricity price it computes for itself.

**Objective.** Levelised cost of methane, €/kg CH₄, defined in `sfp/economics.py`. One
definition makes "operate efficiently" precise, ranks control strategies on one axis, and
is exactly the number the siting tool must report.

## Status

| Milestone | Scope | State |
| --- | --- | --- |
| **M0** | End-to-end slice: PVGIS, PV, battery, aggregate load, LCOM, two baselines, report | **done** |
| M1 | Real subsystem ROMs: carbonator, calciner, electrolyser, H₂ buffer, Sabatier | next |
| M2 | Truth simulator with plant/model mismatch, fault library | |
| M3 | Economic planner, 10 d @ 1 h, plus perfect-foresight oracle | |
| M4 | Inner NMPC and λ-price coordination | |
| M5 | MHE estimator, fault detection, isolation and recovery | |
| M6 | Stochastic solar ensembles, importance subsampling, multistage planner | |
| M7 | European siting sweep; custom NLP solver swapped in and benchmarked against IPOPT | |

Everything is written against a solver-backend interface, so the custom NLP solver drops
into the inner NMPC at M7 as a one-line change and is benchmarked against IPOPT on the
same problem instances.

### What M0 already shows

Run on measured PVGIS data for Seville, 800 kWp / 1500 kWh / 350 kW, ten days from the
June solstice:

| | greedy | rule-based |
| --- | ---: | ---: |
| methane | 1490 kg | 1328 kg |
| utilisation | 55.4 % | 49.8 % |
| curtailed | 2.2 % | 11.8 % |
| battery cycles | 9.34 | 6.34 |
| undervoltage trips | **4547** | **0** |
| LCOM | €4.03/kg | €4.38/kg |

Three findings worth keeping:

1. **Greedy is hard to beat when there is nothing to time-shift.** The M0 placeholder has
   no chemical buffer — methane appears the instant power is applied — so there is
   nothing for a planner to optimise and "run flat out" is near-optimal. This is not a
   disappointing result; it is the thesis stated negatively. The gap between M0 and M1
   measures what the buffers are worth.
2. **Greedy achieves its output by blacking out its own plant 4547 times**, draining the
   battery to the floor every evening. The rule-based controller never trips. Raw
   production is the wrong single metric, which is why LCOM exists.
3. **The plant cools completely every night** and pays a cold start every morning (visible
   in `out/timeline_*.png`). With a 900 °C kiln in M1 that becomes the dominant scheduling
   decision.

## Layout

```
sfp/
  models/     reduced-order subsystem models + provenance-tagged parameters (YAML)
  sim/        truth simulator, DC bus reconciliation, RK4 integration
  control/    controller interface and baseline strategies
  weather/    solar geometry, clear-sky, PVGIS client, forecast degradation
  report/     metrics, plots, assumptions-ledger generator
  economics.py  LCOM and the marginal objective the planner maximises
tests/        310 tests, no network required
docs/ASSUMPTIONS.md   generated, never hand-edited
```

Every model is written once against `sfp/models/mathx.py`, which dispatches on numpy or
CasADi, so the simulator, the estimator and the controller share one source of physics.
A digital twin whose controller silently disagrees with the plant is a demo, not a
control system.

## Assumptions

The brief asks entrants to distinguish supplied, measured, assumed and invented values.
Every parameter declares its own provenance in YAML and `docs/ASSUMPTIONS.md` is rendered
from those tags, so the ledger cannot drift from the code. Current split: 6 supplied,
14 literature (each with a citation), 26 assumed, 0 invented.

Two structural caveats are recorded there and repeated here because they matter:

- **Calcium looping at 420 ppm.** The dynamic Ca-looping model this project draws on
  (Cormos & Simon, 2013) is post-combustion, where CO₂ is a percent-level fraction of the
  gas. Direct air capture works at 420 ppm, where carbonation is far slower — which is why
  systems such as Carbon Engineering's put a KOH contactor and causticiser upstream of the
  CaCO₃/CaO loop rather than contacting CaO with air. The brief specifies calcium looping,
  so it is modelled as specified with rate constants reparameterised for ambient partial
  pressure, and absolute capture rates should be read as indicative.
- **Weather provenance is per-run.** PVGIS TMY is `measured`; the offline synthetic
  fallback is `invented`. Every run records and prints which it used.

## Sources

| Subsystem | Source |
| --- | --- |
| Ca-looping carbonator / calciner | Cormos & Simon (2013), *Chem. Eng. Transactions* 35, 421 |
| Sabatier reactor | Moioli, Gallandat & Züttel (2019), *Chem. Eng. J.* 375; El Sibai et al. (2016) |
| PEM electrolyser | Görgün (2006), *Int. J. Hydrogen Energy* 31, 29 |
| DAC ambient dependence | Shakouri Kalfati & Abdulla (2025), *Carbon Capture Sci. Technol.* 17 |
| Scenario reduction | *Importance subsampling for power system planning under multi-year demand and weather uncertainty* |
| Multistage robust NMPC | *Multi-stage scenario-based oil production optimisation* |
| Solar position / clear-sky | Spencer (1971); Kasten & Young (1989); Ineichen & Perez (2002); Liu & Jordan (1960) |
| Sorbent deactivation | Grasa & Abanades (2006) |
| PV thermal model | Faiman (2008), as used by PVGIS |

Source PDFs are not committed (copyright); see `resources/README.md` for the citation list.
