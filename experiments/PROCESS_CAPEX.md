# Process Capex Experiment

**Run:** 16 September 2026 · `experiments/process_capex.py`
**Data:** `results/process_capex.csv` · **Figures:** `figures/process_capex_{absolute,payback}.png`

A 7 × 5 grid of process-chain capital cost against methane price, at three sites,
with the plant fixed at 1100 kWp / 500 kWh. 105 cells, **zero new solves.**

---

## 1. Why this axis

The process chain — electrolyser stack and BoP, air contactor, calciner,
Sabatier reactor — is the largest capital item in the plant at every sizing
tested, larger than the array:

| item | capex | share |
| --- | ---: | ---: |
| process chain | €1,179,903 | 57 % |
| PV, 1100 kWp | €770,000 | 37 % |
| battery, 500 kWh | €125,000 | 6 % |
| **total** | **€2,074,903** | |

And it is priced by the single weakest number in the model:
`process_capex_per_kw = 2200`, `provenance: assumed`, applied to one aggregate kW
rating. Experiments 1 and 2 both concluded "the plant does not repay its
investment", and in both the dominant term in that conclusion is a number nobody
measured.

It is also the item with the most credible path to falling. The electrolyser is
62 % of the rating, and electrolyser capex is where published learning curves are
steepest — the same mechanism that took PV modules down roughly an order of
magnitude over two decades. Sweeping 0.25 × to 1.00 × is therefore a question
about a plausible 2035 plant, not a sensitivity for its own sake.

## 2. Why it cost nothing to run

**Capital cost never enters the controller's objective.** The dispatch LP trades
methane revenue against battery wear, sorbent deactivation, water and start-ups;
capex appears nowhere in `c`. So the plans are bit-identical at every multiplier,
and the sweep is exact arithmetic over the plans already solved for
`pv_sizing.py`.

Verified directly rather than assumed: Gothenburg at 1100 kWp returns
760.6586 €/day operating and 0.8573 EFC/day at both €2200/kW and €1100/kW. The
CSV shows the same invariance across all seven multipliers — mean operating
profit is €371/day at every row, to the euro.

This is worth stating rather than hiding, because it is also the experiment's
**principal limitation** (§5).

---

## 3. Results

### Time to break even, years (`inf` = not repaid within 25)

**Seville**

| ×capex \ €/kg | 2.00 | 3.00 | 4.00 | 5.00 | 6.00 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | — | — | 15.1 | 9.2 | **6.4** |
| 0.38 | — | — | 19.7 | 11.3 | 8.2 |
| 0.50 | — | — | — | 14.4 | 9.7 |
| 0.62 | — | — | — | 17.6 | 11.3 |
| 0.75 | — | — | — | 22.8 | 13.3 |
| 0.88 | — | — | — | — | 16.4 |
| 1.00 | — | — | — | — | 19.6 |

**London**

| ×capex \ €/kg | 5.00 | 6.00 |
| ---: | ---: | ---: |
| 0.25 | 17.8 | **11.9** |
| 0.38 | — | 15.0 |
| 0.50 | — | 20.0 |
| 0.62 and above | — | — |

**Gothenburg** tracks London within 0.2 yr at every repaying cell.

### Net profitability, EUR/day — Seville

| ×capex \ €/kg | 2.00 | 3.00 | 4.00 | 5.00 | 6.00 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | −188 | −64 | **+60** | +184 | +308 |
| 0.50 | −281 | −158 | −33 | +91 | +215 |
| 0.75 | −375 | −251 | −127 | −3 | +121 |
| 1.00 | −469 | −345 | −221 | −97 | +27 |

Every column is exactly linear in the multiplier at €47/day per 0.125 step, which
is €1,179,903 × (CRF + fixed opex) / 365 × 0.125. That linearity is not a
finding — it is the arithmetic, and it is shown here only so the figure is not
over-read.

---

## 4. Findings

### 4.1 Cheap process equipment is necessary but not sufficient

**Nothing repays at €3.00/kg at any multiplier in the sweep, at any site.** At
0.25 × Seville still needs €4.00/kg to repay at all.

Extrapolating the linear surface below the swept range says how far short it
falls. The multiplier at which net profitability reaches zero:

| site | €2.00/kg | €3.00/kg | €4.00/kg |
| --- | ---: | ---: | ---: |
| Seville | −0.25 | **+0.08** | +0.41 |
| London | −0.40 | −0.16 | +0.08 |
| Gothenburg | −0.40 | −0.17 | +0.07 |

A negative multiplier is not a cost — it is the statement that **the plant does
not break even at that price even if the process chain were free**, and someone
would have to pay to install it. That is the case at €2.00/kg everywhere, and at
€3.00/kg at both cloudy sites.

Seville at €3.00/kg is the one cell that survives the extrapolation, and it needs
the chain at 0.08 × — a **92 % cost reduction**, far outside anything a
learning curve supports on this horizon.

So the two levers are not substitutes. A 4× cost decline buys Seville a
**6.4-year payback** — a genuinely commercial number, against the 19.6 years it
manages today — but only in combination with a product price roughly three times
today's certificated green-methane value of about €1.80/kg. Either lever alone
leaves the plant under water.

### 4.2 The cost decline needed to make the cloudy sites work is much steeper

Seville repays at €6.00/kg at every multiplier in the sweep. London and
Gothenburg need ≤ 0.50 ×, and even at 0.25 × they are at 11.9 and 12.0 years
against Seville's 6.4.

This is the same asymmetry experiment 2 found on the PV axis and it has the same
cause: capital is spent once and the resource arrives every day, so a site with
less resource is penalised on every subsequent term. Falling capital cost helps
the good site more in absolute terms, because it has more production to spread
over the saving.

### 4.3 The sensitivity is large enough that the earlier conclusions are conditional

Experiments 1 and 2 reported "the plant does not repay its investment" as a flat
result. It is more accurately: *the plant does not repay its investment at today's
assumed process capex.* The swing across this one `assumed` parameter is
€282/day at Seville — larger than the swing across the entire battery axis of
experiment 1, and comparable to the whole PV axis of experiment 2.

That is the honest reading of a parameter carrying that much weight with that
little provenance, and it is why `08_SIMULATION_ENVIRONMENT.md` flags
`process_capex_per_kw` first in its list of load-bearing assumptions.

---

## 5. What this does not show

Everything in `RELATIVE_PROFITABILITY.md` section 4, plus two that are specific
to this experiment:

- **A uniform multiplier is the wrong cost model.** It applies electrolyser-like
  learning rates to the calciner and the air contactor, which are conventional
  process equipment with no comparable learning curve. The electrolyser is 62 %
  of the rating, so a defensible 0.25 × on the stack alone is roughly 0.53 ×
  chain-wide — meaning **0.25 × here is optimistic as a whole-chain figure even
  if it is reasonable for the stack.** A per-subsystem capex model would put the
  decline where it belongs, and is the correct fix.
- **The plant does not re-optimise against cheaper capital.** Because capex is
  absent from the controller's objective, the sizing is held at 1100 kWp /
  500 kWh across the whole sweep. In reality a cheaper process chain justifies a
  *larger* one relative to the array, which would raise utilisation and recover
  more of the curtailment experiment 2 found above ~1100 kWp. Every number here
  is therefore a **lower bound** on what a re-sized plant would achieve at that
  capex. Experiment 4 (`joint_sizing.py`) re-opens the sizing axes at 0.75 ×
  for exactly this reason, though it re-opens PV and battery rather than the
  process rating.
- **Learning curves are not modelled**, only asserted. There is no deployment
  forecast, no experience rate, and no date attached to any multiplier. "0.25 ×
  is a 2035 number" is a framing, not a result of this work.

---

## 6. Reproducing

```
python experiments/pv_sizing.py        # must run first -- supplies the plans
python experiments/process_capex.py    # seconds
```

`process_capex.py` reads `results/pv_sizing.csv`, selects the 1100 kWp rows, and
re-costs them. It will refuse to run if that file is missing, and refuse if it
contains no rows at 1100 kWp.
