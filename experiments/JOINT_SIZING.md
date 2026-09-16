# Joint Sizing Experiment

**Run:** 16 September 2026 · `experiments/joint_sizing.py`
**Data:** `results/joint_sizing.csv` · **Figures:** `figures/joint_sizing_{absolute,payback,cycling}.png`

A 5 × 5 grid of PV array against battery capacity at **Seville only**, in three
price panels. 75 cells, 300 dispatch-LP solves, no failures, 83 minutes.
Process capex at **0.75 ×** the assumed €2200/kW.

This closes the hole `PV_SIZING.md` §3 flagged: experiments 1 and 2 each swept
one design axis with the other pinned, and neither can locate a joint optimum
that sits off its own line.

**Seville only** because the cloudy sites cost three times the solve budget to
re-confirm a conclusion both earlier grids already returned for them, and because
the interesting interaction — curtailment at large arrays being recovered by
storage — is strongest where there is surplus to store.

---

## 1. Design of the grid

| axis | values | why |
| --- | --- | --- |
| PV, kWp | 300, 700, 1100, 1500, 1900 | identical to experiment 2, so every row has a comparable predecessor |
| battery, kWh | 100, 250, 500, **1000, 1500** | **not** experiment 1's axis — see below |
| price, €/kg | 3.00, 4.50, 6.00 | panels, not a grid axis |

**The battery axis was moved, not reused.** Experiment 1 ran 500–2500 kWh and
found net profitability falling monotonically, optimum pinned at the smallest
size tested and therefore *not located*. This axis runs 100–1500 kWh so the
optimum has somewhere to be, keeping 500 and 1500 as shared anchors. That change
is what produces finding 2.1.

**Price is the panel dimension** because experiment 2 found the optimal array
moving by a factor of three or four across the price range. A single-price sizing
grid answers a much narrower question than it appears to.

**Process capex at 0.75 ×** is free to assume. Capital cost never enters the
controller's objective, so the multiplier rescales the economics without moving a
single dispatch decision — the invariance experiment 3 verified directly. It
cannot relocate the optimum for a reason the physics does not support. It is set
below 1.00 × because experiment 3 showed that at today's assumed capex most of
this grid sits in territory where nothing is viable and the optimum is hard to
read.

### Cross-checks

Four cells had answers the earlier grids already fixed, on
`operating_EUR_per_day`, which is capex-independent and so unaffected by the
0.75 ×:

| source | cell | here | there | diff |
| --- | --- | ---: | ---: | ---: |
| `pv_sizing.csv` | 1100 kWp, 500 kWh @ €3 | 332.7469 | 332.7469 | −0.0000 |
| `pv_sizing.csv` | 1100 kWp, 500 kWh @ €6 | 704.9933 | 704.9933 | −0.0000 |
| `relative_profitability.csv` | 1100 kWp, 1500 kWh @ €3 | 354.0804 | 354.0776 | +0.0028 |
| `relative_profitability.csv` | 1100 kWp, 1500 kWh @ €6 | 777.3380 | 777.3376 | +0.0004 |

Exact against experiment 2. The €0.003/day against experiment 1 is 8 × 10⁻⁶
relative — MILP tie-breaking between equal-objective integer assignments, not a
discrepancy. A fifth check falls out independently: this grid puts 1100 kWp /
500 kWh at €6.00 at **13.3 years**, and `process_capex.csv` puts Seville at
0.75 × and €6.00 at **13.3 years**. Different scripts, same answer.

---

## 2. Results

### Net profitability, €/day (best cell boxed)

**€6.00/kg**

| kWp \ kWh | 100 | 250 | 500 | 1000 | 1500 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | −88 | −104 | −130 | −173 | −221 |
| 700 | +126 | **+127** | +110 | +57 | +3 |
| 1100 | +115 | +122 | +121 | +113 | +74 |
| 1500 | +58 | +65 | +65 | +53 | +18 |
| 1900 | −10 | −2 | −2 | −21 | −56 |

**€3.00/kg** — nothing positive anywhere

| kWp \ kWh | 100 | 250 | 500 | 1000 | 1500 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | −228 | −244 | −272 | −316 | −366 |
| 700 | **−173** | −182 | −206 | −261 | −316 |
| 1100 | −225 | −230 | −251 | −296 | −349 |
| 1500 | −298 | −303 | −323 | −369 | −421 |
| 1900 | −377 | −381 | −401 | −449 | −501 |

€4.50/kg sits between them, best cell −€24/day at 700 kWp / 100 kWh. Full grid in
the CSV.

### Time to break even, €6.00/kg (years)

| kWp \ kWh | 100 | 250 | 500 | 1000 | 1500 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | — | — | — | — | — |
| 700 | 12.4 | **12.3** | 13.3 | 16.2 | 20.2 |
| 1100 | 14.0 | 13.8 | 13.3 | 13.3 | 14.4 |
| 1500 | 17.8 | 17.3 | 17.2 | 17.6 | 19.2 |
| 1900 | 24.2 | 23.0 | 22.9 | 24.2 | — |

Nothing repays at €3.00 or €4.50 at any cell, consistent with experiment 3.

### Battery throughput, kWh moved per day, €6.00/kg

The quantity that makes the interaction visible:

| kWp \ kWh | 100 | 250 | 500 | 1000 | 1500 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 39 | 36 | 31 | 21 | 28 |
| 700 | 86 | 170 | 243 | 255 | 259 |
| 1100 | 91 | 219 | 427 | 812 | 1127 |
| 1500 | 91 | 218 | 426 | 830 | 1153 |
| 1900 | 90 | 217 | 425 | 823 | 1088 |

---

## 3. Findings

### 3.1 Experiment 1's "smallest battery wins" was a boundary artifact

At €6.00/kg the battery optimum is **interior, at 250 kWh**, for every array at
or above 700 kWp. Experiment 1 started at 500 kWh and so sat entirely on the far
side of it, seeing only the downslope.

The marginal arithmetic at 1100 kWp locates it exactly:

| step | operating gain | capital cost | net |
| --- | ---: | ---: | ---: |
| 100 → 250 kWh | +€25/day | +€18/day | **+€7** |
| 250 → 500 kWh | +€28/day | +€29/day | −€1 |
| 500 → 1000 kWh | +€51/day | +€59/day | −€8 |
| 1000 → 1500 kWh | +€21/day | +€60/day | −€39 |

The marginal value of storage crosses its marginal cost between 250 and 500 kWh.
That is a real optimum, not a grid edge, and no one-dimensional sweep starting at
500 kWh could have found it.

**This does not hold at low price.** At €3.00 and €4.50 the first step already
loses money (+€12/day operating against +€18/day capital at 1100 kWp), so the
optimum is at 100 kWh — still at the grid edge, still not located. Storage earns
its capital only when the product is valuable enough to be worth time-shifting.

### 3.2 The array decides whether the battery has anything to do

This is the interaction, and it is sharp. `joint_sizing_cycling.png` shows a
discontinuity between the 700 and 1100 kWp rows that no economic quantity
reveals on its own.

At 300 kWp a 1500 kWh pack moves **28 kWh/day** — under 2 % of its capacity. The
array is so undersized relative to the process chain that the plant is
intake-limited during daylight and there is simply no surplus to store; the pack
is dead weight carrying full capital. At 1100 kWp and above the same pack moves
**1127 kWh/day**, 75 % of capacity, because the array finally produces more than
the plant can consume in real time.

The value of storage follows exactly:

| array | operating gain, 100 → 1500 kWh |
| ---: | ---: |
| 300 kWp | +€10/day |
| 700 kWp | +€36/day |
| 1100 kWp | +€126/day |
| 1500 kWp | +€126/day |
| 1900 kWp | +€120/day |

It rises twelvefold from 300 to 1100 kWp, then saturates. **Storage and array are
complements below 1100 kWp and independent above it.** That is precisely the
structure a one-dimensional sweep is blind to, and it is why experiment 1's
battery conclusion — measured at a fixed 1100 kWp — happened to be measured on
the right side of the discontinuity and so was not wrong, only lucky.

### 3.3 The joint optimum is a shallow ridge, not a point

At €6.00/kg the three best cells are within €6/day of each other:

| cell | net | payback |
| --- | ---: | ---: |
| 700 kWp / 250 kWh | +€127 | 12.3 yr |
| 700 kWp / 100 kWh | +€126 | 12.4 yr |
| 1100 kWp / 250 kWh | +€122 | 13.8 yr |
| 1100 kWp / 500 kWh | +€121 | 13.3 yr |

€6/day is well inside the model's error bars — the plan-versus-realised bias
alone is −0.8 % to −5.8 %, which at these operating profits is €5–40/day. **The
grid cannot separate these cells and should not be read as doing so.** The
honest statement is a ridge running from (700, 250) to (1100, 500).

### 3.4 Within that ridge, 1100 kWp is the more defensible choice

Not because it wins on net profit — it loses by €5/day — but because it is far
more forgiving of a battery-sizing mistake. The spread of net profitability
across the whole battery axis:

| array | spread, €/day |
| ---: | ---: |
| 300 kWp | 134 |
| 700 kWp | 123 |
| **1100 kWp** | **48** |
| 1500 kWp | 47 |
| 1900 kWp | 55 |

At 700 kWp, getting the battery wrong by sizing it at 1500 kWh instead of 250
costs €124/day and wipes out the entire advantage of the plant. At 1100 kWp the
same mistake costs €48/day. Given that the €5/day preference for 700 kWp is not
resolvable at this grid's fidelity and the robustness difference is a factor of
2.6, **the reference sizing of 1100 kWp survives this experiment** — as it did
experiment 2, and for an independent reason.

### 3.5 Net profit and payback rank the cells differently

At 1100 kWp, net profit prefers 250 kWh (+€122) while payback prefers 500 and
1000 kWh (13.3 yr against 13.8). And payback prefers 700 kWp considerably more
strongly than net profit does.

This is not an inconsistency. Payback puts capital at t = 0 and asks when
cumulative discounted cash turns positive, so it rewards a smaller plant that
recovers a smaller outlay sooner; annuitised net profit asks how much the plant
earns once running and is indifferent to timing. A smaller plant can repay faster
while earning less. **Which metric is right depends on whether capital is
constrained**, which is outside this model. Both are reported for that reason.

### 3.6 Joint re-sizing is worth about a year of payback

At 0.75 × process capex, the reference sizing (1100 kWp / 500 kWh) repays in
13.3 years and the best cell in 12.3. So **re-optimising both axes jointly buys
1.0 year.** Set against the capital-cost axis at the same site and price:

| change | payback |
| --- | ---: |
| reference sizing at 1.00 × process capex | 19.6 yr |
| → 0.75 × process capex | 13.3 yr (−6.3) |
| → jointly re-sized at 0.75 × | 12.3 yr (−1.0) |
| → 0.25 × process capex, reference sizing | 6.4 yr (−13.2 from 1.00 ×) |

Sizing is second-order against capital cost — a quarter of the leverage of the
single 0.75 × assumption, and a thirteenth of the full capex sweep. That ordering
is worth stating plainly, because three of the four experiments in this folder
sweep sizing.

---

## 4. What this does not show

Everything in `RELATIVE_PROFITABILITY.md` §4, plus:

- **One site.** Every finding here is Seville's. §3.2 in particular is a
  statement about where the array/process-chain crossover falls, and that
  crossover moves with resource — at London the same 1100 kWp array produces
  less surplus, so the discontinuity should sit at a larger array. Untested.
- **The process rating is still fixed.** Both swept axes are electrical; the
  calciner, contactor, electrolyser and reactor ratings are held at
  `REFERENCE_SIZING` throughout. §3.2 shows the array/process balance is the
  binding relationship at small arrays, so the *third* axis is arguably more
  informative than either of the two swept here. It is the obvious next
  experiment.
- **Grid resolution.** The battery optimum is bracketed between 250 and 500 kWh
  and located no more precisely than that. The ridge in §3.3 is unresolved by
  construction.
- **C/2 throughout.** Battery power scales with capacity, so the 100 kWh pack has
  50 kW of power against a plant drawing hundreds. Some of what reads as "a small
  pack is efficient" may be a power constraint rather than an energy one, and
  this grid cannot separate the two.
- **Nothing is closed-loop.** These are dispatch plans. Plan fidelity was
  measured at one point only (1100 kWp / 1500 kWh, Seville); it has not been
  established that the −0.8 % to −5.8 % bias is constant across a grid that spans
  a 6× range of array size and a 15× range of pack size. §3.3 rests on that
  assumption more heavily than any other claim here.

---

## 5. Reproducing

```
python experiments/joint_sizing.py --check          # ~83 min
python experiments/joint_sizing.py --quick          # summer only, ~20 min
python experiments/joint_sizing.py --process-capex 1.0
```

`--check` re-runs the four cross-checks in §1 against `results/pv_sizing.csv` and
`results/relative_profitability.csv`, and is skipped under `--quick`.

Cell cost ranged from 3.7 s at 300 kWp to 16.6 s at 1900 kWp — the large-array
cells curtail heavily, which widens the MILP's feasible region and gives branch
and bound more work, the same effect noted in `PV_SIZING.md` §4.
