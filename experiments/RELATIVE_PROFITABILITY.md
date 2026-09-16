# Relative Profitability Experiment

**Run:** 16 September 2026 · `experiments/relative_profitability.py`
**Data:** `results/relative_profitability.csv` · **Figures:** `figures/relative_profitability_{absolute,relative,payback}.png`

A 5 × 5 grid of methane price against battery capacity, at three European sites
with different solar regimes. 75 cells, 300 dispatch-LP solves, no failures.

---

## 1. Method

Each cell is a **240-hour dispatch plan**, not a closed-loop simulation. Battery
capacity changes the plant; methane price changes the controller's objective.
Each cell averages four seasonal windows (days 15, 105, 172, 288), so a single
summer sample cannot flatter the high-latitude sites.

**Battery power scales with capacity at C/2**, so each pack is the same
technology at a different size.

### The headline metric is time to break even

Discounted payback at 7 %, with battery replacement bought when it falls due.

An earlier version of this experiment reported a *break-even price* instead.
That was the wrong metric twice over. Price is one of the two swept axes, so
solving for the price at which profit reaches zero collapses the axis the grid
exists to show; and the implementation used `np.interp`, which **clamps** when
the target lies outside the range, so two of three sites reported €6.00 — the
grid boundary — as though it were a result.

Payback is defined at every cell, takes price as given, and says how long rather
than merely whether. Net €/day is still reported, but it is a weaker statement
than it looks: it annuitises capital at the capital recovery factor, so a
positive figure means only "repays within 25 years at 7 %".

Three adjustments turn annuitised figures into a cash flow:

* capital is an outlay at *t* = 0, so the CRF comes off and capex enters whole;
* fixed opex stays annual;
* **the wear accrual is added back and replacements enter as lumps.** Operating
  profit already subtracts `cost_per_kWh_delivered` on every kWh moved, which
  pre-pays a replacement that has not happened. Leaving that in *and* charging
  the lump would bill the battery twice.

### Why plans rather than simulations

A 240 h plan costs about twelve seconds; the equivalent simulation costs tens of
minutes. That substitution is only legitimate if a plan predicts what the plant
does, which was measured first (`sfp/report/plan_tracking.py`):

| horizon | plan predicted | plant delivered | error | inventory divergence |
| --- | ---: | ---: | ---: | ---: |
| 3 days | €953.9 | €946.6 | −0.8 % | 6.6 % of capacity |
| 7 days | €2468.3 | €2325.8 | −5.8 % | 3.6 % of capacity |

Replan windows scatter, but the error averages out rather than accumulating, and
inventory divergence *falls* as the horizon lengthens.

### Battery economics

Corrected in three ways before this run, all of which move against the battery:

1. **Wear per kWh** now follows `C_deg = CAPEX/(Capacity × DoD × CycleLife ×
   RTE)`. The previous `CAPEX/CycleLife` treated a cycle as moving full
   nameplate with no losses, understating wear by 1/(0.80 × 0.9216) = **1.357**.
2. **Cycle life depends on depth**, `N(DoD) = N_ref(DoD_ref/DoD)^k` with k = 3,
   so a large pack cycled hard ages like a small one.
3. **Replacements are charged.** The capital annuity amortises the pack over the
   project's 25 years and it does not last them. Only the
   calendar-attributable share is added, because cycling is already funded
   through operating profit.

Each cell derives its own EFC/day and mean depth stress from its plan, so every
cell has its own battery life rather than a shared assumption.

---

## 2. Results

### Time to break even, years (∞ = not repaid within the 25-year project)

**Seville**

| kWh \ €/kg | 2.00 | 3.00 | 4.00 | 5.00 | 6.00 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | ∞ | ∞ | ∞ | ∞ | **19.6** |
| 1000 | ∞ | ∞ | ∞ | ∞ | **19.6** |
| 1500 | ∞ | ∞ | ∞ | ∞ | **21.5** |
| 2000 | ∞ | ∞ | ∞ | ∞ | ∞ |
| 2500 | ∞ | ∞ | ∞ | ∞ | ∞ |

**London and Gothenburg: ∞ in every cell.**

### Net profitability, EUR/day

**Seville**

| kWh \ €/kg | 2.00 | 3.00 | 4.00 | 5.00 | 6.00 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | −469 | −345 | −221 | −97 | **+27** |
| 1000 | −525 | −389 | −254 | −117 | **+18** |
| 1500 | −574 | −443 | −304 | −162 | −19 |
| 2000 | −630 | −486 | −346 | −170 | −56 |
| 2500 | −687 | −543 | −402 | −256 | −111 |

**London**

| kWh \ €/kg | 2.00 | 3.00 | 4.00 | 5.00 | 6.00 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | −523 | −435 | −346 | −257 | −168 |
| 1000 | −579 | −484 | −390 | −295 | −201 |
| 1500 | −635 | −536 | −438 | −339 | −240 |
| 2000 | −692 | −591 | −493 | −393 | −293 |
| 2500 | −749 | −648 | −549 | −448 | −347 |

**Gothenburg** is within €5/day of London at every cell (full table in the CSV).

### Capital and battery duty

| battery | first pack €/day | replacement €/day | life yr | EFC/day | DoD stress |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 658 | 17.5 | 8.3 | 0.62 | 1.23 |
| 1000 | 698 | 35.5 | 9.0 | 0.55 | 1.12 |
| 1500 | 738 | 52.9 | 9.5 | 0.48 | 1.04 |
| 2000 | 777 | 68.2 | 10.0 | 0.41 | 1.02 |
| 2500 | 817 | 84.6 | 10.6 | 0.33 | 0.98 |

### Capex breakdown

| | unit cost | quantity | total | share |
| --- | ---: | ---: | ---: | ---: |
| PV | €700/kWp | 1100 kWp | €770,000 | 30–37 % |
| battery | €250/kWh | 500–2500 kWh | €125k–625k | 6–24 % |
| process | €2,200/kW | 536.3 kW | €1,179,903 | 46–57 % |

Process rating: contactor 20.4 kW, calciner 153.0, electrolyser 329.9, sabatier
33.0. **The electrolyser alone is 62 % of the process chain**, and the chain is
the largest capital item — larger than the array. All three unit costs are
`provenance: assumed`.

---

## 3. Findings

### 3.1 The plant does not repay its investment

**Three cells out of seventy-five break even inside the project's life**, all at
Seville, all at €6.00/kg, all with packs of 1500 kWh or less, and all at 19.6 to
21.5 years against a 25-year horizon. Green methane with certificates is about
€1.80/kg, the left-hand column.

This is not a controller result and should not be reported as one. It is what
€2.1–2.6 M of capital producing 96–137 kg/day of methane implies at any price
the market offers. The control architecture is the contribution; the economics
of the reference sizing are what they are, and the write-up is better for saying
so plainly.

### 3.2 The battery is a cost at every size tested

Net profitability falls monotonically with capacity at every site and every
price. The optimum is at **500 kWh — the smallest pack tested — everywhere**,
so the true optimum lies at or below it and this grid cannot locate it.

> **Superseded in part by experiment 4.** [`JOINT_SIZING.md`](JOINT_SIZING.md)
> §3.1 re-ran the axis from 100 kWh at Seville and found the optimum is
> **interior, at 250 kWh**, at €6.00/kg — the monotone decline seen here is the
> downslope of a peak that sits below this grid's floor. The marginal value of
> storage crosses its marginal cost between 250 and 500 kWh. At €3.00 and
> €4.50/kg the optimum does remain at the grid edge, so the conclusion below
> holds at the prices this experiment emphasised but not at €6.00.

This confirms the project's own thesis, reached independently.
`00_MASTER_PLAN.md` argued the battery is "the *least* interesting buffer" and
that the answer to nightfall "should mostly be chemical, not electrical". The
reference sizing of 1500 kWh is oversized on this evidence: Seville loses
€46/day against 500 kWh, London €67, Gothenburg €68.

It is also a stronger conclusion than the first version of this experiment
reported, because the battery economics were wrong then in three ways that all
favoured storage.

### 3.3 Small packs work harder and die sooner — and it still isn't enough

The depth-dependent life model earns its place here. A 500 kWh pack runs at
0.62 EFC/day with a depth stress of 1.23 and lasts **8.3 years**; a 2500 kWh
pack runs at 0.33 EFC/day, stress 0.98, and lasts **10.6 years**. The small pack
needs more replacements per decade, exactly as expected.

That penalty is real and it is still nowhere near enough to reverse the ranking:
the replacement charge spans €17.5 to €84.6/day across the range, moving the
*wrong* way for large packs because replacement scales with capex faster than
life extends.

### 3.4 Consistency of sunlight beats quantity of it

**Gothenburg receives more annual sun than London — 3868 against 3683 kWh/day —
and is worth marginally less at every cell.**

| | winter | spring | summer | autumn | mean | seasonal ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seville | 4306 | 7022 | 6996 | 5259 | 5896 | 1.4× |
| London | 1668 | 5863 | 5314 | 1886 | 3683 | 2.9× |
| Gothenburg | 913 | 7013 | 5490 | 2056 | 3868 | **6.3×** |

Gothenburg's spring matches Seville's best and its winter is nearly dead. The
plant's buffers span hours to days — battery minutes, hydrogen a day, the
calcium loop a few days — so **spring sunshine cannot be banked until
December.** No storage medium in this design bridges a season.

For the siting study: annual irradiance is the wrong figure of merit. Seasonal
evenness is.

### 3.5 Price dominates sizing by roughly three to one

The price axis spans about €496/day at Seville and €355/day elsewhere; the
battery axis spans €138–226/day. The slope in the price direction is exactly
production — d(net)/d(price) = kg/day — recovering **137 kg/day** at Seville
against **129.5** measured in closed-loop simulation, about 6 % apart and in the
same direction as the known plan-fidelity bias.

---

## 4. What this does not show

- **Closed-loop validation across the swept axes.** Plan fidelity was measured at
  one point (Seville, 1500 kWh, €3.00). The bias is assumed constant across
  price and battery size and that has not been checked. Two or three cells
  should be spot-checked with full simulations.
- **The true battery optimum**, which lies at or below the grid's lower edge.
- **Anything about the inner NMPC.** These are dispatch-LP plans only.
- **PV sizing**, held at 1100 kWp throughout. Since PV is a larger capital item
  than the battery and the binding resource at two of three sites, a price × PV
  grid is likely more informative than price × battery, and is the natural next
  experiment.
- **Sensitivity to `dod_exponent`**, which is `assumed` at k = 3. Anything from 2
  to 4 is defensible; a conclusion that flips inside that range should be
  reported as undetermined. The battery ranking here is monotone and wide, so it
  is unlikely to flip, but this has not been swept.
- **Salvage value** at the end of the project, and **learning-rate declines** in
  battery or electrolyser capex, both of which would improve the picture.

---

## 5. Reproducing

```
python experiments/relative_profitability.py                 # full grid, ~45 min
python experiments/relative_profitability.py --quick         # summer only
python experiments/relative_profitability.py --sites Seville
```

Weather is PVGIS v5.2 TMY (SARAH2/ERA5), cached under `data/cache/`. All three
sites returned `provenance=measured`; no synthetic weather was used.
