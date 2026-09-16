# PV Sizing Experiment

**Run:** 16 September 2026 · `experiments/pv_sizing.py`
**Data:** `results/pv_sizing.csv` · **Figures:** `figures/pv_sizing_{absolute,payback}.png`

A 5 × 5 grid of methane price against PV array size, at three European sites.
75 cells, 300 dispatch-LP solves, no failures. Battery held at 500 kWh.

Method, justification and caveats are those of
[`RELATIVE_PROFITABILITY.md`](RELATIVE_PROFITABILITY.md) sections 1 and 4 and
carry over unchanged: 240 h dispatch plans rather than closed-loop simulations,
four seasonal windows per cell, discounted payback at 7 % with battery
replacement bought when it falls due.

**The battery is held at 500 kWh, not the reference 1500.** Experiment 1 found
net profitability falling monotonically with capacity everywhere, so sweeping PV
at 1500 kWh would study a plant already known to be mis-sized. The two grids are
therefore *not* comparable cell-for-cell; they share only 1100 kWp, where the
difference is the battery and is worth €46–68/day by site.

---

## 1. Results

### Net profitability, EUR/day

**Seville**

| kWp \ €/kg | 2.00 | 3.00 | 4.00 | 5.00 | 6.00 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | −412 | −365 | −318 | −271 | −224 |
| 700 | **−404** | **−299** | **−194** | **−89** | +16 |
| 1100 | −469 | −345 | −221 | −97 | **+27** |
| 1500 | −547 | −417 | −288 | −158 | −29 |
| 1900 | −628 | −495 | −362 | −229 | −96 |

**London**

| kWp \ €/kg | 2.00 | 3.00 | 4.00 | 5.00 | 6.00 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | **−438** | −408 | −378 | −348 | −318 |
| 700 | −467 | **−400** | **−333** | −266 | −199 |
| 1100 | −523 | −435 | −346 | **−257** | **−168** |
| 1500 | −597 | −496 | −395 | −294 | −193 |
| 1900 | −674 | −566 | −457 | −349 | −241 |

**Gothenburg** tracks London within €10/day at almost every cell; the full table
is in the CSV.

### Time to break even, years

Only Seville repays, and only at €6.00/kg:

| kWp | payback |
| ---: | ---: |
| 700 | 20.7 yr |
| 1100 | **19.6 yr** |
| all others | never within 25 yr |

London and Gothenburg: ∞ in every cell.

### Production and capital

| kWp | Seville | London | Gothenburg | capital €/day | PV capex | total capex |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 47 kg/d | 30 | 31 | 498 | €210k | €1.51M |
| 700 | 105 | 67 | 70 | 586 | €490k | €1.79M |
| 1100 | 124 | 89 | 89 | 678 | €770k | €2.07M |
| 1500 | 129 | 101 | 97 | 767 | €1.05M | €2.35M |
| 1900 | 133 | 108 | 102 | 855 | €1.33M | €2.63M |

Process capex is €1,179,903 at every row — it does not scale with the array.

---

## 2. Findings

### 2.1 This axis has a real optimum; the battery axis did not

Experiment 1's battery grid was monotone with its optimum pinned at the smallest
size tested, so the grid could not locate it. PV is different: production
saturates while capital keeps climbing linearly, which puts a genuine interior
optimum inside the swept range at every site and every price.

That makes this the more useful of the two sizing studies, which is what the
capital shares suggested it would be.

### 2.2 The reference sizing of 1100 kWp is right — at the price we use

At €6.00/kg the optimum is 1100 kWp at all three sites, and that is where
`REFERENCE_SIZING` already sits. The sizing chosen at M3 on production grounds
(`03_SIZING.md`) turns out to be the economic optimum too, which is a useful
independent check on a decision that was made for different reasons.

### 2.3 But the optimal array is strongly price-dependent

| site | €2.00 | €3.00 | €4.00 | €5.00 | €6.00 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seville | 700 | 700 | 700 | 700 | 1100 |
| London | 300 | 700 | 700 | 1100 | 1100 |
| Gothenburg | 300 | 700 | 700 | 700 | 1100 |

The optimum moves by a factor of three or four across the price range, and it
moves further at the cloudier sites. That is the expected direction — a more
valuable product justifies more capital — but the magnitude is worth stating: a
sizing study run at one price is answering a much narrower question than it
appears to.

It also means **the two experiments' conclusions are not symmetric**. "Build the
smallest battery" held at every price tested *in that grid* — experiment 4 later
found an interior optimum at 250 kWh once the axis was extended below 500. "Build
1100 kWp" holds only at
€6.00, and at the market price of €1.80 the answer would be 300–700 kWp.

### 2.4 Production saturates because the plant is not PV-limited above ~1100 kWp

Seville gains 58 kg/day going 300 → 700 kWp, then 19 going 700 → 1100, then 4
and 4. Beyond about 1100 kWp the binding constraint is the reactor and the
calcium loop, not the array, so further capital buys curtailment.

The saturation is later and softer at the cloudy sites — London still gains 7
kg/day from 1500 to 1900 kWp where Seville gains 4 — because those sites spend
more of the year below the plant's intake capacity. A cloudy site wants a
*relatively* larger array, which is the opposite of the intuition that poor
resource means build less.

### 2.5 The plant still does not repay its investment

Two cells out of seventy-five, both Seville, both €6.00/kg, at 19.6 and 20.7
years against a 25-year project. Green methane with certificates is about
€1.80/kg — the leftmost column, where nothing is close.

As in experiment 1: this is not a controller result. It is what €1.5–2.6 M of
capital producing 30–133 kg/day implies at any price the market offers.

---

## 3. What this does not show

Everything in `RELATIVE_PROFITABILITY.md` section 4, plus:

- **The joint optimum.** PV was swept at a fixed 500 kWh battery and the battery
  was swept at a fixed 1100 kWp array. The two axes interact — a larger array
  makes storage more useful — so the true joint optimum is not either grid's
  best cell. **Settled by experiment 4**,
  [`JOINT_SIZING.md`](JOINT_SIZING.md): the interaction is real and sharp (a
  1500 kWh pack moves 28 kWh/day at 300 kWp against 1127 kWh/day at 1100 kWp),
  but the joint optimum is a shallow ridge from (700 kWp, 250 kWh) to
  (1100 kWp, 500 kWh) spanning €6/day, and 1100 kWp survives on robustness — it
  is 2.6× less sensitive to a battery-sizing error than 700 kWp.
- **Array size below 300 kWp**, where London and Gothenburg put their optimum at
  €2.00/kg. That optimum is at the grid edge and is not located.
- **Inverter and land costs** that scale differently from €/kWp, and any
  economy of scale in installation.
- **Tilt and orientation**, held at the site defaults throughout. A cloudier
  site's optimal tilt differs, and that interacts with 2.4.

---

## 4. Reproducing

```
python experiments/pv_sizing.py                 # full grid, ~90 min
python experiments/pv_sizing.py --quick         # summer only
python experiments/pv_sizing.py --sites Seville
```

Weather is PVGIS v5.2 TMY (SARAH2/ERA5), cached under `data/cache/`; all three
sites returned `provenance=measured`.

Runtime note: this grid took about 90 minutes against the battery grid's 45, at
the same cell count. The large-array cells curtail heavily, which widens the
MILP's feasible region and gives branch-and-bound more work. Cell cost is not
transferable between sweeps whose feasible regions differ.
