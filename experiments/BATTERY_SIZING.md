# Experiment A — Closed-Loop Battery Sizing

**Run:** 17 September 2026 · `experiments/battery_sizing.py`
**Data:** `results/battery_sizing.csv` (40 cells) · `results/series/` (10 trajectories)
**Figures:** `figures/A1`–`A7`

Five battery sizes × two controllers × four seasonal windows × **7 days
closed-loop**, at Seville, 1100 kWp, controller price fixed at €5.50/kg, seed 0.
5.4 hours of simulation. Inner-solver health across all 40 cells: `nmpc_used`
99.21 % minimum, 99.96 % mean, 0.79 % worst-case failure rate.

---

## 1. Why this replaces the plan-based grids

The earlier sizing studies evaluated **240 h dispatch plans** rather than
simulations, justified by a measured plan-versus-realised bias of −0.8 % at
three days. Two things destroyed that licence.

**The bias was re-measured after the battery economics were corrected** and came
back at −34 % at three days, not −0.8 %.

**And it is not common-mode along the swept axis** — which is fatal, because a
uniform bias cancels in an ordering and a varying one tilts the surface. This
experiment measures it directly:

| pack | planned €/day | realised €/day | bias |
| ---: | ---: | ---: | ---: |
| 100 | 583 | 176 | **−70.5 %** |
| 250 | 611 | 251 | −59.1 % |
| 500 | 623 | 311 | −52.3 % |
| 1000 | 548 | 389 | −34.5 % |
| 1500 | 579 | 446 | **−27.8 %** |

A **43-point spread**. The plan-based grids were not measuring a noisy version
of the right answer; they were measuring a different shape.

## 2. Method

| | |
| --- | --- |
| axis | 100, 250, 500, 1000, 1500 kWh, power at C/2 |
| controllers | `dispatch` (outer LP alone), `dispatch-nmpc` (LP + safety filter) |
| fixed | 1100 kWp, Seville, seed 0, forecast skill 0.75 |
| windows | 4 seasons × 7 days, `dt` 60 s, control interval 300 s |

**The controller's methane price is fixed at €5.50/kg, not swept.**
`methane_price_per_kg` is a control tuning parameter, not a market price. The
old grids set it to the swept price, so every cell changed two things at once —
what the plant is worth *and* how the controller behaves. Holding it fixed makes
production independent of the market price, so §3.5 is exact arithmetic over one
set of runs rather than a second sweep.

---

## 3. Results

### 3.1 The outer layer is blind to what storage is for

`A1_plan_vs_realised.png`

**The LP's planned profit is flat across a 15× range of battery capacity**
(583 → 611 → 623 → 548 → 579 €/day). Realised profit rises monotonically and is
**2.5× higher at 1500 kWh than at 100 kWh** (176 → 446 €/day).

So the outer layer believes storage barely matters, and in the plant it is the
dominant factor. That is why the plan-based grids concluded "build the smallest
pack": the LP saw near-constant profit against monotonically rising capital, and
the arithmetic did the rest.

The mechanism is time resolution. The dispatch layer optimises on an **hourly**
grid; the buffering that storage provides is a **sub-hourly** phenomenon. An
hourly average of supply and demand simply does not contain the deficits the
battery exists to cover, so the LP cannot price it.

### 3.2 What the safety layer buys

`A2_safety.png`

| pack | methane | bus trips | shed | LCOM |
| ---: | ---: | --- | ---: | --- |
| 100 | **+17.0 %** | 4764 → 3474 | −55.4 % | 10.34 → 8.95 |
| 250 | **+17.1 %** | 4336 → 1634 | −56.4 % | 9.09 → 8.01 |
| 500 | +4.2 % | 3076 → 1137 | −46.6 % | 8.21 → 7.75 |
| 1000 | +2.3 % | 1186 → 356 | −39.9 % | 8.26 → 7.68 |
| 1500 | +4.6 % | 1041 → 212 | −42.6 % | 8.58 → 7.83 |

The inner layer improves **every metric at every size**. This is stronger than
the architecture required: the expected trade was production surrendered to buy
constraint satisfaction, and an earlier formulation did pay it (−2.8 % methane
for a ninefold trip reduction). A plant that trips less does not lose the
production a trip costs.

**The `soc_min` column is the mechanism.** With a 100 kWh pack, `dispatch` never
discharges below **0.33** — while tripping the bus 4764 times. It sits on stored
energy and lets the interlock shed load instead. The filter reaches the **0.10**
floor at every size.

So the outer layer does not merely over-commit a small store, it **under-uses**
it. At hourly resolution it never sees the deficits that would justify
discharging, so the energy stays in the pack and the interlock sheds load
instead. The inner layer is extracting value the outer layer leaves on the
table — that is a stronger claim for a safety filter than constraint
satisfaction alone.

### 3.3 Why the filter moves the plan

`A3_attribution.png`

Two independent signals per solve: dual mass on each inequality block (what
shapes the optimum) and a counterfactual rollout of the plan's own action (what
it would have done unchecked).

Share of dual mass:

| pack | plan band | slew limit | bus balance | state box |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 62.7 % | 27.7 % | 9.6 % | ~0 |
| 250 | 87.6 % | 11.9 % | 0.5 % | ~0 |
| 500 | 91.6 % | 8.3 % | 0.1 % | ~0 |
| 1000 | 93.0 % | 7.0 % | ~0 | ~0 |
| 1500 | 92.0 % | 8.0 % | ~0 | ~0 |

**The dominant constraint is the plan's own setpoint band, not the bus.** This
contradicted the prediction made when the experiment was designed (bus balance
dominant, SoC floor growing as the pack shrinks) — the SoC floor never appears
at all, and the bus matters only at 100 kWh.

It also promotes a known defect to first place. The band has **no width**:
`Band.setpoint_max` is set to exactly the dispatch the LP intended, so the
reference the filter aims at *is* the ceiling it is penalised for exceeding.
Every upward correction hits a constraint immediately. `Band.dispatch` — the
separate field carrying what the plan actually intended — is published and read
by nothing. Fixing that (a margin above the intended dispatch) is now the
highest-value outstanding change to the architecture.

The two signals disagree in an informative way:

| pack | edit $\|u_0-\bar u_0\|^2$ | plan's bus deficit |
| ---: | ---: | ---: |
| 100 | 0.165 | 29.6 kW |
| 1500 | 0.015 | 24.7 kW |

**How far the plan overreaches is nearly constant; what changes is how expensive
it is to fix.** An 11× swing in edit effort against a 1.2× swing in the
overreach itself. That ratio is the value of storage, stated in control terms.

### 3.4 Commitment: the plant does shut down, and storage decides the pattern

`A6_commitment.png`

**Shutdown is mechanically real.** Power drawn while *decommitted* is
6.8 × 10⁻⁸ kW at worst — zero to numerical precision, the residue being the
enable gate's own floor. That is the regression check on an earlier failure in
which four shut-down machines drew 26.4 kW round the clock, 317 kWh over a
twelve-hour night.

**But idle load is not zero, and it is large.** A *committed* machine held at
zero setpoint still draws its idle power, and that is where the energy goes:

| pack | contactor | calciner | electrolyser | Sabatier | total/day |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1.4 | 91.9 | 167.7 | 29.0 | 41 kWh |
| 250 | 2.0 | 157.4 | 368.1 | 153.7 | 97 kWh |
| 500 | 1.0 | 105.1 | **973.7** | 276.9 | **194 kWh** |
| 1000 | 3.9 | 56.3 | 560.1 | 397.6 | 145 kWh |
| 1500 | 5.7 | 57.8 | 336.2 | 314.5 | 102 kWh |

(kWh per 7 days, `dispatch-nmpc`.) At 500 kWh that is **194 kWh/day spent by
machines producing nothing — 39 % of the pack's entire capacity, daily.**

And the filter is *worse* than the LP alone at 250 and 500 kWh (368 vs 248, and
974 vs 567 kWh on the electrolyser). The cause is architectural rather than a
tuning error: **the filter can drive a setpoint to zero but cannot decommit.**
Commitment is the outer layer's decision, pinned as a bound by `_bind_bands`, so
when the inner layer needs to shed load its only lever is downward setpoint
pressure — and idle draw is a floor it cannot get under. That is the same root
as the zero-width band in §3.3: every tool the filter has pushes in one
direction and stops at a wall the outer layer placed.

This is the clearest efficiency defect the experiment found, and it is not
visible in a plan-based study at all, because the LP charges itself the idle
term and then schedules around it.

**Storage decides whether this is a daylight plant or a continuous one.**

| | 100 kWh | 1500 kWh |
| --- | ---: | ---: |
| calciner duty | 0.54 | 0.94 |
| calciner **night** duty | 0.20 | 0.90 |
| calciner starts/day | **8.8** | **0.64** |
| calciner mean uptime | 1.6 h | 104 h |
| Sabatier night duty | 0.27 | 0.90 |
| Sabatier starts/day | 3.9 | 0.39 |
| Sabatier mean uptime | 3.9 h | 128 h |

At 100 kWh the kiln **thermal-cycles almost nine times a day** on 1.6-hour runs.
At 1500 kWh it starts twice in three days and runs for four days at a stretch.
Given a €40 start charge attributed to refractory damage, and a 5.6 h warm-up
from cold, that difference is a durability result as much as an economic one —
and it is invisible in any plan-based study, because the LP's minimum-up-time
constraint makes the schedule *look* well-behaved.

**The project's central thesis survives, but conditionally.** The premise was
that the plant's answer to nightfall should be chemical rather than electrical.
Night-time production share goes **0.00 → 0.42** for the LP alone and
**0.13 → 0.43** with the filter. So night operation is real — but it is *bought
with the battery*, not delivered by the architecture. At 100 kWh the LP alone
achieves **zero** night-time production; the filter gets 13 % from the same
hardware.

### 3.5 Corrected economics

`A5_economics.png` — net €/day, controller fixed, market price varied
arithmetically.

| pack | €3.00 | €5.50 | €8.00 |
| ---: | ---: | ---: | ---: |
| 100 | −409 | −223 | −38 |
| 250 | −385 | −166 | +53 |
| 500 | −392 | −155 | +82 |
| 1000 | −415 | −148 | +120 |
| 1500 | −445 | −155 | **+136** |

(`dispatch-nmpc`; `dispatch` is €10–40/day worse at every cell.)

**The optimum moves with price, and it moves the opposite way from the old
conclusion**: 250 kWh at €3.00, 1000 kWh at €5.50, 1500 kWh at €8.00. The old
grids said "smallest pack, at every price".

Nothing repays at €3.00 or €5.50. At €8.00 everything from 250 kWh up is
positive. The plant remains uncommercial at any plausible market price — that
conclusion from the earlier studies survives, and is now measured closed-loop
rather than planned.

---

## 4. What this does not show

- **Seasonal means hide almost everything about trips.** At 1000 kWh the
  `dispatch-nmpc` trip counts by season are **0 / 9 / 0 / 1417** (autumn /
  spring / summer / winter). The reported mean of 356 describes no season. Every
  trip figure in §3.2 is really a statement about winter.
- **The optimum at €8.00 sits at the grid edge** (1500 kWh) and is therefore not
  located — the same failure the old grids had, in the opposite direction.
- **One site, one array, one seed.** 1100 kWp at Seville. The array/battery
  interaction that the joint grid probed is untouched here, and the crossover
  between "daylight plant" and "continuous plant" must depend on it.
- **`plan_bias` for `dispatch-nmpc` is not plan fidelity.** With the filter in
  the loop the realised trajectory is not what the plan assumed, so that column
  measures how far the filter moved things. §1 and §3.1 use the `dispatch` rows
  only.
- **7 days is still short.** Plan bias fell with horizon in earlier measurement
  (−34 % at 3 days, −11.7 % at 7 on the reference plant), so these figures may
  still overstate divergence at a 25-year horizon.
- **No faults.** Every run is nominal. The brief requires fault response and
  none of it is exercised here.

## 4.1 Two defects this experiment promotes

Both were known and both were rated lower than they deserved. They share a root:
**every lever the inner layer has pushes downward against a wall the outer layer
placed**, so it can trim but not restructure.

1. **The dispatch band has no width.** `Band.setpoint_max` is the intended
   dispatch, so the filter's reference *is* its ceiling. 62–93 % of all dual
   mass sits on that constraint (§3.3). `Band.dispatch` already carries the
   intended value separately and is read by nothing, so the fix is a margin:
   publish a ceiling above the intent and track the intent.
2. **The filter cannot decommit.** Up to 194 kWh/day is drawn by committed
   machines at zero setpoint (§3.4), and the inner layer has no way to switch
   one off. Either commitment needs to be revisable below the hour, or the
   outer layer needs to decommit more aggressively when its own plan leaves a
   machine idle.

## 5. Reproducing

```
python experiments/battery_sizing.py                       # 40 cells, ~5.4 h
python experiments/battery_sizing.py --days 1 --seasons summer   # smoke test
python experiments/battery_sizing_figures.py               # seconds
```

The CSV is rewritten after every cell, so an interrupted sweep keeps everything
already computed. Full trajectories are stored for the summer windows only,
which is what `A4` and `A6` need.
