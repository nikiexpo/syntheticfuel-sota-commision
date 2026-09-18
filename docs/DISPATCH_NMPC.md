# The `dispatch-nmpc` architecture

**Scope.** Formulations only — what problem each layer solves, what passes
between them, and where the present implementation departs from the intended
design. Subsystem models are not reproduced here; they are in
`docs/ASSUMPTIONS.md` and the model modules.

**Status.** §1–§2 describe the code as it stands. §3 and §4 are kept as the
record of the *tracking* formulation and why it was wrong — that mode still
exists and `HierarchicalController` still uses the economic one. §5 is the
safety filter, which is built and is now the default for `dispatch-nmpc`. §6 is
the change log and the performance story; §7 the measured behaviour.

Nothing in `bookkeeping/` was used as a source.

---

## 1. The closed loop

Three clocks.

```
  1 h   outer   LP/MILP economic dispatch, 240 h horizon, re-solved every 3 h
 10 min inner   safety filter on the full 16-state model, 5 min horizon, dt 60 s
  5 min plant   controller.act() -> Request
  1 min bus     bus.reconcile() -> interlock, shedding, trips; plant integrates
```

The inner layer re-solves every 600 s (`resolve_interval_s`) and **holds** its
answer across the 5-minute control steps inside that interval;
`DispatchNMPCController` tracks this with `nmpc_held`. Note that its solve
cadence and its own discretisation are now different numbers — 600 s apart,
integrated at 60 s — which is deliberate (§5.1) and has one consequence for the
rate limit recorded there.

Information flowing **down**, per interval $k$ — the complete interface, one
`Band` per machine plus three arrays:

| symbol | type | meaning |
| --- | --- | --- |
| $c_{j,k}\in\{0,1\}$ | `Band.committed` | machine $j$ energised over interval $k$ |
| $\bar s_{j,k}$ | `Band.setpoint_max` | setpoint the plan intends, in plant units |
| $d_{j,k}$ | `Band.dispatch` | the raw LP variable (W, or mol/s for the reactor) — **published, never read** |
| $z^{\text{tgt}}_{N}$ | `plan.state_at(...)` | buffer levels the plan expects, for the four states in `TARGET_STATES` |
| ~~$\lambda_k$~~ | `lambda_EUR_per_kWh` | dual on the outer bus balance — **dead, see §1.1** |

Also published and unread: `plan.battery_W` and `plan.curtail`, the outer
layer's own battery and curtailment schedule. Together with `Band.dispatch`
these are the complete proposed action $\bar u_0$ that §5 requires.

### 1.1 The price coordination path is dead

The architecture was originally designed around **price coordination**: the
outer layer publishes $\lambda$, the inner layer solves a local problem at that
price. That path no longer exists.

`nmpc.py:597` binds `price = prices[k]` and never reads it again. Neither
objective branch uses it:

- **tracking** — no energy price by design;
- **economic** — uses the *static* `economics.p.methane_price_per_kg` and the
  battery wear cost, not $\lambda$.

The economic branch lost $\lambda$ deliberately. Three pricing variants were
tried and all were wrong: pricing process power double-counts against the
enforced bus balance and becomes a standing bias toward curtailment; pricing the
battery at the current $\lambda$ says charging is worthless at midday, exactly
when the battery should be filling, because $\lambda$ is zero when the plant is
already saturated and spilling. The resolution was to move all intertemporal
value into the terminal term. That reasoning is sound — but nothing removed the
plumbing afterwards.

**Consequence.** Recovering $\lambda$ from a MILP requires fixing the integer
columns and re-solving the continuous problem — a second LP solve on every
replan — for a number nothing consumes. `solve_lp(..., duals=False)` and
`EconomicDispatch(recover_duals=False)` are now the defaults; the price survives
as a reported diagnostic and in the tests that check the LP is well-formed.

The safety filter of §5 has no economics at all, so there is no route for
$\lambda$ to re-enter. If price coordination is ever wanted again it belongs in
a *third* mode, not in the filter.

Information flowing **up**: none. There is no feedback path from the inner layer
to the outer one. Band violations, shed predictions and solver failures are
recorded as diagnostics and discarded. This is a known gap (the $\zeta^-/\zeta^+$
feedback that was specified and never built).

**On failure** the inner layer returns the outer layer's own `Request`
unmodified. This fallback is silent by design and was responsible for an earlier
episode in which 846 of 864 solves failed while the run reported nothing:
`nmpc_used` is the only signal that separates "the inner layer agreed with the
plan" from "the inner layer was not consulted".

---

## 2. The outer problem

A MILP over $n = 240$ hourly intervals ($\Delta t = 3600$ s), re-solved every
3 h. **7447 columns, 6964 rows, 24 integers.** It carries all economics and no
subsystem dynamics beyond an algebraic power balance and the kiln's energy
balance — which is linear, and is the reason the kiln keeps a state.

### 2.1 States

$z_k \in \mathbb R^{7}$, one vector per interval boundary ($n+1 = 241$ of them,
1687 columns):

| $i$ | state | units | why it is here |
| --- | --- | --- | --- |
| 0 | $\text{soc}$ | — | electrical buffer |
| 1 | $n_{\text{CaCO}_3}$ | mol | the chemical buffer — the intermittency store |
| 2 | $\bar N$ | — | mean sorbent cycle number; carries irreversible deactivation |
| 3 | $n_{\text{H}_2}$ | mol | decouples the two electrical loads |
| 4 | $n_{\text{CO}_2}$ | mol | decouples capture from conversion |
| 5 | $m_{\text{H}_2\text{O}}$ | kg | binding at an arid site |
| 6 | $T_{\text{kiln}}$ | K | makes start-up expensive *in the model*, not just in the cost |

### 2.2 Controls

$u_k \in \mathbb R^{24}$ per interval (5760 columns):

| symbol | count | range | meaning |
| --- | ---: | --- | --- |
| $e_{j,k}$ | 4 | $[0,1]$ | commitment of machine $j$ |
| $\delta_{j,m,k}$ | 7 | $[0, w_{j,m}]$ | PWL segment fills — contactor 3, electrolyser 3, reactor 1 |
| $P^{\text{kiln}}_k$ | 1 | $[0, \bar P]$ | kiln heater power, W |
| $r^{\text{cal}}_k$ | 1 | $[0, \bar r]$ | calcination rate, mol/s |
| $y_k$ | 1 | $\{0,1\}$ for $k<24$, else $[0,1]$ | "hot enough to calcine" |
| $s_{j,k}$ | 4 | $[0,1]$ | start indicator |
| $P^{\text{ch}}_k, P^{\text{dis}}_k$ | 2 | $[0,\bar P_b]$ | battery |
| $\gamma_k$ | 1 | $[0,1]$ | curtailed fraction of available PV |
| $v^{\text{H}_2}_k, v^{\text{CO}_2}_k$ | 2 | $[0,1]$ | vent rates, mol/s |
| $\dot m^{w}_k$ | 1 | $[0,0.02]$ | makeup water, kg/s |

**Piecewise-linear rate maps.** For the contactor, electrolyser and reactor the
dispatch is the segment sum and the rate is the slope-weighted sum:

$$d_{j,k} = \sum_m \delta_{j,m,k}, \qquad r_{j,k} = \sum_m a_{j,m}\,\delta_{j,m,k}$$

with slopes $a_{j,m}$ **strictly decreasing** — falling marginal yield. That
concavity is what makes the relaxation exact: an optimiser maximising output
fills the steepest segment first of its own accord, so no ordering binaries are
needed. `PWLMap` raises if the sampled curve is not concave, because the
formulation is silently wrong otherwise rather than merely inaccurate.

Electrical draw is $P_{j,k} = \text{idle}_j\,e_{j,k} + d_{j,k}$ for the two
power-dispatched machines (contactor, electrolyser) and $\text{idle}_j e_{j,k}$
for the reactor, whose draw is flat in feed.

### 2.3 Dynamics — 7 equalities per interval

$$
\begin{aligned}
\text{soc}_{k+1} &= \text{soc}_k\Big(1 - \tfrac{\Delta t\,\sigma}{86400}\Big)
  + \frac{\Delta t\,\eta_c P^{\text{ch}}_k}{E} - \frac{\Delta t\,P^{\text{dis}}_k}{\eta_d E}\\[2pt]
n^{\text{CaCO}_3}_{k+1} &= n^{\text{CaCO}_3}_k + \Delta t\,(r^{\text{cont}}_k - r^{\text{cal}}_k)\\[2pt]
\bar N_{k+1} &= \bar N_k + \Delta t\,r^{\text{cal}}_k / n_{\text{tot}}\\[2pt]
n^{\text{H}_2}_{k+1} &= n^{\text{H}_2}_k + \Delta t\,(r^{\text{ely}}_k - 4 r^{\text{sab}}_k - v^{\text{H}_2}_k)\\[2pt]
n^{\text{CO}_2}_{k+1} &= n^{\text{CO}_2}_k + \Delta t\,(r^{\text{cal}}_k - r^{\text{sab}}_k - v^{\text{CO}_2}_k)\\[2pt]
m^{w}_{k+1} &= m^{w}_k + \Delta t\big(\dot m^w_k + 2\phi M_{\text{H}_2\text{O}} r^{\text{sab}}_k - M_{\text{H}_2\text{O}} r^{\text{ely}}_k\big)\\[2pt]
C\,\frac{T_{k+1}-T_k}{\Delta t} &= \eta P^{\text{kiln}}_k - UA\,(T_k - T^{\text{amb}}_k) - \Delta H\, r^{\text{cal}}_k
\end{aligned}
$$

The $4{:}1$ coefficient on $r^{\text{sab}}$ in the hydrogen balance is the
Sabatier stoichiometry, and it is what couples the two upstream chains.

Every term is linear in a decision variable, **including the kiln**. The only
genuinely nonlinear part was ever the calcination *rate* as a function of
temperature (Baker equilibrium × Arrhenius), and that enters as a bound in §2.4
rather than as a factor here.

### 2.4 Constraints

**Bus balance** — $n$ equalities, and the block whose dual is $\lambda$:

$$\sum_j P_{j,k} \;+\; w_{\text{comp}}\,r^{\text{cal}}_k \;+\; P^{\text{ch}}_k - P^{\text{dis}}_k \;=\; (1-\gamma_k)\,\text{PV}_k$$

Written demand-minus-supply with the PV term moved right, so relaxing it means
energy arriving from nowhere and the dual is a positive price.

**Segment cap** — $\delta_{j,m,k} \le w_{j,m}\,e_{j,k}$. With the objective this
*is* the dispatch band: a segment can only be used by a committed machine, and
only to its own width.

**Kiln band** — five rows, and the only place integers are needed:

$$
\begin{aligned}
\underline P\,e_k \;\le\;& P^{\text{kiln}}_k \;\le\; \bar P\,e_k\\
& r^{\text{cal}}_k \le \bar r\,y_k\\
& r^{\text{cal}}_k \le \alpha\,(T_k - T^{\text{on}}) + M\,(1-y_k)\\
& y_k \le e_k
\end{aligned}
$$

with $M = \alpha(T^{\text{on}} - 250)$, the smallest value letting a cold kiln
sit feasibly at $r = 0$. The set being described,
$0 \le r \le \max(0, \alpha(T - T^{\text{on}}))$, **is genuinely non-convex** —
writing the cap directly as $r \le \alpha(T-T^{\text{on}})$ rearranges to
$T \ge T^{\text{on}} + r/\alpha$, which at $r=0$ demands a permanently hot kiln.
Relaxing $y$ to $[0,1]$ permits $r^\star = \bar r\,(\alpha(T-T^{\text{on}})+M)/(\bar r + M)$,
which at 841 K is 0.26 mol/s where the real kiln calcines nothing. Hence binary,
but only over `binary_horizon_hours = 24` — beyond that the relaxation is
harmless because only the first three hours are ever implemented.

**Reactor availability** — the taper the plant applies as either buffer runs down:

$$r^{\text{sab}}_k \le \bar r_{\text{sab}}\,\frac{n^{\text{CO}_2}_k - \underline n_{\text{CO}_2}}{50},
\qquad
r^{\text{sab}}_k \le \bar r_{\text{sab}}\,\frac{n^{\text{H}_2}_k - \underline n_{\text{H}_2}}{200}$$

An honest linear envelope of the plant's $\min(\cdot,\cdot)$, needing no
indicator because the taper reaches zero exactly *at* the box floor, so
$r^{\text{sab}}=0$ stays feasible everywhere.

**Minimum load** — $\sum_m \delta_{\text{ely},m,k} \ge \underline d\,e_{\text{ely},k}$ (gas crossover).

**Start counter** — $s_{j,k} \ge e_{j,k} - e_{j,k-1}$, with interval 0 measured
against `_prev_commit` rather than zero.

**Minimum up-time** — $\tau_j s_{j,k} \le \sum_{l=k}^{k+\tau_j-1} e_{j,l}$, with
$\tau = 3$ for the calciner and $2$ for the reactor.

### 2.5 Objective

$$
\min_{z,u}\;\; \sum_{k=0}^{n-1}\Big[
-\,p_{\text{CH}_4} M_{\text{CH}_4}\,\Delta t\, r^{\text{sab}}_k
+ \frac{c_{\text{sorb}}\Delta t\, r^{\text{cal}}_k}{n_{\text{tot}}}
+ \frac{c_{\text{batt}}\Delta t}{2E}\big(P^{\text{ch}}_k + P^{\text{dis}}_k\big)
+ c_w \Delta t\,\dot m^w_k
+ \sum_j c^{\text{start}}_j s_{j,k}\Big] \;-\; V(z_n)
$$

Start costs are €0.50 / €40 / €5 / €40 for contactor / calciner / electrolyser /
reactor. $c_{\text{batt}}$ is `cost_per_efc_EUR()`, the term whose correction
invalidated the old price tuning (§7).

**Terminal value**, with $f =$ `terminal_value_fraction`, everything priced at
the methane it would become:

$$V(z_n) = p_{\text{CH}_4}M_{\text{CH}_4} f\Big[n^{\text{CaCO}_3}_n + \tfrac{1}{4}n^{\text{H}_2}_n + n^{\text{CO}_2}_n + \tfrac{\text{soc}_n E}{4\cdot 56.4\cdot 2.016{\times}10^{-3}\cdot 3.6{\times}10^{6}}\Big]$$

Without it the layer empties every buffer at the horizon end, which is locally
optimal and globally wrong.

### 2.6 Row census at 240 h

| block | rows |
| --- | ---: |
| `initial_state` | 7 |
| `dynamics` | 1680 |
| `bus_balance` | 240 |
| `segment_cap` | 1680 |
| `kiln_band` | 1200 |
| `reactor_availability` | 480 |
| `min_load` | 240 |
| `start_counter` | 960 |
| `min_uptime` | 477 |
| **total** | **6964** |

$\lambda_k$ can be recovered by fixing the integers and re-solving the LP,
because a MILP has no duals of its own. This is now **off by default**
(`recover_duals=False`) — see §1.1.

### 2.7 The property that matters downstream

**The outer model is hourly.** Any buffering requirement that lives below one
hour is invisible to it, and it has no representation of actuator slew at all.
Both are delegated entirely to the inner layer, which is why that layer cannot
be a pass-through.

---

## 3. The tracking formulation (superseded)

Kept as the record of what §4 diagnoses, and still reachable as
`objective="tracking"`. The filter of §5 replaced it as the default.

Horizon $T = 3600$ s, $\Delta t = 600$ s, so $N = 6$ intervals. State
$x\in\mathbb R^{16}$, control $u\in\mathbb R^{11}$ — a
$(\text{setpoint}, \text{enable})$ pair for each of the four process machines,
$(P^{\text{ch}}, P^{\text{dis}})$ for the battery, and one curtailment fraction.

### Decision variables

$$
x_{0:N}\in\mathbb R^{16(N+1)},\qquad
u_{0:N-1}\in\mathbb R^{11N},\qquad
\sigma_k \in [0,\,10],\qquad
\omega_{k,j}\in[0,\,1]
$$

$\sigma$ is predicted load shed (units of the power scale), $\omega$ the
excursion above a soft band ceiling, one per capped column $j\in\mathcal C$.
All variables are scaled to their own bounds.

### Constraints

$$
\begin{aligned}
&x_0 = \hat x && \text{(hard equality)}\\
&x_{k+1} = f(x_k, u_k, w_k) && \text{(hard equality)}\\
&g_{\text{bus}}(x_k,u_k)/P_{\text{scale}} - \sigma_k = 0 && \text{(hard equality, slacked)}\\
&u_{k,j} - \bar s_j - \omega_{k,j} \le 0,\quad j\in\mathcal C && \text{(soft ceiling)}\\
&|u_{k,j} - u_{k-1,j}| \le \rho = 0.34,\quad j\in\mathcal M && \textbf{(hard)}\\
&x^{\text{lo}} \le x_k \le x^{\text{hi}} && \textbf{(hard box)}\\
&u^{\text{lo}} \le u_k \le u^{\text{hi}} && \text{(hard box)}
\end{aligned}
$$

Commitment enters as a **bound, not a variable**: $c_{j,k}=1$ pins the enable
column to 1, $c_{j,k}=0$ pins both enable and setpoint to 0. The enable is never
a free variable, which removes four near-discontinuous gates per interval.

$\mathcal M$ (rate-limited columns) is every column that is not an enable and is
not pinned. $\mathcal C$ (capped columns) is the setpoint column of every
committed machine.

### Objective (tracking mode)

$$
\max_{x,u,\sigma,\omega}\;
-\underbrace{w_u\sum_{k=0}^{N-1}\sum_{j\in\mathcal T}\big(u_{k,j}-\bar s_j\big)^2}_{\text{setpoint tracking}}
-\underbrace{w_\omega\sum_{k,j}\omega_{k,j}}_{\text{band excursion}}
-\underbrace{w_\sigma\sum_k \sigma_k}_{\text{shed}}
-\underbrace{w_T\sum_{i\in\mathcal I}\Big(\tfrac{x_{N,i}-z^{\text{tgt}}_i}{\chi_i}\Big)^2}_{\text{terminal inventory}}
$$

with $w_u = 1$, $w_\omega = 10^2$, $w_\sigma = 10^4$, $w_T = 100$.

$\mathcal T$ is the setpoint column of each committed machine. **Battery and
curtailment are deliberately untracked** — they are the balancing degrees of
freedom that close the bus against weather the outer layer forecast an hour ago.

Only $u_0$ is applied; the rest of the horizon is discarded.

---

## 4. Why §3 was not a safety filter

Every defect below is fixed in §5; this section is the diagnosis that produced
that design, kept because the reasoning is what justifies the choices.

A safety filter answers one question: *what is the smallest edit to the proposed
action that keeps the plant feasible?* Six properties of the formulation above
are inconsistent with that, in rough order of severity.

### 4.1 The slack variables are bounded, so "soft" constraints are still hard

$$\sigma_k \in [0,\,10], \qquad \omega_{k,j}\in[0,\,1]$$

A slack with a finite upper bound does not soften a constraint — it relocates
the infeasibility. If the required band excursion exceeds one scaled unit, or
the required shed exceeds ten power-scale units, the problem is infeasible
exactly as it was before the slack was added.

This is the most likely cause of the 22 `Infeasible_Problem_Detected` and 3
`Restoration_Failed` results in the 3-day run. A true soft constraint has
$\sigma,\omega \in [0,\infty)$.

### 4.2 The state box is hard

$x^{\text{lo}} \le x_k \le x^{\text{hi}}$ is imposed as a hard box on every
state at every step, on top of a hard initial-state equality and hard dynamics.
If the current state plus the model implies a bound crossing that **no
admissible control can prevent**, the problem is infeasible and the filter
returns nothing — precisely when a safety filter is most needed.

The SoC floor at night is the obvious instance: committed machines draw
mandatory idle load, and the only remaining freedom is a battery that is already
at its floor.

A safety filter must never be infeasible. State constraints belong in the
objective with a large penalty, not in the feasible set.

### 4.3 The rate limit is hard, and applied to actuators that have no slew limit

$|u_{k,j}-u_{k-1,j}|\le 0.34$ is applied to **every** movable non-enable column
— which includes the battery's charge and discharge columns and the PV
curtailment fraction.

Neither has a physical slew limit worth modelling. A battery converter responds
in milliseconds; curtailment is a firing-angle change. Rate-limiting them to 34 %
of span per 10-minute interval is an artificial constraint on **the two channels
whose entire purpose is to close the bus balance in real time.**

This is self-inflicted and probably the second-largest source of trouble: the
bus balance needs a shed slack largely because the fast channels that would
otherwise absorb the imbalance have been slew-limited into uselessness.

The rate limit is meaningful for the four process setpoints, which stand in for
valves and heaters the plant model does not represent. It should be restricted
to those columns, and even there it should be soft.

### 4.4 The terminal economic term dominates the tracking term by 100×

$w_T = 100$ against $w_u = 1$. The terminal inventory penalty is applied in
tracking mode (`_terminal_value` returns a non-zero value whenever `targets` is
supplied, regardless of mode).

So the inner layer is not minimally editing the plan — it is **re-optimising the
plant toward inventory targets at a hundred times the weight it places on
following the plan's setpoints.** That is a second economic optimiser, running
on a one-hour horizon, overriding decisions the 240-hour layer already made with
far better information.

The docstring's justification — that without a terminal term the layer drains
every buffer inside its own hour — is a real problem, but it is a symptom of
tracking the whole horizon rather than filtering the first action. A filter that
penalises only $u_0$ has no incentive to drain anything, because it is not
optimising the tail at all.

### 4.5 The tracking target is the constraint boundary

`Band.setpoint` returns `setpoint_max`, and `setpoint_max` is set to exactly the
setpoint the LP intends (`dispatch.py:660`). So $\bar s_j$ is simultaneously:

- the value the objective asks the NMPC to hit, and
- the ceiling above which the band penalty $w_\omega$ applies.

The reference sits **on** the constraint. Any positive tracking error is
immediately penalised at $10^2$; any negative error at $1$. The "band" has no
width — it is a point target with a one-sided penalty, not the power band the
architecture called for.

`Band.dispatch`, documented as "what the plan itself intended", is carried
through the interface and **never read by anything**.

### 4.6 The tracking sum runs over the whole horizon

$\sum_{k=0}^{N-1}$ penalises deviation at every step, which makes this a tracking
MPC. A filter penalises $\|u_0 - \bar u_0\|$ only, and uses steps $1{:}N$ purely
to certify that a feasible continuation exists.

---

## 5. The safety filter, as built

**Implemented** as `objective="filter"`, and the default for `dispatch-nmpc`.
Same variables, same dynamics, same commitment-as-a-bound treatment. Four
changes from §3: unbounded slacks, soft state constraints, first-action-only
objective, and no economics.

### Constraints

$$
\begin{aligned}
&x_0 = \hat x\\
&x_{k+1} = f(x_k,u_k,w_k) && \text{hard — this is the model}\\
&u^{\text{lo}} \le u_k \le u^{\text{hi}} && \text{hard — actuator range is physical}\\[4pt]
&x^{\text{lo}} - \xi^-_k \le x_k \le x^{\text{hi}} + \xi^+_k, && \xi^\pm_k\in[0,\infty)\\
&g_{\text{bus}}(x_k,u_k) = \sigma_k, && \sigma_k\in[0,\infty)\\
&u_{k,j} \le \bar s_j + \omega_{k,j},\quad j\in\mathcal C, && \omega_{k,j}\in[0,\infty)\\
&|u_{k,j}-u_{k-1,j}| \le \rho_j + \eta_{k,j},\quad j\in\mathcal M', && \eta_{k,j}\in[0,\infty)
\end{aligned}
$$

with $\mathcal M'$ = **process setpoint columns only** — battery and curtailment
removed.

Every inequality that can conflict with another now has an unbounded slack, so
**the problem is feasible by construction.** A safety filter that can fail to
return an answer is not a safety filter.

### Objective

$$
\min_{x,u,\xi,\sigma,\omega,\eta}\;
\underbrace{\big\|u_0 - \bar u_0\big\|^2_{W}}_{\text{minimal edit}}
\;+\;
\underbrace{\rho_x\sum_k(\xi^-_k+\xi^+_k)
+\rho_\sigma\sum_k\sigma_k
+\rho_\omega\sum_{k,j}\omega_{k,j}
+\rho_\eta\sum_{k,j}\eta_{k,j}}_{\text{violation, exact-penalty weights}}
\;+\;
\underbrace{\epsilon\sum_{k\ge1}\|u_k-u_{k-1}\|^2}_{\text{regularisation only}}
$$

**$\bar u_0$ is the outer layer's complete proposed action** — all eleven
columns, including battery and curtailment, taken from `Band.dispatch`,
`plan.battery_W` and `plan.curtail`. Those three are already published and
currently unused.

$W$ weights the columns by how much a unit edit matters; a diagonal of ones in
scaled units is the right starting point.

**No terminal term and no economics.** The tail exists only to prove a feasible
continuation exists. $\epsilon$ is small (≈$10^{-3}$) and present solely to keep
the tail well-posed — it must not be large enough to shape $u_0$.

### Weight ordering

The violation penalties must be **exact-penalty** weights: large enough that any
constraint violation is preferred only when the alternative is infeasibility.
With the edit term $O(1)$:

$$\rho_\omega \sim 10^3 \;<\; \rho_\eta \sim 10^3 \;<\; \rho_x \sim 10^5 \;<\; \rho_\sigma \sim 10^6$$

Ordering rationale: exceeding a plan band or a slew limit is a nuisance; leaving
a state box is a safety matter; failing to supply the bus is the thing the whole
layer exists to prevent.

### Band width

The outer layer should publish a genuine band. Either:

- **(a)** keep `setpoint_max` as a ceiling with headroom above the intended
  dispatch — $\bar s_j = (1+\alpha)\,d_j$ for some margin $\alpha$ — and track
  `dispatch`; or
- **(b)** publish $[\underline s_j, \overline s_j]$ explicitly and track the
  midpoint.

Either fixes §4.5. (a) is the smaller change and matches the original intent
that the outer layer sets *the maximum power demandable*, leaving the inner
layer free to modulate beneath it. **Not yet done** — §4.5 still stands.

### 5.1 Three details found by measurement

**Slacks are floored at 1e-8, not 0** (`SLACK_FLOOR`). IPOPT relaxes every bound
by `bound_relax_factor * max(1, |bound|)`, default 1e-8, so a slack declared
`lb = 0` may sit at −1e-8. Multiplied by a penalty weight of 1e6 that numerical
slop contributed **−0.06**, and the state slacks **−0.198**, against an edit term
of 0.003 — together 86× the quantity the filter exists to minimise, so the
solver's strongest gradient pointed at mining bound tolerance. Setting the bound
*to* the relaxation width makes the relaxed bound exactly zero. Two alternatives
were measured and both cost more: `bound_relax_factor = 0` was 39 % slower (it
removes a relaxation the barrier method wants), and scaling the penalties down
by 1e-2 was 52 % slower.

**The soft state box skips states that cannot bind** (`SLOW_STATES`) — monotone
accumulators (`efc`, `consumed_kg`, `ch4_kg`) and degradation states (`fade`,
`cycle_number`, `v_degradation`, `catalyst_activity`), which move on timescales
of weeks. Excluded states keep their *hard* box, so this removes a slack that
can never be needed rather than a constraint. Worth 66 of 360 columns; measured
at 1.8 % of solve time, which is the honest size of it.

**Horizon 5 min, dt 60 s** (was 1 h / 600 s). dt now matches the simulator's own
integration step, so the dynamics constraint and the plant agree by
construction. Against 1 h / 600 s on identical busy states: **half the
iterations, 2.4× faster, a smaller problem, and the same action to within 1 %**
on the edit term. Dropping the horizon is only defensible in filter mode — there
is no terminal term whose placement depends on it.

One caveat carried in `dispatch_nmpc.py`: `rate_limit` is a single per-interval
number, but the k = 0 row spans `resolve_interval_s` (600 s) while k > 0 rows
span `dt_s` (60 s). It is set for k = 0, where it binds against the applied
control; the tail is consequently ten times too permissive. The clean fix is to
express slew per second and scale each row by its own gap.

---

## 6. Change log

| # | change | status |
| --- | --- | --- |
| 1 | slack upper bounds to infinity | done |
| 2 | state box soft, with two-sided slacks | done |
| 3 | rate limit on process setpoints only, and soft | done |
| 4 | drop terminal term in filter mode | done |
| 5 | objective becomes the first-action edit plus penalties | done |
| 6 | reference is all 11 columns from the plan | done |
| 7 | band ceiling gets width above `dispatch` | **not done** — §4.5 stands |
| 8 | remove the dead `prices` argument | not done — §1.1 |
| 9 | dual recovery optional, off by default | done |
| 10 | slack floor at 1e-8 | done — §5.1 |
| 11 | trim soft box on slow states | done — §5.1 |
| 12 | horizon 5 min, dt 60 s | done — §5.1 |
| 13 | SX expansion in the IPOPT backend | done — §6.1 |

### 6.1 Where the time went

An earlier draft of this section concluded **"the filter is slower"**, on the
grounds that it is a larger NLP than the tracking formulation. That was measured
correctly and reasoned about wrongly: the extra columns were never the cost.

Cumulative, on ten identical fully-committed states:

| | s/solve | iterations |
| --- | ---: | ---: |
| tracking formulation, 1 h / 600 s | 2.744 | 49.6 |
| filter at 5 min / 60 s | 1.217 | 24.8 |
| + slack trim (§5.1) | 1.195 | 25.3 |
| + `expand=True` | **0.511** | 25.3 |

**5.4× faster.** The finer dt halved the iterations; SX expansion cut the cost of
each one by 2.4×.

`expand=True` converts the problem from CasADi's MX graph — walked node by node
through an interpreter, with matrix-valued operations — to a scalar SX graph
that CasADi can optimise properly. Same iteration count, control identical to
3e-16, and since construction is *inside* the measured solve the solving itself
went from 1.227 s to 0.211 s, a **5.8×**. It is the single largest win in this
document and it is one option. `EconomicPlanner` opts out: its 240-step horizon
is two orders larger and the trade has not been measured there.

Two things that measurement refuted along the way, recorded so they are not
retried:

* **Removing variables does not help.** The slack trim removed 18 % of the
  columns for 1.8 % of the time. Cost tracks the number of times the plant model
  is evaluated — N intervals of a nonlinear 16-state graph with AD — not the
  column count. This is also why the enables, pinned and carrying no freedom,
  are not worth eliminating: IPOPT's `fixed_variable_treatment` already removes
  them from the KKT system (verified — it reports 357 variables where the
  builder declared 399), and the gate `smooth_step(e-0.5, 0.05)` evaluates to
  2.06e-9 rather than 0, so pinning would not let CasADi fold the expression
  away either.
* **SQP is not competitive here.** qpOASES: 4.5× slower per solve, 33/72
  failures against 12/72, and 40 bus trips where IPOPT produced none. qrqp was
  17× slower; OSQP did not converge. Plausibly the slack-heavy formulation is
  primal-degenerate, which active-set methods handle badly and interior-point
  methods do not notice.

**The largest remaining item is solver construction.** With `expand=True` it is
0.307 s of a 0.511 s solve — 60 %. It is paid every replan because weather is
baked into the expression graph as numeric constants, so the one-entry cache in
`IpoptBackend._solver_for` never hits. Making weather, the initial state and the
reference action CasADi *parameters* would let one solver be built and reused,
and is worth roughly another 2×. (An earlier version of this section dismissed
caching on the grounds that construction was 45 ms — true for MX, and no longer
true.)

## 7. Measured behaviour

**3 days, Seville, reference sizing, €5.50/kg, seed 0.** Both controllers on the
identical window, so the comparison is like-for-like.

| | `dispatch` | `dispatch-nmpc` (filter) |
| --- | ---: | ---: |
| `nmpc_used` | — | **100.0 %** |
| solves / failures | — | 432 / **0** |
| iterations | — | mean 23.3, max 53 |
| solve time | — | mean 0.394 s, max 1.152 s |
| **bus trips** | **180** | **0** |
| bus interventions | 2380 | 769 |
| shed | 2787 kWh | 1577 kWh |
| methane | 138.5 kg/day | **139.9 kg/day** |
| LCOM | 6.13 | **6.10** |
| curtailed | 20.1 % | 19.2 % |
| battery EFC/day | 0.73 | 0.76 |
| limiting subsystem | reactor capacity | solar |
| wall time | 98 s | 251 s |

**Zero solver failures over three days** — the first long-window verification
this layer has passed. Every earlier attempt looked healthy on a short window
and fell apart at three days: the tracking formulation reported 98.6 % success
at a quarter day and 78.2 % at three, with 22 infeasibilities.

The inner layer **eliminates bus trips and makes slightly more methane**. That
is stronger than the architecture required — the expected trade was production
for constraint satisfaction, and the tracking formulation did pay it (−2.8 %
methane for 243 → 26 trips). A plant that never trips does not lose the
production a trip costs.

### 7.1 One thing that does not add up

`nmpc_predicted_shed_kWh` reads **0.000, mean and max**, while the plant shed
**1577 kWh**.

That diagnostic exists to say *before the fact* that the dispatch layer has
committed more load than the bus can carry, and it is silent while the bus
sheds. So the filter halved the shedding without ever predicting any of it — the
reduction is a side effect, not the mechanism working as designed. Either the
5-minute horizon is too short to see the shortfall, or the inner bus model
disagrees with `bus.reconcile`. This is the largest open gap between what this
layer claims and what it does.

### 7.2 What these numbers supersede

Everything previously recorded for this layer was measured at
`methane_price_per_kg = 3.00`, which was mis-tuned: that figure was chosen
against a battery wear cost of €0.042/kWh, and correcting the battery economics
raised it 35 % to €0.0565/kWh — across the ≈€0.043/kWh night-time shadow price —
so the LP stopped cycling the battery and ran it flat, reproducing the failure
mode documented at €1.80/kg. The default is now 5.50, above the corrected
crossover with margin rather than at its edge.

Consequently **methane and LCOM here are not comparable** to the four experiment
reports in `experiments/`, or to the plan-fidelity figures, all of which are
still at €3.00 and are due a re-run.
