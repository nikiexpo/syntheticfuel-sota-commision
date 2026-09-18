# The `dispatch-nmpc` architecture

**Scope.** Formulations only — what problem each layer solves, what passes
between them, and where the present implementation departs from the intended
design. Subsystem models are not reproduced here; they are in
`docs/ASSUMPTIONS.md` and the model modules.

**Status.** This describes the accepted design as built and measured. §1 is the
closed loop, §2 the outer MILP, §3 the inner safety filter, §4 where the solve
time goes, §5 what a three-day run measured, §6 what is still open.

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
| $d_{j,k}$ | `Band.dispatch` | the raw LP variable (W, or mol/s for the reactor) |
| $P^{\text{ch/dis}}_k,\ \gamma_k$ | `plan.battery_W`, `plan.curtail` | the outer layer's own battery and curtailment schedule |
| $z^{\text{tgt}}_{N}$ | `plan.state_at(...)` | buffer levels the plan expects, for the four states in `TARGET_STATES` — unused by the filter |
| ~~$\lambda_k$~~ | `lambda_EUR_per_kWh` | dual on the outer bus balance — **dead, see §1.1** |

The setpoints, battery channels and curtailment fraction together are the
complete proposed action $\bar u_0$ that §3 minimally edits.

### 1.1 The price coordination path is dead

The architecture was originally designed around **price coordination**: the
outer layer publishes $\lambda$, the inner layer solves a local problem at that
price. That path no longer exists.

`InnerNMPC.solve` binds `price = prices[k]` and never reads it again. The filter
has no economics at all, so there is nothing for $\lambda$ to price.

$\lambda$ was dropped deliberately, before the filter existed. Three pricing
variants were tried and all were wrong: pricing process power double-counts
against the enforced bus balance and becomes a standing bias toward
curtailment; pricing the battery at the current $\lambda$ says charging is
worthless at midday, exactly when the battery should be filling, because
$\lambda$ is zero when the plant is already saturated and spilling. The
resolution was to move all intertemporal value upward — and the filter took
that to its conclusion by removing intertemporal value from this layer
entirely. What was left behind is the plumbing.

**Consequence.** Recovering $\lambda$ from a MILP requires fixing the integer
columns and re-solving the continuous problem — a second LP solve on every
replan — for a number nothing consumes. `solve_lp(..., duals=False)` and
`EconomicDispatch(recover_duals=False)` are the defaults; the price survives as
a reported diagnostic and in the tests that check the LP is well-formed.

There is no route for $\lambda$ to re-enter the filter. If price coordination is
wanted again it belongs in a separate mode. (`HierarchicalController`, the
NLP-planner comparator, is where the price-coordinated formulation still lives.)

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
invalidated the old price tuning (§5.2).

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

## 3. The inner problem: a predictive safety filter

`objective="filter"`, the default for `dispatch-nmpc`. Horizon $T = 300$ s,
$\Delta t = 60$ s, so $N = 5$ intervals. State $x\in\mathbb R^{16}$, control
$u\in\mathbb R^{11}$ — a $(\text{setpoint}, \text{enable})$ pair for each of the
four process machines, $(P^{\text{ch}}, P^{\text{dis}})$ for the battery, and
one curtailment fraction.

It answers exactly one question: **what is the smallest edit to the proposed
action that keeps the plant feasible?** It makes no economic decision, and it
decides nothing about the tail — steps $1{:}N$ exist only to certify that a
feasible continuation exists. Only $u_0$ is applied.

### Decision variables

$$
x_{0:N}\in\mathbb R^{16(N+1)},\qquad
u_{0:N-1}\in\mathbb R^{11N},\qquad
\xi^\pm_k,\ \sigma_k,\ \omega_{k,j},\ \eta_{k,j}\in[0,\infty)
$$

$\sigma$ is predicted load shed (units of the power scale), $\xi^\pm$ the
two-sided state-box excursion, $\omega$ the excursion above a band ceiling, and
$\eta$ the slew-limit excursion. All variables are scaled to their own bounds.

Commitment enters as a **bound, not a variable**: $c_{j,k}=1$ pins the enable
column to 1, $c_{j,k}=0$ pins both enable and setpoint to 0. The enable is never
free, which removes four near-discontinuous gates per interval.

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

$\mathcal C$ (capped columns) is the setpoint column of every committed machine.
$\mathcal M'$ (rate-limited columns) is the **process setpoint columns only**.
Battery and curtailment are excluded: neither has a physical slew limit worth
modelling — a converter responds in milliseconds, curtailment is a firing-angle
change — and they are precisely the two channels whose purpose is to close the
bus balance in real time, so limiting them manufactures the imbalance $\sigma$
then has to absorb.

Three things are hard, and each for a physical reason: the initial state is
measured, the dynamics *are* the model, and the actuator range is a range. Every
inequality that can conflict with another carries an **unbounded** slack, so
**the problem is feasible by construction.** This matters more than it sounds: a
slack with a finite upper bound does not soften a constraint, it relocates the
infeasibility, and a safety filter that can fail to return an answer is not a
safety filter. The state box is the sharpest case — a hard box on top of hard
dynamics and a hard initial state is infeasible whenever the state and the model
imply a crossing no admissible control can prevent (a flat battery at night
against the SoC floor, with committed machines drawing mandatory idle load),
which is exactly when the filter is most needed.

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
columns, including battery and curtailment, from `Band.setpoint_max`,
`plan.battery_W` and `plan.curtail`. A channel left out of the reference is a
channel the inner layer is re-deciding rather than filtering.

$W$ is a diagonal of ones in scaled units, so a full-span edit on one column
costs 1. That is what makes the penalty weights below absolute rather than
relative.

**No terminal term and no economics.** A terminal inventory penalty is an
economic objective — a second optimiser on a five-minute horizon, overriding the
240-hour layer that has far better information. A filter that penalises only
$u_0$ has no incentive to drain a buffer, because it is not optimising the tail.
$\epsilon \approx 10^{-3}$ exists solely to keep the tail well-posed and must
not be large enough to shape $u_0$.

### Weight ordering

The violation penalties must be **exact-penalty** weights: large enough that any
constraint violation is preferred only when the alternative is infeasibility.
With the edit term $O(1)$:

$$\rho_\omega \sim 10^3 \;<\; \rho_\eta \sim 10^3 \;<\; \rho_x \sim 10^5 \;<\; \rho_\sigma \sim 10^6$$

Ordering rationale: exceeding a plan band or a slew limit is a nuisance; leaving
a state box is a safety matter; failing to supply the bus is the thing the whole
layer exists to prevent.

### 3.1 Three details found by measurement

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

**Horizon 5 min, dt 60 s.** dt matches the simulator's own integration step, so
the dynamics constraint and the plant agree by construction rather than by
approximation. Against 1 h / 600 s on identical busy states: **half the
iterations, 2.4× faster, a smaller problem, and the same action to within 1 %**
on the edit term. A horizon this short is only defensible for a filter — there
is no terminal term whose placement depends on it, and the tail is a feasibility
certificate rather than a plan.

One caveat carried in `dispatch_nmpc.py`: `rate_limit` is a single per-interval
number, but the k = 0 row spans `resolve_interval_s` (600 s) while k > 0 rows
span `dt_s` (60 s). It is set for k = 0, where it binds against the applied
control; the tail is consequently ten times too permissive. The clean fix is to
express slew per second and scale each row by its own gap.

---

## 4. Where the solve time goes

Cumulative, on ten identical fully-committed states:

| configuration | s/solve | iterations |
| --- | ---: | ---: |
| 1 h / 600 s | 2.744 | 49.6 |
| 5 min / 60 s | 1.217 | 24.8 |
| + slack trim (§3.1) | 1.195 | 25.3 |
| + `expand=True` | **0.511** | 25.3 |

**5.4× faster.** The finer dt halved the iterations; SX expansion cut the cost of
each one by 2.4×. (The first row was measured under the superseded objective, so
it is a configuration baseline rather than an objective-for-objective
comparison; the second row onward are all the filter.)

`expand=True` converts the problem from CasADi's MX graph — walked node by node
through an interpreter, with matrix-valued operations — to a scalar SX graph
that CasADi can optimise properly. Same iteration count, control identical to
3e-16, and since construction is *inside* the measured solve the solving itself
went from 1.227 s to 0.211 s, a **5.8×**. It is the single largest win here and
it is one option. `EconomicPlanner` opts out: its 240-step horizon is two orders
larger and the trade has not been measured there.

Two things measurement refuted, recorded so they are not retried:

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
and is worth roughly another 2×.

---

## 5. Measured behaviour

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

**Zero solver failures over three days.** A short window proves nothing here:
earlier formulations reported 98.6 % success at a quarter day and 78.2 % at
three days, with 22 infeasibilities, so three days is the shortest window worth
quoting.

The inner layer **eliminates bus trips and makes slightly more methane**. That
is stronger than the architecture required — the expected trade was production
for constraint satisfaction, and earlier formulations did pay it (−2.8 % methane
for 243 → 26 trips). A plant that never trips does not lose the production a
trip costs.

### 5.1 One thing that does not add up

`nmpc_predicted_shed_kWh` reads **0.000, mean and max**, while the plant shed
**1577 kWh**.

That diagnostic exists to say *before the fact* that the dispatch layer has
committed more load than the bus can carry, and it is silent while the bus
sheds. So the filter halved the shedding without ever predicting any of it — the
reduction is a side effect, not the mechanism working as designed. Either the
5-minute horizon is too short to see the shortfall, or the inner bus model
disagrees with `bus.reconcile`. This is the largest open gap between what this
layer claims and what it does.

### 5.2 The controller's methane price

`methane_price_per_kg` is a **controller tuning parameter, not a market price**,
and the default of €5.50 is set against the battery's wear cost. An earlier
default of €3.00 was mis-tuned: it had been chosen against a wear cost of
€0.042/kWh, and correcting the battery economics raised that 35 % to €0.0565/kWh
— across the ≈€0.043/kWh night-time shadow price — so the LP stopped cycling the
battery and ran it flat. €5.50 sits above the corrected crossover with margin.

Measured directly (`experiments/price_strategy.py`), the parameter is saturated
above roughly €4: controllers at €4 and €8 differ by under 1 % on methane, EFC,
curtailment and LCOM, with overlapping commitment traces. The threshold matters;
the level above it does not.

Numbers quoted above are at €5.50. The sizing experiments in `experiments/` run
the controller at €6.00 and are internally consistent, so their methane and LCOM
figures are not directly comparable with this table.

---

## 6. Open items

**The band has no width.** `Band.setpoint_max` is set to exactly the setpoint the
LP intends, and the reference $\bar u_0$ is read from the same field. So $\bar
s_j$ is simultaneously the value the filter starts from and the ceiling above
which $\rho_\omega$ applies: the reference sits *on* the constraint, and any
upward edit is penalised at $10^3$ while any downward edit is free. The outer
layer should publish a genuine band — either keep `setpoint_max` as a ceiling
with headroom, $\bar s_j = (1+\alpha)\,d_j$, and reference `Band.dispatch`; or
publish $[\underline s_j,\ \overline s_j]$ and reference the midpoint. The first
is smaller and matches the intent that the outer layer sets the *maximum power
demandable*, leaving the inner layer free to modulate beneath it.

**The `prices` argument is dead.** §1.1. It is still threaded from
`DispatchNMPCController` into `InnerNMPC.solve` and bound per interval.

**`nmpc_predicted_shed_kWh` reads zero.** §5.1 — the largest gap between what
this layer claims and what it does.

**No information flows up.** §1. Band violations, shed predictions and solver
failures are recorded as diagnostics and discarded; the outer layer never learns
that its plan was infeasible.

**The rate limit spans the wrong interval at $k = 0$.** §3.1.

**The filter cannot decommit.** It can drive a setpoint to zero but cannot clear
an enable the outer layer has pinned, so a committed machine keeps drawing idle
load — up to 194 kWh/day at the smallest pack size measured. See
`experiments/BATTERY_SIZING.md` §4.1.
