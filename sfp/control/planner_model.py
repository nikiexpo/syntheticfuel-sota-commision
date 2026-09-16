"""The planner's reduced model: six states that carry value across hours.

The full plant has sixteen states. The planner does not need them. Stack and
reactor temperatures, catalyst activity and voltage degradation all settle within
minutes against an hourly grid, so they are eliminated by assuming quasi-steady
state and holding them at their regulated values. What is left is exactly the set
of quantities that *store* something from one hour to the next:

    z = (SoC, n_CaCO3, N, n_H2, n_CO2, T_kiln)

Each is a buffer. The battery stores electricity, the carbonate stores carbon
cheaply, the two gas tanks store it expensively but usably, the kiln's refractory
stores heat, and N is the running cost of having used the carbonate buffer at all.
Scheduling this plant *is* co-scheduling those five things, which is why the
planner's state vector is the list of them.

Fidelity, and where it is deliberately given up
-----------------------------------------------
Rate expressions are taken from the real subsystem models wherever the reduced
state supports them -- the Baker equilibrium, the Arrhenius kinetics, the
Grasa-Abanades deactivation curve and the Butler-Volmer polarisation curve are
the same equations the plant integrates, evaluated symbolically. `test_planner.py`
pins the reduced model against the full one so the two cannot drift apart.

Three things are approximated, and each is a deliberate trade:

1. **Enables are relaxed to [0, 1].** The real subsystems gate on a sharp
   `smooth_step` at 0.5, which is nearly a discontinuity and would wreck an NLP.
   Here commitment is a continuous variable `e` with `s <= e` and, where a
   minimum load exists, `s >= s_min * e`. That is the standard relaxation of a
   unit-commitment binary, and it is rounded on the way down to the bus.

2. **The reactor runs at its regulated temperature or not at all.** With the
   temperature eliminated there is no relight transient, so the 31 kWh light-off
   is charged explicitly as a start-up cost instead. The kiln needs no such term:
   its temperature *is* a planner state, so the cost of a cold start emerges from
   the energy balance rather than being priced in by hand.

3. **The water tank is not a planner state.** Over a ten-day horizon at the
   reference sizing the tank falls from 70 % to roughly 5 %, so it is close to
   binding and this is a real omission rather than a safe one. It is recorded in
   `bookkeeping/03_SIZING.md` and belongs in the planner as a seventh state if a
   run is ever extended past ten days.

The inner NMPC at M4 carries the full sixteen states, so these approximations are
corrected at the layer that acts on them -- which is the entire argument for
having two layers rather than one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from sfp.models import mathx as mx
from sfp.units import DH_CALCINATION, F_FARADAY, M_CACO3, M_CH4, M_H2O

#: Order of the reduced state vector. Used everywhere; never index by number.
#:
#: `water_kg` is the seventh, added after the M3 review. It was flagged as a
#: known omission when this model was first written and it is a real one: at the
#: reference sizing the plant consumes about 312 kg/day net of condensate
#: recovery, against 3250 kg usable in the tank, so the tank empties in around
#: ten days. Over a seven-day horizon it does not bind, which is precisely why it
#: has to be in the model -- a constraint that is nearly active is one the
#: planner should be trading against, not one it should be blind to.
STATE_NAMES: tuple[str, ...] = (
    "soc", "n_caco3", "cycle_number", "n_h2", "n_co2", "kiln_temperature_K",
    "water_kg",
)

#: Order of the reduced control vector.
#:
#: The two vent rates are not really controls -- no operator opens a relief valve
#: on purpose -- but they must be decision variables, because without them the
#: problem is *infeasible* rather than merely unattractive. The dynamics are
#: equalities and the tank levels are box-bounded, so a plan in which the
#: electrolyser fills the hydrogen tank has no solution at all unless the model
#: can say where the surplus went. The real plant vents (`gas_h2_vented_mol_s`),
#: so the planner must be able to as well. Venting is never attractive -- it
#: throws away methane the objective wants -- so the optimiser avoids it on its
#: own, and a plan that vents anyway is reporting something worth reading.
#:
#: `water_makeup_kg_s` is a real decision: water arrives by road at a remote arid
#: site, so scheduling deliveries is something an operator does. Making it a
#: control rather than a fixed parameter also removes the positive part from the
#: cost -- the delivery *is* the positive part -- and gives the tank a recourse,
#: so a long horizon cannot become infeasible simply by running dry.
CONTROL_NAMES: tuple[str, ...] = (
    "contactor_flow", "calciner_heat", "electrolyser_load", "sabatier_feed",
    "contactor_on", "calciner_on", "electrolyser_on", "sabatier_on",
    "battery_charge_W", "battery_discharge_W", "curtail",
    "h2_vent_mol_s", "co2_vent_mol_s", "water_makeup_kg_s",
)

#: Subsystems the planner commits, in the order their setpoint/enable pairs appear.
COMMITTED: tuple[str, ...] = ("contactor", "calciner", "electrolyser", "sabatier")

N_Z = len(STATE_NAMES)
N_U = len(CONTROL_NAMES)

_IZ = {name: i for i, name in enumerate(STATE_NAMES)}
_IU = {name: i for i, name in enumerate(CONTROL_NAMES)}


def zi(name: str) -> int:
    """Index of a reduced state by name."""
    return _IZ[name]


def ui(name: str) -> int:
    """Index of a reduced control by name."""
    return _IU[name]


@dataclass
class PlannerLimits:
    """Box bounds on the reduced state, in state units."""

    soc: tuple[float, float]
    n_caco3: tuple[float, float]
    cycle_number: tuple[float, float]
    n_h2: tuple[float, float]
    n_co2: tuple[float, float]
    kiln_temperature_K: tuple[float, float]
    water_kg: tuple[float, float]

    def lower(self) -> np.ndarray:
        return np.array([getattr(self, n)[0] for n in STATE_NAMES], dtype=float)

    def upper(self) -> np.ndarray:
        return np.array([getattr(self, n)[1] for n in STATE_NAMES], dtype=float)


@dataclass
class PlannerModel:
    """Six-state reduced model, evaluated numerically or symbolically.

    Constructed from a `Plant`, so it can never be parameterised differently from
    the plant the controller is attached to. (The *truth* simulator's parameters
    are perturbed away from these at M5; that mismatch is the point, and it enters
    through the plant the controller is given, not through this class.)
    """

    plant: Any
    #: minimum electrolyser load as a fraction of rating, when committed
    electrolyser_min_load: float = field(init=False)

    def __post_init__(self) -> None:
        self.pv = self.plant["pv"]
        self.battery = self.plant["battery"]
        self.solids = self.plant["solids"]
        self.contactor = self.plant["contactor"]
        self.calciner = self.plant["calciner"]
        self.electrolyser = self.plant["electrolyser"]
        self.gas = self.plant["gas"]
        self.sabatier = self.plant["sabatier"]

        self.electrolyser_min_load = float(
            self.electrolyser.p.current_density_min_fraction
        )

        # Fast states, held at the values their local regulatory loops maintain.
        # The electrolyser's degradation term is dropped rather than frozen: over
        # ten days it moves by ~0.2 mV, which is below the resolution of anything
        # the planner decides.
        self._x_electrolyser = np.array(
            [self.electrolyser.p.temperature_setpoint_K, 0.0], dtype=float
        )
        self._x_sabatier = np.array(
            [self.sabatier.p.temperature_target_K, 1.0, 0.0], dtype=float
        )

    # --- capacities -------------------------------------------------------
    @property
    def n_total_mol(self) -> float:
        return float(self.solids.p.n_total_mol)

    @property
    def h2_capacity_mol(self) -> float:
        return float(self.gas.p.h2_capacity_mol)

    @property
    def co2_capacity_mol(self) -> float:
        return float(self.gas.p.co2_capacity_mol)

    def limits(self, z0=None, horizon_s: float = 0.0) -> PlannerLimits:
        """Bounds the planner may not plan outside of.

        Given the current state and horizon, the two monotone states get bounds
        snug enough to be informative rather than nominal.

        **Why snugness matters here and not elsewhere.** The cycle number moves
        by about 2 over a week but was previously bounded `[0, 1e4]`. A box three
        orders of magnitude wider than the reachable set tells the solver nothing,
        wastes the barrier's interior, and -- in any formulation that scales by
        bound range -- can shrink a state's dynamics defect until it vanishes
        inside the convergence test. The same applies to the carbonate inventory:
        deactivation means capacity only ever falls, so `n_tot * X_N(N_0)` is a
        hard ceiling that the nominal `n_tot` overstates by a factor of three.

        The carbonate *capacity* limit proper is still a nonlinear constraint in
        `planner.py`, because it depends on the cycle number, which is a state.
        This box is the constant part of it.
        """
        b = self.battery.p
        cap_kg = float(self.plant["water"].p.water_capacity_kg)

        n0 = float(z0[_IZ["cycle_number"]]) if z0 is not None else \
            float(self.solids.p.cycle_number_initial)
        # the loop cannot turn faster than the kiln feeder allows
        reach = (float(self.calciner.p.calcination_rate_max_mol_s)
                 * max(horizon_s, 0.0) / self.n_total_mol)
        caco3_ceiling = self.n_total_mol * float(self.max_conversion(n0))

        return PlannerLimits(
            soc=(float(b.soc_min), float(b.soc_max)),
            n_caco3=(0.0, caco3_ceiling),
            # a small margin below the current value: the cycle counter is
            # monotone, so n0 is a true floor, but putting the initial-state
            # equality exactly on a box face makes the barrier start on the
            # boundary for no benefit
            cycle_number=(max(n0 - 0.1, 0.0), n0 + max(reach, 1.0)),
            n_h2=(float(self.gas.p.h2_min_fraction) * self.h2_capacity_mol,
                  self.h2_capacity_mol),
            n_co2=(float(self.gas.p.co2_min_fraction) * self.co2_capacity_mol,
                   self.co2_capacity_mol),
            kiln_temperature_K=(250.0, float(self.calciner.p.temperature_max_K)),
            water_kg=(float(self.plant["water"].p.water_min_fraction) * cap_kg, cap_kg),
        )

    def state_scale(self) -> np.ndarray:
        """Characteristic magnitude of each reduced state.

        The planner solves in non-dimensional variables. Without this the NLP
        mixes a state of order 0.5 (SoC) with one of order 30,000 (moles of
        hydrogen) and one of order 1000 (kelvin), and IPOPT's convergence test,
        which is a norm over all of them at once, is then dominated by whichever
        happens to be largest. Scaling every state to order one is the single
        change that turns this problem from "maximum iterations exceeded" into a
        clean solve.
        """
        return np.array(
            [1.0, self.n_total_mol, 1.0,
             self.h2_capacity_mol, self.co2_capacity_mol, 1000.0,
             float(self.plant["water"].p.water_capacity_kg)],
            dtype=float,
        )

    def control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Physical box bounds on one control vector."""
        lo = np.zeros(N_U)
        hi = np.ones(N_U)
        p_max = float(self.battery.max_power_W)
        hi[_IU["battery_charge_W"]] = p_max
        hi[_IU["battery_discharge_W"]] = p_max
        # The vents exist for feasibility, not as a real duty: they only ever
        # need to carry an overflow, and the largest possible one is the
        # electrolyser's full output.
        hi[_IU["h2_vent_mol_s"]] = 1.0
        hi[_IU["co2_vent_mol_s"]] = 1.0
        # net consumption is ~3.6e-3 kg/s at full production; this is ample
        hi[_IU["water_makeup_kg_s"]] = 0.02
        return lo, hi

    def control_scale(self) -> np.ndarray:
        """Characteristic magnitude of each reduced control.

        Defined as the upper bound, so **every control is [0, 1] once scaled**.
        Deriving the scale from the bound rather than picking it separately is
        what keeps the two consistent: an earlier version scaled the water
        make-up by 0.01 while bounding it at 0.5, which put a scaled variable on
        a box fifty units wide in a problem where everything else lived in the
        unit interval. That is exactly the kind of mismatch the non-dimensional
        formulation exists to prevent, and it cost hundreds of iterations.
        """
        return self.control_bounds()[1]

    def power_scale(self) -> float:
        """Characteristic bus power, W. Scales the balance constraint to order one."""
        return float(max(self.pv.rated_ac_W, 1e5))

    def initial_state(self, plant_state: Mapping[str, np.ndarray]) -> np.ndarray:
        """Project the full plant state onto the reduced one."""
        return np.array(
            [
                float(plant_state["battery"][0]),
                float(plant_state["solids"][1]),
                float(plant_state["solids"][2]),
                float(plant_state["gas"][0]),
                float(plant_state["gas"][1]),
                float(plant_state["calciner"][0]),
                float(plant_state["water"][0]),
            ],
            dtype=float,
        )

    # --- derived quantities ----------------------------------------------
    #
    # Guards deleted rather than smoothed
    # -----------------------------------
    # The plant's models clamp several quantities that a state box already makes
    # unreachable. Deleting such a guard is strictly better than smoothing it:
    # it is exact, it costs no variable and no row, and it removes a kink that
    # would otherwise sit on the feasible set. Each deletion below names the box
    # that makes it safe, because the argument is only valid while that box
    # stands -- `test_deleted_guards_are_unreachable` re-checks them.
    #
    #   fmax(cycle_number, 0)      -> box  cycle_number >= N_0 >= 0
    #   fmax(capacity, 1.0)        -> X_N >= X_r = 0.075, so capacity >= 3750 mol
    #   fmax(n_H2 - cushion, 0)    -> box  n_H2  >= h2_min_fraction * capacity
    #   fmax(n_CO2 - cushion, 0)   -> box  n_CO2 >= co2_min_fraction * capacity
    #   smooth_step(water - 1, 1)  -> box  water >= water_min_fraction * capacity

    def max_conversion(self, cycle_number):
        """Grasa-Abanades sorbent capacity at mean cycle number N.

        Written out rather than delegated to `SolidsInventory.max_conversion`
        only to drop its `fmax(N, 0)` guard, which the cycle-number box makes
        unreachable. Same equation otherwise.
        """
        x_r = self.solids.p.grasa_residual_conversion
        k = self.solids.p.grasa_deactivation_constant
        return 1.0 / (k * cycle_number + 1.0 / (1.0 - x_r)) + x_r

    def capture_capacity_mol(self, cycle_number):
        return self.n_total_mol * self.max_conversion(cycle_number)

    def loading(self, z):
        # capacity >= n_tot * X_r = 3750 mol, so no floor guard is needed
        capacity = self.capture_capacity_mol(z[_IZ["cycle_number"]])
        return mx.smooth_clip(z[_IZ["n_caco3"]] / capacity, 0.0, 1.0, eps=1e-4)

    # --- rates ------------------------------------------------------------
    def rates(self, z, u, w: Mapping[str, Any]) -> dict[str, Any]:
        """The four material rates, mol/s.

        How commitment is relaxed, and why it is *not* a factor here
        -----------------------------------------------------------
        A subsystem's rate is set by its setpoint alone. The enable appears only
        in `powers`, where it scales the parasitic load, and in the constraint
        `setpoint <= enable`. That combination is the standard relaxation of an
        on/off decision: the optimiser *usually* drives the enable down to the
        setpoint, since a smaller enable means less parasitic draw and little
        else depends on it, so at most intervals `e = s` and the parasitic load
        is a linear interpolation of the on/off cost.

        **Usually, not always.** The start-up epigraph `s_k >= e_k - e_{k-1}`
        gives the enable a second job, and the optimiser will hold a partial
        commitment through an idle interval when that is cheaper than paying a
        full start at the next one. Measured: the reactor sitting at `e = 0.465`
        with `s = 0`, drawing 3.7 kW for nothing at about EUR 0.19/h to avoid a
        EUR 2.00 start-up. That is the relaxation working, not failing -- but it
        means `e = s` is not an invariant and nothing should be built on it. What
        the rounding relies on is weaker and does hold:
        `test_commitment_rounding_is_harmless`.

        Multiplying the *rate* by the enable as well -- the obvious first move --
        is wrong, and quietly so. With `s <= e` and rate proportional to `s * e`,
        the achieved rate is quadratic in the commitment, so the planner reports
        a plant that produces `e^2` of its setpoint. It converges, the schedule
        looks plausible, and the plant under-produces relative to the plan for
        reasons nothing in the output explains.

        The calciner is the exception, and physically so: its rate depends on
        kiln temperature and feedstock, not on the heater setting, so without a
        gate a hot kiln would calcine whether or not it was committed. There `e`
        is the *feeder*, which is a real piece of equipment that is either
        turning or not, and the rate is linear in it.

        Why these expressions are written out rather than delegated
        ----------------------------------------------------------
        Every *physical* term here is taken from the corresponding subsystem --
        the Baker equilibrium and Arrhenius kinetics from `Calciner`, the NTU
        capture curve from `AirContactor`, the logistic equilibrium and Damkohler
        approach from `SabatierReactor`, Faraday's law from `Electrolyser`. What
        is changed is only the *non-smooth* guards.

        The plant's models use `fmax` and `fmin` to clamp rates at zero and to
        pick whichever buffer is shorter. Those are exactly right for simulation
        and are poison for a gradient-based optimiser, because each is a kink and
        every one of them sits precisely where the planner wants to operate: at
        `rate = 0` for a cold kiln, at the crossover between a CO2-limited and an
        H2-limited reactor. IPOPT crawls along such ridges with step sizes of
        1e-2 and hits its iteration limit. Replacing them with the smooth
        equivalents already in `mathx` -- at a smoothing width small enough to be
        physically irrelevant -- is what makes the problem solvable.

        `test_planner_rates_match_the_plant` pins these against the subsystem
        models so the smoothing cannot quietly become a different model.
        """
        flow = u[_IU["contactor_flow"]]
        heat = u[_IU["calciner_heat"]]
        load = u[_IU["electrolyser_load"]]
        feed = u[_IU["sabatier_feed"]]
        e_cal = u[_IU["calciner_on"]]

        t_kiln = z[_IZ["kiln_temperature_K"]]
        n_caco3 = z[_IZ["n_caco3"]]

        # --- contactor: the real capture model, which is already smooth
        con = self.contactor
        w_con = dict(w)
        w_con["solids_loading"] = self.loading(z)
        r_carb = con.capture_rate_mol_s(mx.vertcat(flow, 1.0), w_con)

        # --- calciner: rate is set by temperature and feedstock, not by the
        # heater setting. The heater acts only through the energy balance, which
        # is why kiln commitment is a genuinely dynamic decision.
        #
        # Composed from the calciner's own three limits, without its outer
        # `fmax(rate, 0)`: the three factors are individually non-negative, so
        # the clamp is redundant, and its kink sits at exactly the operating
        # point of a cold kiln.
        cal = self.calciner
        r_calc = (
            cal.p.calcination_rate_max_mol_s
            * cal.driving_force(t_kiln)
            * mx.smooth_clip(cal.kinetic_factor(t_kiln), 0.0, 1.0, eps=1e-3)
            * cal.feedstock_factor(n_caco3)
            * e_cal
        )

        # --- electrolyser: Faraday's law at the commanded current
        ele = self.electrolyser
        current = load * ele.p.current_density_max_A_cm2 * ele.p.active_area_cm2
        r_h2 = ele.p.faraday_efficiency * ele.p.n_cells * current / (2.0 * F_FARADAY)

        # --- reactor: conversion at the regulated temperature, throttled by
        # whichever buffer is short. `smooth_min` replaces the plant's `fmin`;
        # the 1e-3 width is a thousandth of a fractional availability.
        # The cushion guards are deleted: the tank boxes put both differences at
        # or above zero everywhere on the feasible set. The water gate goes the
        # same way, since `water_kg >= 250` is now a state box.
        sab = self.sabatier
        co2_free = z[_IZ["n_co2"]] - self.gas.p.co2_min_fraction * self.co2_capacity_mol
        h2_free = z[_IZ["n_h2"]] - self.gas.p.h2_min_fraction * self.h2_capacity_mol
        # Two different widths, because the two operators fail in opposite ways.
        #
        # The *clips* get 1e-2 rather than the plant's 1e-3. Deleting the cushion
        # guards helped the gradient but moved the sharpest curvature inward:
        # what the guard used to smooth over ~1 mol, the clip now turns over in
        # 0.05 mol of CO2 -- a near-discontinuity sitting on the tank's own lower
        # box, which is exactly where a starved reactor operates.
        #
        # The *min* keeps 1e-3, because `smooth_min` is biased low by eps/2
        # wherever its arguments are equal -- and they are equal, at exactly 1.0,
        # whenever both buffers are comfortable, which is most of the time. At
        # 1e-2 that is a systematic 0.5 % under-prediction of the reactor rate
        # across the whole plan: small, invisible, and in the direction that
        # would quietly make the planner pessimistic about its own product.
        availability = mx.smooth_min(
            mx.smooth_clip(co2_free / 50.0, 0.0, 1.0, eps=1e-2),
            mx.smooth_clip(h2_free / 200.0, 0.0, 1.0, eps=1e-2),
            eps=1e-3,
        )
        conversion = sab.conversion(self._x_sabatier, mx.vertcat(feed, 1.0))
        r_sab = feed * sab.p.co2_feed_max_mol_s * conversion * availability

        return {"carbonation": r_carb, "calcination": r_calc,
                "electrolysis": r_h2, "methanation": r_sab}

    # --- powers -----------------------------------------------------------
    def powers(self, z, u, w: Mapping[str, Any], rates=None) -> dict[str, Any]:
        """Electrical load by subsystem, W, positive = consumed.

        Mirrors each subsystem's own power expression with the parasitic term
        scaled by the relaxed enable instead of the sharp gate. Pinned against
        the real models by `test_planner_powers_match_the_subsystems`.
        """
        rates = rates or self.rates(z, u, w)
        flow = u[_IU["contactor_flow"]]
        heat = u[_IU["calciner_heat"]]
        load = u[_IU["electrolyser_load"]]
        e_con = u[_IU["contactor_on"]]
        e_cal = u[_IU["calciner_on"]]
        e_ele = u[_IU["electrolyser_on"]]
        e_sab = u[_IU["sabatier_on"]]

        con = self.contactor
        p_contactor = (con.fan_power_rated_W * mx.power(flow, con.p.fan_exponent)
                       + con.fan_idle_W * e_con)

        cal = self.calciner
        p_calciner = heat * cal.heater_power_rated_W + cal.standby_power_W * e_cal

        ele = self.electrolyser
        i = load * ele.p.current_density_max_A_cm2
        v_cell = ele.cell_voltage(i, self._x_electrolyser[0], 0.0)
        p_stack = ele.p.n_cells * v_cell * i * ele.p.active_area_cm2
        # stack heat above thermoneutral is rejected by the cooling circuit
        heat_W = ele.p.n_cells * (v_cell - ele.p.thermoneutral_voltage_V) \
            * i * ele.p.active_area_cm2
        p_electrolyser = (p_stack + ele.auxiliary_power_W * e_ele
                          + ele.p.cooling_parasitic_fraction
                          * mx.smooth_max(heat_W, 0.0, eps=100.0))

        # The reactor's draw is flat in feed rate -- preheat and auxiliaries --
        # which is why the bus treats it as all-or-nothing and why the planner
        # charges a start-up cost rather than a throttling cost.
        p_sabatier = self.sabatier.p.auxiliary_power_kw * 1e3 * e_sab

        # The CO2 compressor is not commanded: its draw follows calcination.
        p_compressor = self.gas.p.co2_compressor_kJ_per_mol * 1e3 * rates["calcination"]

        return {"contactor": p_contactor, "calciner": p_calciner,
                "electrolyser": p_electrolyser, "sabatier": p_sabatier,
                "compressor": p_compressor}

    def total_load_W(self, z, u, w, rates=None):
        return sum(self.powers(z, u, w, rates).values())

    # --- dynamics ---------------------------------------------------------
    def rhs(self, z, u, w: Mapping[str, Any]):
        """dz/dt for the reduced model."""
        rates = self.rates(z, u, w)
        r_carb = rates["carbonation"]
        r_calc = rates["calcination"]
        r_h2 = rates["electrolysis"]
        r_sab = rates["methanation"]

        # --- battery
        b = self.battery.p
        p_chg = u[_IU["battery_charge_W"]]
        p_dis = u[_IU["battery_discharge_W"]]
        usable_J = self.battery.nominal_energy_J  # fade is not a planner state
        d_soc = ((b.eta_charge * p_chg - p_dis / b.eta_discharge) / usable_J
                 - b.self_discharge_per_day / 86400.0 * z[_IZ["soc"]])

        # --- kiln energy balance, the real one
        heater_W = (u[_IU["calciner_heat"]] * self.calciner.heater_power_rated_W
                    * self.calciner.p.heater_efficiency)
        t_kiln = z[_IZ["kiln_temperature_K"]]
        temp_amb_K = w.get("temp_air", 20.0) + 273.15
        ua = (self.calciner.p.heat_loss_UA_W_K
              + self.calciner.p.wind_loss_coefficient * mx.fmax(w.get("wind_speed", 1.0), 0.0))
        loss_W = ua * (t_kiln - temp_amb_K)
        reaction_W = r_calc * DH_CALCINATION
        # smoothed at 1 K, which is below the resolution of anything the planner
        # decides, and removes a kink at the cold-kiln operating point
        sensible_W = (r_calc * M_CACO3 * self.calciner.p.solids_heat_capacity_J_kg_K
                      * mx.smooth_max(
                          t_kiln - self.calciner.p.solids_feed_temperature_K, 0.0, eps=1.0))
        d_temperature = (heater_W - loss_W - reaction_W - sensible_W) \
            / self.calciner.p.thermal_capacity_J_K

        # --- water: electrolysis consumes one mole per mole of H2, the reactor
        # returns two per mole of CH4 less what the condenser misses
        water = self.plant["water"]
        d_water = (u[_IU["water_makeup_kg_s"]]
                   + 2.0 * water.p.condensate_recovery * r_sab * M_H2O
                   - r_h2 * M_H2O)

        return mx.vertcat(
            d_soc,
            r_carb - r_calc,
            r_calc / self.n_total_mol,
            r_h2 - 4.0 * r_sab - u[_IU["h2_vent_mol_s"]],
            r_calc - r_sab - u[_IU["co2_vent_mol_s"]],
            d_temperature,
            d_water,
        )

    #: Longest RK4 sub-step, s. The kiln climbs ~170 K/h at full heater duty
    #: while its own reaction rate changes sharply with temperature, so a single
    #: RK4 step over an hour visibly overshoots the calcination threshold. Half
    #: an hour is short enough; the check is
    #: `test_rk4_step_agrees_with_a_fine_forward_euler`.
    MAX_SUBSTEP_S: float = 1800.0

    #: Ceiling on sub-steps per interval, regardless of how long the interval is.
    MAX_SUBSTEPS: int = 3

    def substeps_for(self, dt_s: float) -> int:
        """How many RK4 sub-steps an interval of `dt_s` needs.

        Two competing requirements, and the cap is where they are traded.

        Accuracy wants the sub-step short: a six-hour interval integrated in two
        steps would be badly wrong for a kiln whose reaction rate switches within
        an hour. But scaling the count with the interval length defeats the
        graded grid entirely. Seven days at `MAX_SUBSTEP_S` costs 1344 RHS
        evaluations whether the grid is 168 uniform hours or 56 graded intervals
        -- *identical* -- because halving the number of intervals doubles the
        sub-steps in each. The grid would shrink the variable count and leave the
        expression graph, and therefore the cost per iteration, untouched. Worse,
        a twelve-sub-step RK4 chain is a far more nonlinear constraint row than a
        two-sub-step one, so the Hessian gets denser exactly where the saving was
        supposed to come from.

        The cap resolves it by spending accuracy where accuracy is cheap to lose.
        The first day is hourly and integrates at full resolution; the six-hourly
        tail is a *valuation device* for the terminal inventories, not a schedule
        anyone implements, and it is re-planned long before it arrives. Coarse
        integration out there costs little and buys the whole point of grading.
        """
        return max(1, min(self.MAX_SUBSTEPS, int(np.ceil(dt_s / self.MAX_SUBSTEP_S))))

    def step(self, z, u, w: Mapping[str, Any], dt_s: float, substeps: int | None = None):
        """One RK4 step of length `dt_s`, subdivided to bound the local error."""
        substeps = self.substeps_for(dt_s) if substeps is None else substeps
        h = dt_s / substeps
        for _ in range(substeps):
            k1 = self.rhs(z, u, w)
            k2 = self.rhs(z + 0.5 * h * k1, u, w)
            k3 = self.rhs(z + 0.5 * h * k2, u, w)
            k4 = self.rhs(z + h * k3, u, w)
            z = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return z

    # --- economics --------------------------------------------------------
    def stage_profit_EUR(self, z, u, w, economics, dt_s: float, rates=None):
        """Operating profit over one interval, EUR. Maximised by the planner.

        Only marginal terms appear -- capital is sunk by the time the planner
        runs. Sorbent deactivation is included at its true (small) monetary
        value; the *real* penalty for cycling the calcium loop is the capacity
        constraint it tightens, which the dynamics impose directly.
        """
        rates = rates or self.rates(z, u, w)
        r_sab = rates["methanation"]
        r_calc = rates["calcination"]
        r_h2 = rates["electrolysis"]

        ch4_kg = r_sab * M_CH4 * dt_s
        revenue = economics.p.methane_price_per_kg * ch4_kg

        throughput_J = (u[_IU["battery_charge_W"]] + u[_IU["battery_discharge_W"]]) * dt_s
        efc = throughput_J / (2.0 * self.battery.nominal_energy_J)
        battery_cost = efc * self.battery.cost_per_efc_EUR()

        d_cycles = r_calc / self.n_total_mol * dt_s
        sorbent_cost = d_cycles * self._sorbent_cost_per_cycle_EUR()

        # Water is charged on what is actually delivered. That is both more
        # honest -- the cost is a road tanker, not a stoichiometric balance --
        # and strictly better conditioned, because the delivery is a non-negative
        # control and so *is* the positive part, with no smoothing needed.
        water_kg = u[_IU["water_makeup_kg_s"]] * dt_s
        water_cost = economics.p.water_cost_per_m3 * water_kg / 1000.0

        return revenue - battery_cost - sorbent_cost - water_cost

    def _sorbent_cost_per_cycle_EUR(self) -> float:
        """Value of the capacity destroyed by advancing the mean cycle by one.

        Uses the same derivative as `SolidsInventory.marginal_deactivation_cost`,
        evaluated at the *initial* cycle number so the planner's objective stays
        a fixed linear cost rather than a state-dependent one. The magnitude is
        small (the whole sorbent charge is worth a few hundred euro); the term is
        carried for completeness and because it is the honest place to put a
        larger number if the sorbent price is ever revised upward.
        """
        price = float(self.solids.p.sorbent_cost_per_mol)
        n0 = float(self.solids.p.cycle_number_initial)
        k = float(self.solids.p.grasa_deactivation_constant)
        x_r = float(self.solids.p.grasa_residual_conversion)
        denominator = k * n0 + 1.0 / (1.0 - x_r)
        capacity_lost_per_cycle = k / denominator**2 * self.n_total_mol
        return capacity_lost_per_cycle * price
