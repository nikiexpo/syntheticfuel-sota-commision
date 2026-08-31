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
STATE_NAMES: tuple[str, ...] = (
    "soc", "n_caco3", "cycle_number", "n_h2", "n_co2", "kiln_temperature_K",
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
CONTROL_NAMES: tuple[str, ...] = (
    "contactor_flow", "calciner_heat", "electrolyser_load", "sabatier_feed",
    "contactor_on", "calciner_on", "electrolyser_on", "sabatier_on",
    "battery_charge_W", "battery_discharge_W", "curtail",
    "h2_vent_mol_s", "co2_vent_mol_s",
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

    def limits(self) -> PlannerLimits:
        """Bounds the planner may not plan outside of.

        The carbonate bound is the *total* inventory, not the deactivated
        capacity: capacity depends on the cycle number, which is itself a state,
        so that limit is imposed as a nonlinear constraint in `planner.py` rather
        than as a box.
        """
        b = self.battery.p
        return PlannerLimits(
            soc=(float(b.soc_min), float(b.soc_max)),
            n_caco3=(0.0, self.n_total_mol),
            cycle_number=(0.0, 1e4),
            n_h2=(float(self.gas.p.h2_min_fraction) * self.h2_capacity_mol,
                  self.h2_capacity_mol),
            n_co2=(float(self.gas.p.co2_min_fraction) * self.co2_capacity_mol,
                   self.co2_capacity_mol),
            kiln_temperature_K=(250.0, float(self.calciner.p.temperature_max_K)),
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
             self.h2_capacity_mol, self.co2_capacity_mol, 1000.0],
            dtype=float,
        )

    def control_scale(self) -> np.ndarray:
        """Characteristic magnitude of each reduced control.

        Setpoints, enables and the curtailment fraction are already fractions of
        one. Only the battery powers (watts) and the vent rates (mol/s) need
        rescaling.
        """
        scale = np.ones(N_U)
        p_max = float(self.battery.max_power_W)
        scale[_IU["battery_charge_W"]] = p_max
        scale[_IU["battery_discharge_W"]] = p_max
        scale[_IU["h2_vent_mol_s"]] = 1.0
        scale[_IU["co2_vent_mol_s"]] = 1.0
        return scale

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
            ],
            dtype=float,
        )

    # --- derived quantities ----------------------------------------------
    def max_conversion(self, cycle_number):
        """Grasa-Abanades sorbent capacity at mean cycle number N."""
        return self.solids.max_conversion(cycle_number)

    def capture_capacity_mol(self, cycle_number):
        return self.n_total_mol * self.max_conversion(cycle_number)

    def loading(self, z):
        capacity = mx.fmax(self.capture_capacity_mol(z[_IZ["cycle_number"]]), 1.0)
        return mx.smooth_clip(z[_IZ["n_caco3"]] / capacity, 0.0, 1.0, eps=1e-4)

    # --- rates ------------------------------------------------------------
    def rates(self, z, u, w: Mapping[str, Any]) -> dict[str, Any]:
        """The four material rates, mol/s.

        How commitment is relaxed, and why it is *not* a factor here
        -----------------------------------------------------------
        A subsystem's rate is set by its setpoint alone. The enable appears only
        in `powers`, where it scales the parasitic load, and in the constraint
        `setpoint <= enable`. That combination is the standard relaxation of an
        on/off decision: the optimiser always drives the enable down to the
        setpoint (a smaller enable means less parasitic draw, and nothing else
        depends on it), so at the solution `e = s` and the parasitic load is a
        linear interpolation of the on/off cost.

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
        sab = self.sabatier
        co2_free = mx.smooth_max(
            z[_IZ["n_co2"]] - self.gas.p.co2_min_fraction * self.co2_capacity_mol,
            0.0, eps=1.0)
        h2_free = mx.smooth_max(
            z[_IZ["n_h2"]] - self.gas.p.h2_min_fraction * self.h2_capacity_mol,
            0.0, eps=1.0)
        availability = mx.smooth_min(
            mx.smooth_clip(co2_free / 50.0, 0.0, 1.0, eps=1e-3),
            mx.smooth_clip(h2_free / 200.0, 0.0, 1.0, eps=1e-3),
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

        return mx.vertcat(
            d_soc,
            r_carb - r_calc,
            r_calc / self.n_total_mol,
            r_h2 - 4.0 * r_sab - u[_IU["h2_vent_mol_s"]],
            r_calc - r_sab - u[_IU["co2_vent_mol_s"]],
            d_temperature,
        )

    def step(self, z, u, w: Mapping[str, Any], dt_s: float, substeps: int = 2):
        """One RK4 step of length `dt_s`, optionally subdivided.

        Two substeps by default. At full heater duty the kiln climbs ~170 K in an
        hour while its own reaction rate changes sharply with temperature, and a
        single RK4 step over 3600 s visibly overshoots the calcination threshold.
        Two halves cost one extra RHS evaluation per stage and remove the error.
        """
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

        # net make-up: electrolysis consumes 1 mol H2O per mol H2, the reactor
        # returns 2 per mol CH4 less condenser losses
        water_kg = (r_h2 - 2.0 * self.gas.p.condensate_recovery * r_sab) * M_H2O * dt_s
        water_cost = (economics.p.water_cost_per_m3
                      * mx.smooth_max(water_kg, 0.0, eps=1e-3) / 1000.0)

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
