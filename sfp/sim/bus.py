"""DC bus reconciliation -- the regulatory layer the controller cannot override.

An off-grid plant has one inviolable algebraic constraint:

    PV delivered + battery discharge  ==  process load + battery charge

A controller -- optimal, rule-based or broken -- hands down a *request*. This
module turns that request into a dispatch that actually balances, by clamping in
a fixed priority order, and records exactly how much it had to intervene. Two
reasons that matters:

1.  It is how a real plant works. The DCS and the interlocks sit below the
    optimiser and are not negotiable. A submission whose "controller" is allowed
    to violate the power balance is not a control system.
2.  The intervention log is a free diagnostic. A good controller should almost
    never be corrected here; if the shed counter is climbing, its model of its
    own plant is wrong.

Priority order, highest first:
    1. parasitic and safety loads (standby, warm-up) -- never shed
    2. the commanded process load
    3. battery charging
    4. curtailment absorbs whatever is left

Curtailment is the residue of this process, not a decision: it is PV that was
available and that nothing could absorb. That is precisely the quantity the
brief asks us to report, and off-grid it has nothing to do with grid limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TOL_W = 1e-6


@dataclass
class Dispatch:
    """A feasible, balanced instantaneous power allocation. All values in W."""

    pv_available_W: float
    pv_used_W: float
    pv_curtailed_W: float
    pv_clipped_W: float
    battery_charge_W: float
    battery_discharge_W: float
    process_power_W: float
    process_load_fraction: float
    process_enable: float = 1.0

    # diagnostics: how far the request had to be moved to become feasible
    load_shed_W: float = 0.0
    charge_denied_W: float = 0.0
    unserved_W: float = 0.0
    tripped: bool = False
    intervened: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def balance_residual_W(self) -> float:
        """Should be zero to numerical precision; asserted by the simulator."""
        supply = self.pv_used_W + self.battery_discharge_W
        demand = self.process_power_W + self.battery_charge_W
        return supply - demand

    def as_dict(self) -> dict[str, float]:
        return {
            "pv_available_W": self.pv_available_W,
            "pv_used_W": self.pv_used_W,
            "pv_curtailed_W": self.pv_curtailed_W,
            "pv_clipped_W": self.pv_clipped_W,
            "battery_charge_W": self.battery_charge_W,
            "battery_discharge_W": self.battery_discharge_W,
            "process_power_W": self.process_power_W,
            "process_load_fraction": self.process_load_fraction,
            "process_enable": self.process_enable,
            "load_shed_W": self.load_shed_W,
            "charge_denied_W": self.charge_denied_W,
            "unserved_W": self.unserved_W,
            "bus_tripped": float(self.tripped),
            "bus_intervened": float(self.intervened),
        }


@dataclass
class Request:
    """What the controller would like to happen, before feasibility is checked."""

    load_fraction: float = 0.0
    battery_charge_W: float = 0.0
    battery_discharge_W: float = 0.0
    curtail_fraction: float = 0.0
    enable: float = 1.0


def reconcile(
    request: Request,
    *,
    pv_available_W: float,
    pv_clipped_W: float,
    process,
    process_state: np.ndarray,
    battery,
    battery_state: np.ndarray,
    dt_s: float,
) -> Dispatch:
    """Turn a controller request into a balanced dispatch.

    `process` and `battery` are the subsystem models; their states are needed
    because both the achievable battery power and the process overheads depend
    on where the plant currently is.
    """
    notes: list[str] = []
    intervened = False
    tripped = False

    charge_limit = float(battery.max_charge_power_W(battery_state, dt_s))
    discharge_limit = float(battery.max_discharge_power_W(battery_state, dt_s))

    load = float(np.clip(request.load_fraction, 0.0, 1.0))
    enable = 1.0 if float(request.enable) >= 0.5 else 0.0
    if 0.0 < load < process.p.min_load_fraction:
        load = 0.0

    requested_load = load
    pv_offer = max(pv_available_W * (1.0 - float(np.clip(request.curtail_fraction, 0.0, 1.0))), 0.0)
    charge_request = max(request.battery_charge_W, 0.0)

    load_shed_W = 0.0
    charge_denied_W = 0.0
    unserved_W = 0.0

    # Trip check first. If the bus cannot even carry the parasitic load, the
    # plant goes dark -- it does not run up an "unserved" debt drawing power
    # that is not there. A dark plant draws nothing and coasts down in
    # temperature, which is what actually happens when a remote site runs out
    # of stored energy at 3 a.m.
    if enable >= 0.5:
        parasitic_W = float(process.parasitic_power_W(process_state, 1.0))
        if parasitic_W > pv_offer + discharge_limit + TOL_W:
            enable = 0.0
            load = 0.0
            requested_load = 0.0
            tripped = True
            intervened = True
            notes.append("undervoltage trip: available power below parasitic load")
    else:
        load = 0.0
        requested_load = 0.0

    # --- stage 1: settle on a load the bus can actually carry ---------------
    # Fixed-point iteration, because reducing the load also reduces the warm-up
    # overhead, which frees a little more power. It converges in two or three
    # passes; the bound is a safety net, and falling through it means the load
    # is simply zero.
    supply_ceiling = pv_offer + discharge_limit
    for _ in range(4):
        process_W = float(process.power_for_load(process_state, load, enable))
        if process_W <= supply_ceiling + TOL_W:
            break
        reduced = process.load_for_power(process_state, supply_ceiling, enable)
        if reduced >= load - 1e-9:
            load = 0.0
            intervened = True
            break
        load = reduced
        intervened = True

    # --- stage 2: allocate supply against that load, once -------------------
    process_W = float(process.power_for_load(process_state, load, enable))
    # The subsystem models use smooth gates so that the same expressions can be
    # differentiated inside the NMPC. A smooth gate never reaches exactly zero,
    # so a fully stopped plant reports a few microwatts. That is a modelling
    # artefact, not demand, and letting it through would show up as a permanent
    # sliver of "unserved" load on every dark night. Below a milliwatt the plant
    # is off.
    if process_W < 1e-3:
        process_W = 0.0

    if process_W <= pv_offer + TOL_W:
        # Solar covers the process. Surplus may charge the battery, up to what
        # the controller asked for -- if it asked for less, the rest is curtailed
        # on purpose, which is a legitimate choice when the degradation cost of a
        # cycle exceeds the value of the stored energy.
        pv_used = process_W
        surplus = pv_offer - process_W
        charge = min(charge_request, surplus, charge_limit)
        # Capping the charge by surplus and by the pack is routine physics, not
        # the controller asking for something impossible, so it is not counted
        # as an intervention.
        if charge_request > charge + TOL_W:
            charge_denied_W = charge_request - charge
        pv_used += charge
        discharge = 0.0
    else:
        # Solar is short: draw the deficit from the battery.
        pv_used = pv_offer
        deficit = process_W - pv_used
        discharge = min(deficit, discharge_limit)
        charge = 0.0
        if charge_request > TOL_W:
            charge_denied_W = charge_request
        # Anything still uncovered is genuinely unserved. After stage 1 this can
        # only be residual parasitic load on a plant that is already dark.
        unserved_W = max(deficit - discharge, 0.0)

    if requested_load > load + 1e-9:
        load_shed_W = (requested_load - load) * process.rated_power_W
        notes.append(f"load shed from {requested_load:.3f} to {load:.3f}")
        intervened = True

    curtailed = max(pv_available_W - pv_used, 0.0)

    dispatch = Dispatch(
        pv_available_W=pv_available_W,
        pv_used_W=pv_used,
        pv_curtailed_W=curtailed,
        pv_clipped_W=pv_clipped_W,
        battery_charge_W=charge,
        battery_discharge_W=discharge,
        process_power_W=process_W,
        process_load_fraction=load,
        process_enable=enable,
        load_shed_W=load_shed_W,
        charge_denied_W=charge_denied_W,
        unserved_W=unserved_W,
        tripped=tripped,
        intervened=intervened,
        notes=notes,
    )

    residual = dispatch.balance_residual_W
    if abs(residual) > max(1e-3, 1e-9 * max(pv_available_W, 1.0)) and unserved_W <= TOL_W:
        raise AssertionError(
            f"bus reconciliation failed to balance: residual {residual:.6e} W "
            f"(pv_used={pv_used:.3f}, dis={discharge:.3f}, load={process_W:.3f}, ch={charge:.3f})"
        )

    return dispatch
