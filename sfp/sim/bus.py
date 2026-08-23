"""DC bus reconciliation -- the regulatory layer no controller can override.

One inviolable algebraic constraint:

    PV delivered + battery discharge  ==  total plant load + battery charge

A controller -- optimal, rule-based or broken -- hands down a `Request` of
setpoints. This module turns it into a `Dispatch` that actually balances, and
records how far it had to move the request to get there.

What changed at M1
------------------
There is no longer a single load. Four subsystems compete for the same bus, and
they are not interchangeable, so when supply falls short the bus sheds in a fixed
priority order rather than scaling everything down proportionally:

    shed first   calciner      biggest and most deferrable -- CaCO3 keeps
                 contactor     cheap, and the sorbent will still be there later
                 electrolyser  flexible, but its H2 feeds the reactor
    shed last    sabatier      protects product in progress and the catalyst

That ordering is a plant-engineering judgement, not an optimisation: the bus's
job is to keep the plant alive and feasible, and the *controller's* job is to
make sure the bus never has to intervene. A rising intervention count means the
controller's model of its own plant is wrong.

Two subtleties worth knowing
----------------------------
**Turning the Sabatier feed down saves nothing.** Its draw is preheat plus
auxiliaries, essentially independent of feed rate (8 kW at any feed). The only
way to shed it is to shut it off, which costs a 31 kWh relight later. The bus
therefore treats it as all-or-nothing.

**The CO2 compressor is not commanded.** Its draw follows the calcination rate,
so it cannot be sized directly. The bus allocates the controllable subsystems
using their own power curves, then evaluates the *whole plant* to get the exact
total including the compressor, and repeats if that pushed it over budget. That
is why `reconcile` needs the plant and its state rather than just a few numbers.

Curtailment remains a **residue, not a decision**: `pv_available - pv_used`. It is
PV that was available and that nothing could absorb. Off-grid that has nothing to
do with grid limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

TOL_W = 1e-6

#: Order in which load is given up when supply falls short. First entry sheds first.
SHED_ORDER: tuple[str, ...] = ("calciner", "contactor", "electrolyser", "sabatier")

#: Subsystems the controller commands directly. Others (buffers, inventories)
#: either draw nothing or draw as a consequence of these.
CONTROLLABLE: tuple[str, ...] = SHED_ORDER

#: Subsystems that are not loads at all.
NON_LOADS: frozenset[str] = frozenset({"pv", "battery"})


@dataclass
class Request:
    """What the controller would like to happen, before feasibility is checked."""

    setpoints: dict[str, float] = field(default_factory=dict)
    enables: dict[str, float] = field(default_factory=dict)
    battery_charge_W: float = 0.0
    battery_discharge_W: float = 0.0
    curtail_fraction: float = 0.0

    def setpoint(self, key: str) -> float:
        return float(np.clip(self.setpoints.get(key, 0.0), 0.0, 1.0))

    def enable(self, key: str) -> float:
        return 1.0 if float(self.enables.get(key, 1.0)) >= 0.5 else 0.0

    @classmethod
    def all_off(cls) -> "Request":
        return cls(
            setpoints={k: 0.0 for k in CONTROLLABLE},
            enables={k: 0.0 for k in CONTROLLABLE},
        )

    @classmethod
    def uniform(cls, setpoint: float, enable: float = 1.0, **kwargs) -> "Request":
        return cls(
            setpoints={k: setpoint for k in CONTROLLABLE},
            enables={k: enable for k in CONTROLLABLE},
            **kwargs,
        )


@dataclass
class Dispatch:
    """A feasible, balanced instantaneous power allocation. All powers in W."""

    pv_available_W: float
    pv_used_W: float
    pv_curtailed_W: float
    pv_clipped_W: float
    battery_charge_W: float
    battery_discharge_W: float
    setpoints: dict[str, float] = field(default_factory=dict)
    enables: dict[str, float] = field(default_factory=dict)
    loads_W: dict[str, float] = field(default_factory=dict)
    total_load_W: float = 0.0

    # diagnostics
    shed_W: float = 0.0
    charge_denied_W: float = 0.0
    unserved_W: float = 0.0
    tripped: bool = False
    tripped_subsystems: tuple[str, ...] = ()
    intervened: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def balance_residual_W(self) -> float:
        supply = self.pv_used_W + self.battery_discharge_W
        demand = self.total_load_W + self.battery_charge_W
        return supply - demand

    def as_dict(self) -> dict[str, float]:
        out = {
            "pv_available_W": self.pv_available_W,
            "pv_used_W": self.pv_used_W,
            "pv_curtailed_W": self.pv_curtailed_W,
            "pv_clipped_W": self.pv_clipped_W,
            "battery_charge_W": self.battery_charge_W,
            "battery_discharge_W": self.battery_discharge_W,
            "total_load_W": self.total_load_W,
            "shed_W": self.shed_W,
            "charge_denied_W": self.charge_denied_W,
            "unserved_W": self.unserved_W,
            "bus_tripped": float(self.tripped),
            "bus_intervened": float(self.intervened),
            "n_tripped": float(len(self.tripped_subsystems)),
        }
        for key in CONTROLLABLE:
            out[f"setpoint_{key}"] = self.setpoints.get(key, 0.0)
            out[f"enable_{key}"] = self.enables.get(key, 0.0)
        return out


def _plant_load_W(plant, t, x, setpoints, enables, w) -> tuple[float, dict[str, float]]:
    """Exact total plant load for a candidate dispatch, including the compressor.

    Evaluates the whole coupled plant rather than summing per-subsystem power
    curves, because some draws (the CO2 compressor most of all) follow reaction
    rates rather than commands.
    """
    u = _inputs_from(plant, setpoints, enables)
    powers = plant.electrical_powers_W(t, x, u, w)
    loads = {k: v for k, v in powers.items() if k not in NON_LOADS}
    return float(sum(loads.values())), loads


def _inputs_from(plant, setpoints, enables) -> dict[str, np.ndarray]:
    """Build the plant input mapping from setpoint/enable dictionaries."""
    u: dict[str, np.ndarray] = {}
    for key, sub in plant:
        if key == "pv":
            u[key] = np.array([0.0])
        elif key == "battery":
            u[key] = np.array([0.0, 0.0])
        elif sub.n_inputs == 0:
            u[key] = np.zeros(0)
        else:
            u[key] = np.array([setpoints.get(key, 0.0), enables.get(key, 0.0)], dtype=float)
    return u


def reconcile(
    request: Request,
    *,
    plant,
    state: np.ndarray,
    weather: Mapping[str, Any],
    pv_available_W: float,
    pv_clipped_W: float,
    dt_s: float,
    t: float = 0.0,
) -> Dispatch:
    """Turn a controller request into a balanced dispatch."""
    notes: list[str] = []
    intervened = False
    tripped_subsystems: list[str] = []

    battery = plant["battery"]
    states = plant.split(state)
    battery_state = states["battery"]

    charge_limit = float(battery.max_charge_power_W(battery_state, dt_s))
    physical_discharge_limit = float(battery.max_discharge_power_W(battery_state, dt_s))

    # The controller's requested discharge is a *cap*, not a demand. Honouring it
    # is what lets a controller hold an evening reserve: it says "you may draw at
    # most this much from the pack", and the bus sheds load rather than digging
    # deeper. Ignoring it -- as an earlier version did, discharging whatever the
    # deficit required -- silently defeats every reserve policy a controller
    # might have, and the plant then blacks out at 3 a.m. having spent the
    # battery on the afternoon's electrolyser.
    discharge_limit = min(physical_discharge_limit, max(request.battery_discharge_W, 0.0))

    pv_offer = max(pv_available_W * (1.0 - float(np.clip(request.curtail_fraction, 0.0, 1.0))), 0.0)
    charge_request = max(request.battery_charge_W, 0.0)
    supply_ceiling = pv_offer + discharge_limit

    enables = {k: request.enable(k) for k in CONTROLLABLE}
    requested = {k: request.setpoint(k) for k in CONTROLLABLE}

    def parasitic_total(active: dict[str, float]) -> float:
        total = 0.0
        for key in CONTROLLABLE:
            if active[key] < 0.5 or key not in plant:
                continue
            total += float(plant[key].parasitic_power_W(states[key], 1.0, weather))
        return total

    # --- stage 1: trip subsystems until the parasitic floor fits ------------
    for key in SHED_ORDER:
        if parasitic_total(enables) <= supply_ceiling + TOL_W:
            break
        if enables.get(key, 0.0) < 0.5:
            continue
        enables[key] = 0.0
        requested[key] = 0.0
        tripped_subsystems.append(key)
        intervened = True
    if tripped_subsystems:
        notes.append(f"undervoltage trip: shut down {', '.join(tripped_subsystems)}")

    # --- stage 2: allocate the remaining budget in priority order ----------
    setpoints = {k: 0.0 for k in CONTROLLABLE}
    budget = supply_ceiling - parasitic_total(enables)

    for key in reversed(SHED_ORDER):  # highest priority first
        if enables[key] < 0.5 or key not in plant:
            continue
        sub = plant[key]
        x_sub = states[key]
        floor = float(sub.parasitic_power_W(x_sub, 1.0, weather))
        want = requested[key]
        want_power = float(sub.power_for_setpoint(x_sub, want, 1.0, weather))
        incremental = max(want_power - floor, 0.0)

        granted = min(incremental, max(budget, 0.0))
        if granted >= incremental - TOL_W:
            chosen = want
        else:
            chosen = sub.setpoint_for_power(
                x_sub, floor + granted, 1.0, weather, min_setpoint=sub.min_setpoint
            )
            intervened = True
        if 0.0 < chosen < sub.min_setpoint:
            chosen = 0.0
        setpoints[key] = chosen
        budget -= max(
            float(sub.power_for_setpoint(x_sub, chosen, 1.0, weather)) - floor, 0.0
        )

    # --- stage 3: exact total, then settle the supply side -----------------
    unserved_W = 0.0
    for _ in range(3):
        total_load, loads = _plant_load_W(plant, t, state, setpoints, enables, weather)
        if total_load <= supply_ceiling + 1e-3:
            break
        # the uncommanded compressor pushed us over; shed the next subsystem
        for key in SHED_ORDER:
            if enables[key] >= 0.5 and setpoints[key] > 0.0:
                setpoints[key] = 0.0
                intervened = True
                notes.append(f"secondary shed: {key} to zero to cover uncommanded load")
                break
        else:
            break
    else:  # pragma: no cover - three passes always suffice in practice
        pass

    total_load, loads = _plant_load_W(plant, t, state, setpoints, enables, weather)
    if total_load < 1e-3:
        total_load = 0.0

    if total_load <= pv_offer + TOL_W:
        pv_used = total_load
        surplus = pv_offer - total_load
        charge = min(charge_request, surplus, charge_limit)
        if charge_request > charge + TOL_W:
            charge_denied_W = charge_request - charge
        else:
            charge_denied_W = 0.0
        pv_used += charge
        discharge = 0.0
    else:
        pv_used = pv_offer
        deficit = total_load - pv_used
        discharge = min(deficit, discharge_limit)
        charge = 0.0
        charge_denied_W = charge_request if charge_request > TOL_W else 0.0
        unserved_W = max(deficit - discharge, 0.0)

    shed_W = 0.0
    for key in CONTROLLABLE:
        if requested[key] > setpoints[key] + 1e-9 and key in plant:
            shed_W += max(
                float(plant[key].power_for_setpoint(states[key], requested[key], 1.0, weather))
                - float(plant[key].power_for_setpoint(states[key], setpoints[key], enables[key], weather)),
                0.0,
            )
    if shed_W > TOL_W:
        intervened = True

    curtailed = max(pv_available_W - pv_used, 0.0)

    dispatch = Dispatch(
        pv_available_W=pv_available_W,
        pv_used_W=pv_used,
        pv_curtailed_W=curtailed,
        pv_clipped_W=pv_clipped_W,
        battery_charge_W=charge,
        battery_discharge_W=discharge,
        setpoints=setpoints,
        enables=enables,
        loads_W=loads,
        total_load_W=total_load,
        shed_W=shed_W,
        charge_denied_W=charge_denied_W,
        unserved_W=unserved_W,
        tripped=bool(tripped_subsystems),
        tripped_subsystems=tuple(tripped_subsystems),
        intervened=intervened,
        notes=notes,
    )

    residual = dispatch.balance_residual_W
    if abs(residual) > max(1e-3, 1e-9 * max(pv_available_W, 1.0)) and unserved_W <= TOL_W:
        raise AssertionError(
            f"bus reconciliation failed to balance: residual {residual:.6e} W "
            f"(pv_used={pv_used:.3f}, dis={discharge:.3f}, load={total_load:.3f}, ch={charge:.3f})"
        )

    return dispatch
