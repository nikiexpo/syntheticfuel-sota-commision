"""Gas buffers and the water balance -- the inventories that decouple the plant.

Two subsystems here, both pure accumulators with no manipulated variables of
their own. Everything that moves material through them is computed by the
producing and consuming subsystems and arrives as coupling signals through `w`
(see `sfp.sim.plant.Plant` for the two-phase evaluation that makes that work).

Why the sizes are what they are
-------------------------------
The gas buffers are deliberately modest -- about 6-8 hours each -- while the
CaCO3 inventory in `SolidsInventory` holds roughly 12 hours of CO2 demand. That
is not an accident of tuning. A mole of CO2 stored as calcium carbonate sits in
an unpressurised pile of rock and costs essentially nothing to hold; the same
mole stored as gas needs a pressure vessel. Making the gas buffers large would
quietly solve the plant's intermittency problem with capital instead of control,
and would hide the thing this project is about.

The hydrogen tank is the one buffer that must be generous, because it is what
lets the Sabatier reactor run at night on hydrogen made at noon.

Water
-----
Easy to forget and genuinely binding at an arid site. Electrolysis consumes
4 mol H2O per mol CH4 eventually produced; the Sabatier reaction gives 2 of them
back. With 95 % condensate recovery the net is about 2.36 kg of water per kg of
methane -- delivered by road at a remote site, so it carries a real cost.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sfp.models import mathx as mx
from sfp.models.base import Signal, Subsystem, VectorSpec
from sfp.units import M_CH4, M_CO2, M_H2, M_H2O


class GasBuffer(Subsystem):
    """Hydrogen and carbon-dioxide inventories between the reactors."""

    name = "gas"

    @property
    def states(self) -> VectorSpec:
        return VectorSpec(
            (
                Signal("n_h2", "mol", "hydrogen inventory",
                       lower=0.0, upper=self.p.h2_capacity_mol),
                Signal("n_co2", "mol", "carbon dioxide inventory",
                       lower=0.0, upper=self.p.co2_capacity_mol),
            )
        )

    # --- coupling ---------------------------------------------------------
    @property
    def requires(self) -> tuple[str, ...]:
        # `outputs` needs both inflow rates: the calciner's for the compressor
        # draw, the electrolyser's for the venting accounting. Declaring the
        # electrolyser here is what forces it to be registered *before* the gas
        # buffer -- without the declaration the venting term silently evaluated
        # against a zero inflow and 48 % of the hydrogen left the mass balance
        # with nothing to attribute it to.
        return ("r_calcination_mol_s", "r_electrolysis_h2_mol_s")

    @property
    def requires_for_rhs(self) -> tuple[str, ...]:
        return ("r_electrolysis_h2_mol_s", "r_calcination_mol_s", "r_sabatier_co2_mol_s")

    @property
    def provides(self) -> tuple[str, ...]:
        return ("gas_h2_available_mol", "gas_co2_available_mol", "gas_h2_fill", "gas_co2_fill")

    @property
    def inputs(self) -> VectorSpec:
        return VectorSpec(())

    # --- levels -----------------------------------------------------------
    def h2_fill(self, x):
        return mx.smooth_clip(x[0] / mx.fmax(self.p.h2_capacity_mol, 1.0), 0.0, 1.0, eps=1e-4)

    def co2_fill(self, x):
        return mx.smooth_clip(x[1] / mx.fmax(self.p.co2_capacity_mol, 1.0), 0.0, 1.0, eps=1e-4)

    def h2_available_mol(self, x):
        """Hydrogen above the cushion, i.e. actually drawable."""
        return mx.fmax(x[0] - self.p.h2_min_fraction * self.p.h2_capacity_mol, 0.0)

    def co2_available_mol(self, x):
        return mx.fmax(x[1] - self.p.co2_min_fraction * self.p.co2_capacity_mol, 0.0)

    def h2_headroom_mol(self, x):
        """Room left before the tank is full -- the electrolyser must stop here."""
        return mx.fmax(self.p.h2_capacity_mol - x[0], 0.0)

    def co2_headroom_mol(self, x):
        return mx.fmax(self.p.co2_capacity_mol - x[1], 0.0)

    # --- acceptance gates -------------------------------------------------
    def _taper_mol(self, capacity_mol: float) -> float:
        """Headroom over which inflow is tapered off, mol.

        Two failure modes have to be avoided at once, and they pull in opposite
        directions:

        *Too narrow* and a single timestep overshoots the tank bound before the
        gate has closed. `Plant.clip_state` then truncates the state and the
        excess simply vanishes -- silent mass destruction that no balance check
        attributes to anything. A 20 mol taper against a 48 mol-per-step inflow
        lost 48 % of the hydrogen this way.

        *Too wide* and the gate is still noticeably open at zero headroom, so a
        genuinely full tank keeps absorbing.

        1 % of capacity, floored at 100 mol, is several steps' worth of inflow at
        the reference rates while remaining a small fraction of either tank.
        """
        return max(0.01 * capacity_mol, 100.0)

    def _accept_gates(self, x):
        """Fraction of incoming gas each tank can still accept, in [0, 1]."""
        h2_taper = self._taper_mol(self.p.h2_capacity_mol)
        co2_taper = self._taper_mol(self.p.co2_capacity_mol)
        h2_accept = mx.smooth_clip(self.h2_headroom_mol(x) / h2_taper, 0.0, 1.0, eps=1e-3)
        co2_accept = mx.smooth_clip(self.co2_headroom_mol(x) / co2_taper, 0.0, 1.0, eps=1e-3)
        return h2_accept, co2_accept

    # --- dynamics ---------------------------------------------------------
    def rhs(self, t, x, u, w: Mapping[str, Any]):
        h2_in = w.get("r_electrolysis_h2_mol_s", 0.0)
        co2_in = w.get("r_calcination_mol_s", 0.0)
        co2_out = w.get("r_sabatier_co2_mol_s", 0.0)
        # stoichiometry: CO2 + 4 H2 -> CH4 + 2 H2O
        h2_out = 4.0 * co2_out

        h2_accept, co2_accept = self._accept_gates(x)

        d_h2 = h2_in * h2_accept - h2_out
        d_co2 = co2_in * co2_accept - co2_out
        return mx.vertcat(d_h2, d_co2)

    def venting_rates_mol_s(self, x, w: Mapping[str, Any]) -> tuple[Any, Any]:
        """Gas produced that the tanks cannot accept, and which is therefore lost.

        A full tank forces the producer to vent (or the plant to trip). Either
        way the molecules are gone, and the energy that made them is wasted. This
        is tracked explicitly rather than allowed to disappear from the mass
        balance, because it is one of the most damning diagnostics available: a
        power-follow controller can vent nearly half the hydrogen it makes,
        having paid ~56 kWh/kg to make it, purely because it never looked at the
        tank level.
        """
        h2_in = w.get("r_electrolysis_h2_mol_s", 0.0)
        co2_in = w.get("r_calcination_mol_s", 0.0)
        h2_accept, co2_accept = self._accept_gates(x)
        return h2_in * (1.0 - h2_accept), co2_in * (1.0 - co2_accept)

    def outputs(self, t, x, u, w: Mapping[str, Any]) -> dict[str, Any]:
        co2_in = w.get("r_calcination_mol_s", 0.0)
        compressor_W = co2_in * self.p.co2_compressor_kJ_per_mol * 1e3
        h2_vent, co2_vent = self.venting_rates_mol_s(x, w)

        return {
            "power_electrical_W": compressor_W,
            "gas_h2_vented_mol_s": h2_vent,
            "gas_co2_vented_mol_s": co2_vent,
            "gas_n_h2_mol": x[0],
            "gas_n_co2_mol": x[1],
            "gas_h2_fill": self.h2_fill(x),
            "gas_co2_fill": self.co2_fill(x),
            "gas_h2_kg": x[0] * M_H2,
            "gas_co2_kg": x[1] * M_CO2,
            "gas_h2_available_mol": self.h2_available_mol(x),
            "gas_co2_available_mol": self.co2_available_mol(x),
            "gas_compressor_W": compressor_W,
            # how many mol of CH4 the tanks could make right now, stoichiometry-limited
            "gas_ch4_potential_mol": mx.fmin(
                self.co2_available_mol(x), self.h2_available_mol(x) / 4.0
            ),
        }

    def initial_state(self) -> np.ndarray:
        return np.array(
            [
                self.p.h2_initial_fraction * self.p.h2_capacity_mol,
                self.p.co2_initial_fraction * self.p.co2_capacity_mol,
            ],
            dtype=float,
        )

    def stoichiometric_balance(self, x) -> float:
        """H2:CO2 ratio against the required 4:1.

        Greater than 1 means hydrogen-rich, less than 1 carbon-rich. This is the
        coupling the plan warns about: the two upstream chains must be
        co-scheduled or one of them fills a tank that nothing can drain.
        """
        co2 = float(self.co2_available_mol(x))
        h2 = float(self.h2_available_mol(x))
        if co2 <= 1e-9:
            return float("inf") if h2 > 1e-9 else 1.0
        return (h2 / 4.0) / co2


class WaterTank(Subsystem):
    """Demineralised water inventory: consumed by electrolysis, partly recovered."""

    name = "water"

    @property
    def states(self) -> VectorSpec:
        return VectorSpec(
            (
                Signal("mass_kg", "kg", "water inventory",
                       lower=0.0, upper=self.p.water_capacity_kg),
                Signal("consumed_kg", "kg", "cumulative net make-up drawn", lower=0.0),
            )
        )

    # --- coupling ---------------------------------------------------------
    @property
    def requires_for_rhs(self) -> tuple[str, ...]:
        return ("r_electrolysis_water_mol_s", "r_sabatier_co2_mol_s")

    @property
    def provides(self) -> tuple[str, ...]:
        return ("water_available_kg", "water_fill")

    @property
    def inputs(self) -> VectorSpec:
        return VectorSpec(())

    def fill(self, x):
        return mx.smooth_clip(x[0] / mx.fmax(self.p.water_capacity_kg, 1.0), 0.0, 1.0, eps=1e-4)

    def available_kg(self, x):
        return mx.fmax(x[0] - self.p.water_min_fraction * self.p.water_capacity_kg, 0.0)

    def rhs(self, t, x, u, w: Mapping[str, Any]):
        consumed = w.get("r_electrolysis_water_mol_s", 0.0) * M_H2O
        produced = 2.0 * w.get("r_sabatier_co2_mol_s", 0.0) * M_H2O
        recovered = produced * self.p.condensate_recovery

        d_mass = self.p.water_makeup_rate_kg_s + recovered - consumed
        d_consumed = mx.fmax(consumed - recovered, 0.0)
        return mx.vertcat(d_mass, d_consumed)

    def outputs(self, t, x, u, w: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "power_electrical_W": 0.0,
            "water_mass_kg": x[0],
            "water_fill": self.fill(x),
            "water_available_kg": self.available_kg(x),
            "water_consumed_kg": x[1],
        }

    def initial_state(self) -> np.ndarray:
        return np.array(
            [self.p.water_initial_fraction * self.p.water_capacity_kg, 0.0], dtype=float
        )

    def specific_consumption_kg_per_kg_ch4(self) -> float:
        """Net make-up water per kg of methane, after condensate recovery."""
        per_mol_ch4 = (4.0 - 2.0 * self.p.condensate_recovery) * M_H2O
        return per_mol_ch4 / M_CH4
