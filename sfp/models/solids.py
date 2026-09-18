"""Circulating calcium sorbent inventory -- the plant's cheapest and largest buffer.

    carbonation   CaO   + CO2 -> CaCO3      ambient, exothermic, fans only
    calcination   CaCO3 -> CaO + CO2        900 degC, endothermic, enormous

The inventory between those two reactions is captured CO2 held as a solid, and
**holding it costs nothing** -- no pressure vessel, no self-discharge, no
round-trip loss, against the battery's 0.021 EUR/kWh of throughput wear. The
calcium loop, not the battery, is the buffer that decides whether the plant
survives a cloudy week.

This subsystem owns the inventory and nothing else. The rates that move material
through it come from the contactor and the calciner as coupling signals, so
`rhs` reads them out of `w` rather than from its own inputs.

Sorbent capacity falls with cycle number (Grasa & Abanades 2006):

    X_N = 1 / (k*N + 1/(1 - X_r)) + X_r        X_r ~ 0.075,  k ~ 0.52

X_0 = 1.0 falling to ~0.16 by cycle 20, so **every calcination permanently
destroys some capture capacity** and calcining is never free even when
electricity is. To keep that differentiable, the cycle counter is a
*continuous* state advanced by calcination throughput:

    dN/dt = r_calcination / n_total
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sfp.models import mathx as mx
from sfp.models.base import Signal, Subsystem, VectorSpec


class SolidsInventory(Subsystem):
    """CaO / CaCO3 inventory with cycle-dependent capacity decay."""

    name = "solids"

    @property
    def states(self) -> VectorSpec:
        n_tot = self.p.n_total_mol
        return VectorSpec(
            (
                Signal("n_cao", "mol", "calcium oxide available to carbonate",
                       lower=0.0, upper=n_tot),
                Signal("n_caco3", "mol", "calcium carbonate holding captured CO2",
                       lower=0.0, upper=n_tot),
                Signal("cycle_number", "cycles", "mean carbonation/calcination cycle count",
                       lower=0.0, upper=1e4),
            )
        )

    # --- coupling ---------------------------------------------------------
    @property
    def requires_for_rhs(self) -> tuple[str, ...]:
        return ("r_carbonation_mol_s", "r_calcination_mol_s")

    @property
    def provides(self) -> tuple[str, ...]:
        return ("solids_loading", "solids_n_caco3_mol", "solids_n_cao_mol",
                "solids_max_conversion", "solids_available_cao_mol")

    @property
    def inputs(self) -> VectorSpec:
        # Driven entirely by the contactor and calciner; no manipulated variables
        # of its own. An inventory is a consequence, not a decision.
        return VectorSpec(())

    # --- capacity ---------------------------------------------------------
    def max_conversion(self, cycle_number):
        """Grasa-Abanades maximum CaO conversion at mean cycle number N."""
        x_r = self.p.grasa_residual_conversion
        k = self.p.grasa_deactivation_constant
        n = mx.fmax(cycle_number, 0.0)
        return 1.0 / (k * n + 1.0 / (1.0 - x_r)) + x_r

    def capture_capacity_mol(self, x):
        """Total CO2 the inventory could still hold at its present activity, mol."""
        return self.p.n_total_mol * self.max_conversion(x[2])

    def loading(self, x):
        """Fractional saturation of the sorbent, 0 (fully regenerated) to 1 (spent).

        This is the state of charge of the chemical battery. It is what the air
        contactor's rate tapers against and what the planner will hold a target on.
        """
        capacity = mx.fmax(self.capture_capacity_mol(x), 1.0)
        return mx.smooth_clip(x[1] / capacity, 0.0, 1.0, eps=1e-4)

    def available_cao_mol(self, x):
        """CaO that can still take up CO2 this cycle, mol."""
        return mx.fmax(self.capture_capacity_mol(x) - x[1], 0.0)

    # --- dynamics ---------------------------------------------------------
    def rhs(self, t, x, u, w: Mapping[str, Any]):
        r_carb = w.get("r_carbonation_mol_s", 0.0)
        r_calc = w.get("r_calcination_mol_s", 0.0)

        makeup = self.p.makeup_rate_mol_s
        purge = self.p.purge_fraction * r_calc

        d_cao = r_calc - r_carb + makeup - purge
        d_caco3 = r_carb - r_calc

        # Continuous cycle counter: one full pass of the whole inventory through
        # the kiln advances the mean cycle number by exactly one. Fresh make-up
        # dilutes the mean age, which is how a purge/make-up policy would cap
        # deactivation in a real plant.
        n_tot = mx.fmax(self.p.n_total_mol, 1.0)
        d_cycles = r_calc / n_tot - makeup * mx.fmax(x[2], 0.0) / n_tot

        return mx.vertcat(d_cao, d_caco3, d_cycles)

    def outputs(self, t, x, u, w: Mapping[str, Any]) -> dict[str, Any]:
        conversion = self.max_conversion(x[2])
        capacity = self.capture_capacity_mol(x)
        return {
            "power_electrical_W": 0.0,
            "solids_n_cao_mol": x[0],
            "solids_n_caco3_mol": x[1],
            "solids_cycle_number": x[2],
            "solids_max_conversion": conversion,
            "solids_capture_capacity_mol": capacity,
            "solids_loading": self.loading(x),
            "solids_available_cao_mol": self.available_cao_mol(x),
            "solids_total_mol": x[0] + x[1],
            # captured CO2 currently banked in solid form, expressed as the
            # methane it could eventually become (1 mol CO2 -> 1 mol CH4)
            "solids_stored_co2_kg": x[1] * 44.0095e-3,
        }

    def initial_state(self) -> np.ndarray:
        n_tot = self.p.n_total_mol
        n_caco3 = self.p.n_caco3_initial_fraction * n_tot
        return np.array([n_tot - n_caco3, n_caco3, self.p.cycle_number_initial], dtype=float)

    # --- economics --------------------------------------------------------
    def marginal_deactivation_cost_per_mol(self, x, sorbent_capex_per_mol: float = 0.0) -> float:
        """Cost of the capacity permanently lost by calcining one more mole.

        dX/dN < 0, so each calcination shrinks the capacity of the *entire*
        inventory a little. The controller pays this per mole calcined, which is
        what makes it reluctant to cycle the loop harder than the methane is
        worth. Returned per mole of CaCO3 calcined.
        """
        n_tot = self.p.n_total_mol
        x_r = self.p.grasa_residual_conversion
        k = self.p.grasa_deactivation_constant
        n = float(np.maximum(x[2], 0.0))
        denominator = k * n + 1.0 / (1.0 - x_r)
        # d(X_N)/dN
        dx_dn = -k / (denominator**2)
        # one mole calcined advances N by 1/n_tot, and the capacity lost is
        # n_tot * dX -- so the n_tot cancels
        capacity_lost_mol = -dx_dn
        return capacity_lost_mol * sorbent_capex_per_mol
