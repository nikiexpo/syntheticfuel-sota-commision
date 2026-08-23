"""PV array: irradiance and air temperature in, AC power out.

Algebraic (no states). The physics that matters for *scheduling* is not the
peak rating but the two things that make available power differ from nameplate:

    cell temperature    a hot still day in Seville costs ~8 % of output relative
                        to STC, and it peaks at exactly the hour the kiln most
                        wants power
    inverter clipping   with a DC/AC ratio > 1 the array clips at midday, which
                        is *free* energy to any load that can absorb DC directly

Cell temperature uses the Faiman (2008) model rather than the older NOCT one,
because it is wind-dependent and because PVGIS -- our irradiance source -- uses
the same model. Consistency between the data source and the array model is worth
more here than a marginally more detailed thermal network.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sfp.models import mathx as mx
from sfp.models.base import Signal, Subsystem, VectorSpec
from sfp.units import SECONDS_PER_YEAR


class PVArray(Subsystem):
    """Fixed-tilt PV array with a central inverter.

    The single input is a curtailment fraction in [0, 1]: 0 delivers everything
    available, 1 shuts the array down. Curtailment is not waste to be avoided at
    all costs -- when the battery is full and every load is at its limit, the
    array *must* be backed off, and the energy that goes with it is exactly the
    "energy lost through curtailment" the brief asks us to report.
    """

    name = "pv"

    @property
    def states(self) -> VectorSpec:
        return VectorSpec(())

    @property
    def inputs(self) -> VectorSpec:
        return VectorSpec(
            (
                Signal(
                    "curtail_fraction",
                    "-",
                    "fraction of available AC power deliberately not taken",
                    lower=0.0,
                    upper=1.0,
                ),
            )
        )

    # --- ratings ----------------------------------------------------------
    @property
    def rated_dc_W(self) -> float:
        return self.p.capacity_kwp * 1e3

    @property
    def rated_ac_W(self) -> float:
        """Inverter AC limit, set by the DC/AC ratio."""
        return self.rated_dc_W / self.p.dc_ac_ratio

    def derate_factor(self) -> float:
        """Time-invariant losses: soiling, wiring and age."""
        age = (1.0 - self.p.degradation_per_year) ** self.p.age_years
        return (1.0 - self.p.soiling_loss) * (1.0 - self.p.wiring_loss) * age

    # --- physics ----------------------------------------------------------
    def cell_temperature(self, poa_global, temp_air, wind_speed):
        """Faiman (2008) module temperature, degC.

        T_cell = T_air + G / (u0 + u1 * v_wind)
        """
        denominator = self.p.faiman_u0 + self.p.faiman_u1 * mx.fmax(wind_speed, 0.0)
        return temp_air + poa_global / mx.fmax(denominator, 1.0)

    def available_power(self, w: Mapping[str, Any]):
        """AC power the array could deliver right now, W (before curtailment).

        `w` must carry `poa_global` (W/m^2), `temp_air` (degC) and `wind_speed`
        (m/s). Returns a value already limited by the inverter.
        """
        poa = mx.fmax(w["poa_global"], 0.0)
        t_cell = self.cell_temperature(poa, w["temp_air"], w.get("wind_speed", 1.0))

        # linear temperature derating about STC
        temp_factor = 1.0 + self.p.gamma_power * (t_cell - self.p.temp_ref_C)

        p_dc = (
            self.rated_dc_W
            * (poa / self.p.irradiance_ref)
            * temp_factor
            * self.derate_factor()
        )
        p_dc = mx.fmax(p_dc, 0.0)

        # inverter: constant efficiency then a hard clip at the AC rating
        p_ac = mx.fmin(p_dc * self.p.eta_inverter, self.rated_ac_W)
        return mx.fmax(p_ac, 0.0)

    def rhs(self, t, x, u, w: Mapping[str, Any]):
        return mx.zeros_like_state(x, 0)

    def outputs(self, t, x, u, w: Mapping[str, Any]) -> dict[str, Any]:
        curtail = mx.clip(u[0], 0.0, 1.0)
        available = self.available_power(w)
        delivered = available * (1.0 - curtail)

        poa = mx.fmax(w["poa_global"], 0.0)
        t_cell = self.cell_temperature(poa, w["temp_air"], w.get("wind_speed", 1.0))

        # what the array would have made with no inverter limit, for a clean
        # split between clipping loss and deliberate curtailment
        temp_factor = 1.0 + self.p.gamma_power * (t_cell - self.p.temp_ref_C)
        p_dc = mx.fmax(
            self.rated_dc_W * (poa / self.p.irradiance_ref) * temp_factor * self.derate_factor(),
            0.0,
        )
        clipped = mx.fmax(p_dc * self.p.eta_inverter - self.rated_ac_W, 0.0)

        return {
            # generator: negative because the sign convention is "positive = consumed"
            "power_electrical_W": -delivered,
            "pv_available_W": available,
            "pv_delivered_W": delivered,
            "pv_curtailed_W": available - delivered,
            "pv_clipped_W": clipped,
            "cell_temperature_C": t_cell,
            "poa_global_W_m2": poa,
        }

    def initial_state(self) -> np.ndarray:
        return np.zeros(0)

    def capacity_factor(self, delivered_energy_J: float, duration_s: float) -> float:
        """Delivered energy as a fraction of nameplate x time."""
        if duration_s <= 0.0:
            return 0.0
        return float(delivered_energy_J / (self.rated_dc_W * duration_s))

    def annual_degradation_rate_per_s(self) -> float:
        return self.p.degradation_per_year / SECONDS_PER_YEAR
