"""Site weather bound to an array geometry, resampled to the simulation timestep.

Hourly weather has to reach a one-minute simulator somehow. Interpolating raw
irradiance linearly is the obvious way and it is wrong: it puts nonzero sun
before sunrise and smears the sharp shoulders of the day, which matters because
sunrise is exactly when the controller decides whether to commit the kiln.

Instead we interpolate the two *dimensionless* quantities that genuinely vary
smoothly -- the clear-sky index k = GHI/GHI_clear and the diffuse fraction
DHI/GHI -- and rebuild irradiance at the fine timestep from an exactly computed
clear-sky curve. Sunrise and sunset then land at the right minute for free.

The same object serves the controller as a forecast: `forecast()` returns a view
with configurable error, so the planner can be fed something that is not the
truth. Feeding a controller the exact future is the single easiest way to make a
plant look autonomous when it is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sfp.weather import clearsky


@dataclass
class Site:
    """Where the plant is and how the array is pointed."""

    latitude: float
    longitude: float
    altitude: float = 0.0
    tilt_deg: float | None = None
    surface_azimuth_deg: float = 180.0
    albedo: float = 0.20
    linke_turbidity: float = 3.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.tilt_deg is None or self.tilt_deg < 0.0:
            self.tilt_deg = clearsky.optimal_fixed_tilt(self.latitude)
        if not self.name:
            self.name = f"{self.latitude:.3f},{self.longitude:.3f}"


class WeatherSeries:
    """Hourly site weather, densifiable to any simulation timestep."""

    def __init__(self, frame: pd.DataFrame, site: Site) -> None:
        self.site = site
        self.frame = frame.sort_index()
        self.provenance = frame.attrs.get("provenance", "unknown")
        self.source = frame.attrs.get("source", "")

        pos = clearsky.solar_position(self.frame.index, site.latitude, site.longitude)
        clear = clearsky.clearsky_irradiance(
            pos, altitude=site.altitude, linke_turbidity=site.linke_turbidity
        )
        ghi = self.frame["ghi"].to_numpy(dtype=float)
        self._k = clearsky.clearsky_index(ghi, clear["ghi"])
        with np.errstate(invalid="ignore", divide="ignore"):
            kd = np.where(ghi > 1.0, self.frame["dhi"].to_numpy(dtype=float) / np.maximum(ghi, 1.0), 1.0)
        self._kd = np.clip(kd, 0.0, 1.0)

        self.t0 = self.frame.index[0]

    # --- densification ----------------------------------------------------
    def densify(self, dt_s: float, duration_s: float | None = None) -> pd.DataFrame:
        """Weather on a uniform `dt_s` grid, with plane-of-array irradiance added."""
        if duration_s is None:
            span = (self.frame.index[-1] - self.frame.index[0]).total_seconds() + 3600.0
            duration_s = span
        n = int(round(duration_s / dt_s))
        offsets = np.arange(n) * dt_s
        index = self.t0 + pd.to_timedelta(offsets, unit="s")

        hourly_offsets = (self.frame.index - self.t0).total_seconds().to_numpy(dtype=float)

        def interp(values: np.ndarray) -> np.ndarray:
            return np.interp(offsets, hourly_offsets, values)

        pos = clearsky.solar_position(index, self.site.latitude, self.site.longitude)
        clear = clearsky.clearsky_irradiance(
            pos, altitude=self.site.altitude, linke_turbidity=self.site.linke_turbidity
        )

        k = np.clip(interp(self._k), 0.0, 1.2)
        kd = np.clip(interp(self._kd), 0.0, 1.0)

        ghi = np.maximum(clear["ghi"] * k, 0.0)
        dhi = ghi * kd
        with np.errstate(invalid="ignore", divide="ignore"):
            dni = np.where(pos.cos_zenith > 1e-3, (ghi - dhi) / np.maximum(pos.cos_zenith, 1e-3), 0.0)
        dni = np.clip(dni, 0.0, 1400.0)
        dhi = np.maximum(ghi - dni * pos.cos_zenith, 0.0)

        poa = clearsky.plane_of_array(
            dni,
            dhi,
            ghi,
            pos,
            tilt_deg=self.site.tilt_deg,
            surface_azimuth_deg=self.site.surface_azimuth_deg,
            albedo=self.site.albedo,
        )

        out = pd.DataFrame(
            {
                "ghi": ghi,
                "dni": dni,
                "dhi": dhi,
                "ghi_clear": clear["ghi"],
                "clearsky_index": k,
                "poa_global": poa["poa_global"],
                "temp_air": interp(self.frame["temp_air"].to_numpy(dtype=float)),
                "relative_humidity": interp(
                    self.frame.get(
                        "relative_humidity", pd.Series(60.0, index=self.frame.index)
                    ).to_numpy(dtype=float)
                ),
                "wind_speed": interp(self.frame["wind_speed"].to_numpy(dtype=float)),
                "cos_zenith": pos.cos_zenith,
            },
            index=index,
        )
        out["time_s"] = offsets
        out.attrs["provenance"] = self.provenance
        out.attrs["source"] = self.source
        return out

    # --- forecasts --------------------------------------------------------
    def forecast(
        self,
        skill: float = 0.75,
        seed: int = 0,
        bias: float = 0.0,
    ) -> "WeatherSeries":
        """A deliberately imperfect view of this weather, for the planner.

        The error is applied to the clear-sky index, grows with lead time, and is
        temporally correlated -- a real forecast is wrong in multi-hour blocks,
        not independently each hour. `skill` is the correlation with truth at
        lead time zero; it decays towards climatology over the horizon.

        `skill=1.0` returns perfect foresight, which is what the oracle baseline
        uses to bound the problem.
        """
        if skill >= 1.0 and bias == 0.0:
            return self

        rng = np.random.default_rng(seed)
        n = len(self.frame)
        lead_hours = np.arange(n, dtype=float)

        # skill decays with a ~72 h e-folding time towards climatology
        weight = skill * np.exp(-lead_hours / 72.0)

        noise = np.empty(n)
        state = 0.0
        for i in range(n):
            state = 0.85 * state + rng.normal(0.0, 0.35)
            noise[i] = state

        climatological_k = float(np.mean(self._k[self._k > 0.0])) if np.any(self._k > 0.0) else 0.5
        k_forecast = weight * self._k + (1.0 - weight) * (climatological_k + 0.12 * noise) + bias
        k_forecast = np.clip(k_forecast, 0.0, 1.15)

        pos = clearsky.solar_position(self.frame.index, self.site.latitude, self.site.longitude)
        clear = clearsky.clearsky_irradiance(
            pos, altitude=self.site.altitude, linke_turbidity=self.site.linke_turbidity
        )
        ghi = np.maximum(clear["ghi"] * k_forecast, 0.0)
        dhi = ghi * self._kd
        with np.errstate(invalid="ignore", divide="ignore"):
            dni = np.where(pos.cos_zenith > 1e-3, (ghi - dhi) / np.maximum(pos.cos_zenith, 1e-3), 0.0)

        frame = self.frame.copy()
        frame["ghi"] = ghi
        frame["dni"] = np.clip(dni, 0.0, 1400.0)
        frame["dhi"] = dhi
        frame.attrs["provenance"] = self.provenance
        frame.attrs["source"] = f"{self.source} [forecast, skill={skill:.2f}]"
        return WeatherSeries(frame, self.site)

    # --- convenience ------------------------------------------------------
    @property
    def duration_s(self) -> float:
        return float((self.frame.index[-1] - self.frame.index[0]).total_seconds() + 3600.0)

    def daily_energy_kwh_m2(self) -> pd.Series:
        return self.frame["ghi"].resample("D").sum() / 1000.0

    def __len__(self) -> int:
        return len(self.frame)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"WeatherSeries(site={self.site.name!r}, hours={len(self)}, "
            f"provenance={self.provenance!r})"
        )
