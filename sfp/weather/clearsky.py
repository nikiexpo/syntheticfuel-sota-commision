"""Solar geometry, clear-sky irradiance and plane-of-array transposition.

Standard, citable models -- nothing invented here:

    solar position      Spencer (1971) Fourier series for declination and the
                        equation of time
    air mass            Kasten & Young (1989)
    clear-sky           Ineichen & Perez (2002) simplified model, Linke turbidity
    transposition       Liu & Jordan (1960) isotropic sky

These are the same formulations pvlib implements; carried here so the project
has no heavyweight dependency and the clear-sky baseline works offline. The
curve matters twice over: it sets PV output, and it is the denominator of the
clear-sky index k = GHI / GHI_clear that the weather resampling is built on.

All angles are radians internally; the public functions take and return degrees
only where a caller would naturally think in degrees (tilt, azimuth, latitude).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SOLAR_CONSTANT = 1361.0  # W/m^2, Kopp & Lean (2011)


@dataclass
class SolarPosition:
    """Apparent sun position. Arrays are aligned with the input timestamps."""

    zenith: np.ndarray  # rad, 0 = overhead
    azimuth: np.ndarray  # rad, clockwise from true north
    elevation: np.ndarray  # rad, 0 = horizon
    cos_zenith: np.ndarray  # clipped at 0 -- sun below horizon contributes nothing
    airmass: np.ndarray  # relative, 0 when sun is down
    extraterrestrial: np.ndarray  # W/m^2 on a plane normal to the beam


def _day_angle(day_of_year: np.ndarray) -> np.ndarray:
    """Spencer's day angle B, radians."""
    return 2.0 * np.pi * (day_of_year - 1) / 365.0


def declination(day_of_year: np.ndarray) -> np.ndarray:
    """Solar declination in radians (Spencer 1971, ~0.01 deg accuracy)."""
    b = _day_angle(day_of_year)
    return (
        0.006918
        - 0.399912 * np.cos(b)
        + 0.070257 * np.sin(b)
        - 0.006758 * np.cos(2 * b)
        + 0.000907 * np.sin(2 * b)
        - 0.002697 * np.cos(3 * b)
        + 0.001480 * np.sin(3 * b)
    )


def equation_of_time(day_of_year: np.ndarray) -> np.ndarray:
    """Equation of time in minutes (Spencer 1971)."""
    b = _day_angle(day_of_year)
    return 229.18 * (
        0.000075
        + 0.001868 * np.cos(b)
        - 0.032077 * np.sin(b)
        - 0.014615 * np.cos(2 * b)
        - 0.040849 * np.sin(2 * b)
    )


def extraterrestrial_irradiance(day_of_year: np.ndarray) -> np.ndarray:
    """Beam irradiance at the top of the atmosphere, W/m^2 (eccentricity correction)."""
    b = _day_angle(day_of_year)
    return SOLAR_CONSTANT * (
        1.00011
        + 0.034221 * np.cos(b)
        + 0.001280 * np.sin(b)
        + 0.000719 * np.cos(2 * b)
        + 0.000077 * np.sin(2 * b)
    )


def solar_position(times: pd.DatetimeIndex, latitude: float, longitude: float) -> SolarPosition:
    """Sun position for a UTC timestamp index at one site.

    `latitude` is positive north, `longitude` positive east, both in degrees.
    """
    times = pd.DatetimeIndex(times)
    if times.tz is None:
        times = times.tz_localize("UTC")
    else:
        times = times.tz_convert("UTC")

    doy = times.dayofyear.to_numpy(dtype=float)
    utc_hours = (
        times.hour.to_numpy(dtype=float)
        + times.minute.to_numpy(dtype=float) / 60.0
        + times.second.to_numpy(dtype=float) / 3600.0
    )

    decl = declination(doy)
    eot = equation_of_time(doy)

    # apparent solar time: longitude correction is 4 min per degree east
    solar_time = utc_hours + (4.0 * longitude + eot) / 60.0
    hour_angle = np.deg2rad(15.0 * (solar_time - 12.0))

    phi = np.deg2rad(latitude)
    cos_z = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    cos_z = np.clip(cos_z, -1.0, 1.0)
    zenith = np.arccos(cos_z)

    # azimuth measured clockwise from true north
    az_from_south = np.arctan2(
        np.sin(hour_angle),
        np.cos(hour_angle) * np.sin(phi) - np.tan(decl) * np.cos(phi),
    )
    azimuth = np.mod(az_from_south + np.pi, 2.0 * np.pi)

    cos_z_pos = np.maximum(cos_z, 0.0)

    # Kasten & Young relative air mass; undefined below the horizon
    zenith_deg = np.rad2deg(zenith)
    with np.errstate(invalid="ignore", divide="ignore"):
        am = 1.0 / (
            cos_z_pos + 0.50572 * np.power(np.maximum(96.07995 - zenith_deg, 1e-6), -1.6364)
        )
    am = np.where(cos_z > 0.0, am, 0.0)

    return SolarPosition(
        zenith=zenith,
        azimuth=azimuth,
        elevation=np.pi / 2.0 - zenith,
        cos_zenith=cos_z_pos,
        airmass=am,
        extraterrestrial=extraterrestrial_irradiance(doy),
    )


def clearsky_irradiance(
    pos: SolarPosition,
    altitude: float = 0.0,
    linke_turbidity: float = 3.0,
) -> dict[str, np.ndarray]:
    """Ineichen & Perez (2002) simplified clear-sky GHI / DNI / DHI in W/m^2.

    `linke_turbidity` ~2 is a very clean cold-climate atmosphere, ~3 is a
    typical European annual mean, ~5 is a hazy industrial summer.
    """
    h = float(altitude)
    fh1 = np.exp(-h / 8000.0)
    fh2 = np.exp(-h / 1250.0)
    cg1 = 5.09e-5 * h + 0.868
    cg2 = 3.92e-5 * h + 0.0387

    am = pos.airmass
    i0 = pos.extraterrestrial
    tl = float(linke_turbidity)

    ghi = (
        cg1
        * i0
        * pos.cos_zenith
        * np.exp(-cg2 * am * (fh1 + fh2 * (tl - 1.0)))
        * np.exp(0.01 * np.power(np.maximum(am, 0.0), 1.8))
    )
    ghi = np.where(pos.cos_zenith > 0.0, np.maximum(ghi, 0.0), 0.0)

    b = 0.664 + 0.163 / fh1
    dni = b * i0 * np.exp(-0.09 * am * (tl - 1.0))
    dni = np.where(pos.cos_zenith > 0.0, np.maximum(dni, 0.0), 0.0)

    # keep the closure GHI = DNI*cos(z) + DHI physically consistent
    dni = np.minimum(dni, np.where(pos.cos_zenith > 1e-6, ghi / np.maximum(pos.cos_zenith, 1e-6), 0.0))
    dhi = np.maximum(ghi - dni * pos.cos_zenith, 0.0)

    return {"ghi": ghi, "dni": dni, "dhi": dhi}


def plane_of_array(
    dni: np.ndarray,
    dhi: np.ndarray,
    ghi: np.ndarray,
    pos: SolarPosition,
    tilt_deg: float,
    surface_azimuth_deg: float = 180.0,
    albedo: float = 0.2,
) -> dict[str, np.ndarray]:
    """Transpose horizontal irradiance onto a tilted plane (Liu & Jordan isotropic sky).

    `surface_azimuth_deg` is clockwise from true north, so 180 is due south.
    The isotropic model is mildly conservative against measured POA (it ignores
    circumsolar brightening); that bias is recorded in the assumptions ledger.
    """
    beta = np.deg2rad(tilt_deg)
    gamma_p = np.deg2rad(surface_azimuth_deg)

    cos_aoi = np.cos(pos.zenith) * np.cos(beta) + np.sin(pos.zenith) * np.sin(beta) * np.cos(
        pos.azimuth - gamma_p
    )
    cos_aoi = np.maximum(cos_aoi, 0.0)

    beam = dni * cos_aoi
    sky_diffuse = dhi * (1.0 + np.cos(beta)) / 2.0
    ground_diffuse = ghi * albedo * (1.0 - np.cos(beta)) / 2.0

    return {
        "poa_global": beam + sky_diffuse + ground_diffuse,
        "poa_beam": beam,
        "poa_sky_diffuse": sky_diffuse,
        "poa_ground_diffuse": ground_diffuse,
        "cos_aoi": cos_aoi,
    }


def clearsky_index(ghi: np.ndarray, ghi_clear: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """k = GHI / GHI_clear, defined as 0 when the sun is effectively down.

    `floor` (W/m^2) avoids the 0/0 that would otherwise make dawn and dusk noisy;
    the stochastic solar model in `sfp.weather.ensembles` is fitted on this
    quantity, so a stable definition matters.
    """
    return np.where(ghi_clear > floor, ghi / np.maximum(ghi_clear, floor), 0.0)


def optimal_fixed_tilt(latitude: float) -> float:
    """A reasonable default fixed tilt for a mid-latitude site, degrees.

    Rule of thumb that lands within ~1% of annual-yield-optimal across Europe:
    tilt ~ 0.76*|lat| + 3.1 deg. Used only as a default when the user does not
    specify an array geometry.
    """
    return float(np.clip(0.76 * abs(latitude) + 3.1, 0.0, 60.0))
