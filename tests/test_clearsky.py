"""Solar geometry and clear-sky irradiance, checked against closed-form truth.

These are the cheapest high-value tests in the project: every downstream number
is proportional to the irradiance, so an error here is an error everywhere, and
solar geometry has exact analytic answers to check against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfp.weather import clearsky


def _index(day: str, tz: str = "UTC") -> pd.DatetimeIndex:
    return pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="h", tz=tz)


def test_solstice_noon_elevation_matches_analytic():
    """At solar noon on the June solstice, elevation = 90 - lat + declination."""
    latitude = 48.85
    pos = clearsky.solar_position(_index("2001-06-21"), latitude, 0.0)
    expected = 90.0 - latitude + 23.44
    assert np.rad2deg(pos.elevation).max() == pytest.approx(expected, abs=0.15)


def test_declination_extremes():
    """Declination reaches +-23.44 deg at the solstices and ~0 at the equinoxes."""
    doy = np.arange(1, 366)
    decl = np.rad2deg(clearsky.declination(doy))
    assert decl.max() == pytest.approx(23.44, abs=0.1)
    assert decl.min() == pytest.approx(-23.44, abs=0.1)
    assert abs(decl[79]) < 0.5  # ~20 March


def test_equation_of_time_range():
    """EoT swings roughly -14 to +16 minutes over the year."""
    eot = clearsky.equation_of_time(np.arange(1, 366))
    assert eot.min() == pytest.approx(-14.0, abs=1.0)
    assert eot.max() == pytest.approx(16.4, abs=1.0)


def test_solar_noon_lands_at_expected_utc_hour():
    """Solar noon at 2.35 deg E is ~11:51 UTC, so the 12:00 sample is the peak."""
    pos = clearsky.solar_position(_index("2001-06-21"), 48.85, 2.35)
    assert int(np.argmax(pos.cos_zenith)) == 12


def test_irradiance_closure():
    """GHI must equal DNI*cos(zenith) + DHI exactly, by construction."""
    pos = clearsky.solar_position(_index("2001-04-15"), 40.0, -3.7)
    cs = clearsky.clearsky_irradiance(pos, altitude=650.0)
    residual = cs["ghi"] - (cs["dni"] * pos.cos_zenith + cs["dhi"])
    assert np.abs(residual).max() < 1e-9


def test_no_irradiance_at_night():
    pos = clearsky.solar_position(_index("2001-12-21"), 60.0, 0.0)
    cs = clearsky.clearsky_irradiance(pos)
    night = pos.cos_zenith <= 0.0
    assert night.any()
    assert np.all(cs["ghi"][night] == 0.0)
    assert np.all(cs["dni"][night] == 0.0)


def test_clearsky_peak_is_physical():
    """Peak clear-sky GHI at a southern European site sits in 850-1050 W/m^2."""
    pos = clearsky.solar_position(_index("2001-06-21"), 37.4, -6.0)
    cs = clearsky.clearsky_irradiance(pos, altitude=10.0, linke_turbidity=3.2)
    assert 850.0 < cs["ghi"].max() < 1050.0


def test_turbidity_reduces_irradiance():
    """A hazier atmosphere must not produce more ground irradiance."""
    pos = clearsky.solar_position(_index("2001-06-21"), 40.0, 0.0)
    clean = clearsky.clearsky_irradiance(pos, linke_turbidity=2.0)["ghi"].sum()
    hazy = clearsky.clearsky_irradiance(pos, linke_turbidity=6.0)["ghi"].sum()
    assert hazy < clean


def test_latitude_ordering_in_summer():
    """In June, daily clear-sky energy should not increase towards the pole."""
    totals = []
    for lat in (35.0, 45.0, 55.0, 65.0):
        pos = clearsky.solar_position(_index("2001-06-21"), lat, 0.0)
        totals.append(clearsky.clearsky_irradiance(pos)["ghi"].sum())
    # high latitudes gain daylight hours but lose elevation; the net is a mild
    # decline, so we only assert the endpoints are ordered
    assert totals[0] > totals[-1]


def test_tilted_plane_beats_horizontal_in_winter():
    """A latitude-tilted array collects more than a horizontal one in winter."""
    latitude = 45.0
    pos = clearsky.solar_position(_index("2001-12-21"), latitude, 0.0)
    cs = clearsky.clearsky_irradiance(pos)
    tilted = clearsky.plane_of_array(
        cs["dni"], cs["dhi"], cs["ghi"], pos, tilt_deg=clearsky.optimal_fixed_tilt(latitude)
    )
    assert tilted["poa_global"].sum() > cs["ghi"].sum()


def test_horizontal_transposition_is_identity():
    """A zero-tilt plane must see exactly GHI."""
    pos = clearsky.solar_position(_index("2001-05-01"), 45.0, 0.0)
    cs = clearsky.clearsky_irradiance(pos)
    poa = clearsky.plane_of_array(cs["dni"], cs["dhi"], cs["ghi"], pos, tilt_deg=0.0)
    assert np.abs(poa["poa_global"] - cs["ghi"]).max() < 1e-9


def test_clearsky_index_is_zero_at_night():
    ghi = np.array([0.0, 0.0, 500.0])
    ghi_clear = np.array([0.0, 0.5, 800.0])
    k = clearsky.clearsky_index(ghi, ghi_clear)
    assert k[0] == 0.0
    assert k[1] == 0.0  # below the floor, treated as dark
    assert k[2] == pytest.approx(0.625)


def test_optimal_tilt_is_sane():
    assert clearsky.optimal_fixed_tilt(0.0) == pytest.approx(3.1, abs=0.1)
    assert 25.0 < clearsky.optimal_fixed_tilt(37.4) < 40.0
    assert clearsky.optimal_fixed_tilt(-52.0) == clearsky.optimal_fixed_tilt(52.0)
