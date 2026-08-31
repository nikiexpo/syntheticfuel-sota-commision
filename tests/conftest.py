"""Shared fixtures.

Everything here is synthetic and seeded, so the suite runs offline and gives the
same answer on every machine. A test that needed the network would fail in CI for
reasons unrelated to the code.
"""

from __future__ import annotations

import pytest

from sfp.weather import pvgis
from sfp.weather.series import Site, WeatherSeries

#: Southern-European reference site, matching the one the report uses.
SITE = Site(37.3891, -5.9845, 10.0, name="Seville, ES")


@pytest.fixture(scope="session")
def site() -> Site:
    return SITE


@pytest.fixture(scope="session")
def synthetic_weather() -> WeatherSeries:
    """Two days of deterministic synthetic weather from the summer solstice."""
    frame = pvgis.synthetic_tmy(SITE.latitude, SITE.longitude, SITE.altitude, seed=7)
    return WeatherSeries(pvgis.slice_days(frame, 172, 2), SITE)
