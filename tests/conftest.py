"""Shared fixtures.

Everything here is synthetic and seeded, so the suite runs offline and gives the
same answer on every machine. A test that needed the network would fail in CI for
reasons unrelated to the code.
"""

from __future__ import annotations

import pytest

from sfp.cli import build_reference_plant
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


@pytest.fixture
def reference_plant():
    """The reference plant, at the one sizing every simulation uses.

    Function-scoped on purpose: a `Plant` is cheap to build and carries mutable
    parameter objects, so sharing one across a module invites a test that
    overrides a parameter to change the answer of an unrelated test that runs
    after it.
    """
    return build_reference_plant()
