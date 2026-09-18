"""Weather input: PVGIS typical meteorological year, with an offline fallback.

PVGIS (EU Joint Research Centre) covers all of Europe, the deployment region the
brief specifies. The TMY endpoint gives, on one hourly index, every driver this
plant model needs:

    G(h), Gb(n), Gd(h)   GHI / DNI / DHI  -> PV output
    T2m                  air temperature  -> PV derating, kiln losses, carbonation rate
    RH                   relative humidity-> carbonation rate (moisture promotes it)
    WS10m                wind speed       -> module and kiln convective losses

Results are cached under `data/cache/` keyed by rounded coordinates, so a sweep
hits the network once per site and every later run is offline and reproducible.

If the network is unavailable, `load_weather` falls back to a synthetic year
built from the clear-sky model times a stochastic clear-sky index. That keeps the
whole project runnable on a plane, but it is *invented data* -- the returned
frame carries `attrs["provenance"]` so every downstream report can say which it
used, and the CLI prints a warning.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sfp.weather import clearsky

log = logging.getLogger(__name__)

PVGIS_TMY_URL = "https://re.jrc.ec.europa.eu/api/v5_2/tmy"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# PVGIS TMY column -> our canonical name
_PVGIS_COLUMNS = {
    "G(h)": "ghi",
    "Gb(n)": "dni",
    "Gd(h)": "dhi",
    "T2m": "temp_air",
    "RH": "relative_humidity",
    "WS10m": "wind_speed",
    "SP": "pressure",
}

CANONICAL_COLUMNS = (
    "ghi",
    "dni",
    "dhi",
    "temp_air",
    "relative_humidity",
    "wind_speed",
    "pressure",
)


class WeatherError(RuntimeError):
    """Raised when weather data can be neither fetched nor synthesised."""


def _cache_path(latitude: float, longitude: float) -> Path:
    return CACHE_DIR / f"tmy_{latitude:+08.4f}_{longitude:+09.4f}.csv"


def _parse_pvgis_tmy(payload: dict) -> pd.DataFrame:
    """PVGIS TMY JSON -> canonical hourly frame indexed in UTC."""
    rows = payload["outputs"]["tmy_hourly"]
    frame = pd.DataFrame(rows)

    # PVGIS stamps TMY hours as "20050101:0010" -- real month/day, donor year,
    # and the :10 offset marks the middle of the hour. Normalise to hour starts
    # on a single non-leap reference year so the index is monotonic.
    stamps = frame["time(UTC)"].astype(str)
    month = stamps.str[4:6].astype(int)
    day = stamps.str[6:8].astype(int)
    hour = stamps.str[9:11].astype(int)
    index = pd.to_datetime(
        {"year": 2001, "month": month, "day": day, "hour": hour}
    )

    out = pd.DataFrame(index=pd.DatetimeIndex(index, name="time"))
    for source, target in _PVGIS_COLUMNS.items():
        if source in frame:
            out[target] = pd.to_numeric(frame[source].to_numpy(), errors="coerce")

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out.index = out.index.tz_localize("UTC")
    return out


def fetch_pvgis_tmy(latitude: float, longitude: float, timeout: float = 30.0) -> pd.DataFrame:
    """Download a TMY from PVGIS. Raises `WeatherError` on any failure."""
    import requests  # imported lazily so the offline path needs no network stack

    params = {
        "lat": f"{latitude:.4f}",
        "lon": f"{longitude:.4f}",
        "outputformat": "json",
        "browser": 0,
    }
    try:
        response = requests.get(PVGIS_TMY_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # network, HTTP, or JSON -- all mean "no data"
        raise WeatherError(f"PVGIS request failed for ({latitude}, {longitude}): {exc}") from exc

    try:
        return _parse_pvgis_tmy(payload)
    except (KeyError, ValueError) as exc:
        raise WeatherError(f"unexpected PVGIS payload shape: {exc}") from exc


def synthetic_tmy(
    latitude: float,
    longitude: float,
    altitude: float = 0.0,
    linke_turbidity: float = 3.0,
    seed: int = 0,
) -> pd.DataFrame:
    """A physically plausible but *invented* year, for offline use.

    Clear-sky irradiance modulated by a stochastic clear-sky index with two
    timescales, because both matter to the planner:

        day-to-day   a beta-distributed daily mean (cloudy days cluster into
                     multi-day weather systems, which is exactly what makes a
                     10-day forecast horizon worth having)
        hour-to-hour an AR(1) ripple (passing cumulus)

    Air temperature is a seasonal sinusoid plus a diurnal swing that scales with
    the day's clearness; humidity is anti-correlated with clearness.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2001-01-01", "2001-12-31 23:00", freq="h", tz="UTC")

    pos = clearsky.solar_position(index, latitude, longitude)
    clear = clearsky.clearsky_irradiance(pos, altitude=altitude, linke_turbidity=linke_turbidity)

    doy = index.dayofyear.to_numpy()
    n_days = int(doy.max())

    # seasonal clearness: sunnier in summer, and more so at lower latitudes
    season = np.cos(2 * np.pi * (np.arange(1, n_days + 1) - 172) / 365.25)
    mean_k = np.clip(0.55 + 0.12 * season - 0.0015 * (abs(latitude) - 45.0), 0.30, 0.78)

    # AR(1) across days gives multi-day weather systems
    daily_k = np.empty(n_days)
    innovation = 0.0
    for d in range(n_days):
        innovation = 0.62 * innovation + rng.normal(0.0, 0.16)
        daily_k[d] = np.clip(mean_k[d] + innovation, 0.05, 0.92)

    # AR(1) within the day for passing cloud
    hourly_noise = np.empty(len(index))
    ripple = 0.0
    for i in range(len(index)):
        ripple = 0.80 * ripple + rng.normal(0.0, 0.09)
        hourly_noise[i] = ripple

    k = np.clip(daily_k[doy - 1] + hourly_noise, 0.03, 1.05)

    ghi = clear["ghi"] * k
    # cloud suppresses beam far more than diffuse
    beam_fraction = np.clip((k - 0.20) / 0.75, 0.0, 1.0) ** 1.6
    dni = clear["dni"] * beam_fraction
    dni = np.minimum(dni, np.where(pos.cos_zenith > 1e-6, ghi / np.maximum(pos.cos_zenith, 1e-6), 0.0))
    dhi = np.maximum(ghi - dni * pos.cos_zenith, 0.0)

    hour = index.hour.to_numpy()
    annual_mean = 14.0 - 0.42 * (abs(latitude) - 45.0)
    seasonal_amp = 9.0 + 0.05 * abs(latitude)
    diurnal_amp = 3.0 + 5.0 * daily_k[doy - 1]
    temp_air = (
        annual_mean
        + seasonal_amp * np.cos(2 * np.pi * (doy - 200) / 365.25)
        - diurnal_amp * np.cos(2 * np.pi * (hour - 15) / 24.0)
        + rng.normal(0.0, 0.7, len(index))
    )

    relative_humidity = np.clip(
        92.0 - 45.0 * daily_k[doy - 1] - 0.9 * (temp_air - annual_mean) + rng.normal(0.0, 4.0, len(index)),
        15.0,
        100.0,
    )
    wind_speed = np.clip(rng.gamma(2.0, 1.6, len(index)), 0.0, 25.0)
    pressure = np.full(len(index), 101325.0 * np.exp(-altitude / 8400.0))

    return pd.DataFrame(
        {
            "ghi": ghi,
            "dni": dni,
            "dhi": dhi,
            "temp_air": temp_air,
            "relative_humidity": relative_humidity,
            "wind_speed": wind_speed,
            "pressure": pressure,
        },
        index=index,
    )


def load_weather(
    latitude: float,
    longitude: float,
    altitude: float = 0.0,
    use_cache: bool = True,
    allow_network: bool = True,
    allow_synthetic: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Canonical hourly weather for a site: cache -> PVGIS -> synthetic.

    The returned frame carries `attrs["provenance"]` (`"measured"` for real PVGIS
    data, `"invented"` for the synthetic fallback) and `attrs["source"]`. Report
    generation reads those, so a run can never silently present made-up weather
    as if it were measured.
    """
    cache = _cache_path(latitude, longitude)

    if use_cache and cache.exists():
        frame = pd.read_csv(cache, index_col=0, parse_dates=True)
        frame.index = pd.DatetimeIndex(frame.index)
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        frame.attrs["provenance"] = "measured"
        frame.attrs["source"] = f"PVGIS TMY (cached at {cache.name})"
        log.info("weather: using cached PVGIS TMY for (%.4f, %.4f)", latitude, longitude)
        return frame

    if allow_network:
        try:
            frame = fetch_pvgis_tmy(latitude, longitude)
            if use_cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(cache)
            frame.attrs["provenance"] = "measured"
            frame.attrs["source"] = "PVGIS v5.2 TMY (SARAH2 / ERA5), JRC European Commission"
            log.info("weather: downloaded PVGIS TMY for (%.4f, %.4f)", latitude, longitude)
            return frame
        except WeatherError as exc:
            if not allow_synthetic:
                raise
            warnings.warn(
                f"PVGIS unavailable ({exc}); falling back to SYNTHETIC weather. "
                "Results are illustrative only and must not be reported as measured.",
                stacklevel=2,
            )

    if not allow_synthetic:
        raise WeatherError("no cached weather, network disabled, synthetic disabled")

    frame = synthetic_tmy(latitude, longitude, altitude=altitude, seed=seed)
    frame.attrs["provenance"] = "invented"
    frame.attrs["source"] = "synthetic clear-sky x stochastic clearness index (offline fallback)"
    return frame


def slice_days(frame: pd.DataFrame, start_day: int, n_days: int) -> pd.DataFrame:
    """`n_days` of hourly weather beginning at day-of-year `start_day`, wrapping the year.

    A TMY is cyclic, so a window that runs past 31 December wraps to 1 January
    and the index is made continuous. Used to pick the simulation window.
    """
    if not 1 <= start_day <= 365:
        raise ValueError(f"start_day must be in 1..365, got {start_day}")
    hours = int(n_days * 24)
    start = (start_day - 1) * 24
    n = len(frame)
    positions = (np.arange(start, start + hours)) % n
    out = frame.iloc[positions].copy()
    out.index = pd.date_range(frame.index[start], periods=hours, freq="h")
    out.attrs = dict(frame.attrs)
    return out
