"""Sunrise and sunset estimation for a calendar day."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

@dataclass(frozen=True)
class SunTimes:
    date: str
    timezone: str
    sunrise_local: str
    sunset_local: str
    sunrise_utc: str
    sunset_utc: str


def compute_sun_times(
    day: date,
    latitude: float,
    longitude: float,
    timezone_name: str,
) -> SunTimes:
    from astral import LocationInfo
    from astral.sun import sun

    tz = ZoneInfo(timezone_name)
    location = LocationInfo(
        name="camera",
        region="",
        timezone=timezone_name,
        latitude=latitude,
        longitude=longitude,
    )
    times = sun(location.observer, date=day, tzinfo=tz)
    sunrise = times["sunrise"]
    sunset = times["sunset"]

    def fmt_local(dt: datetime) -> str:
        return dt.astimezone(tz).strftime("%H:%M")

    def fmt_utc(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).strftime("%H:%M")

    return SunTimes(
        date=day.isoformat(),
        timezone=timezone_name,
        sunrise_local=fmt_local(sunrise),
        sunset_local=fmt_local(sunset),
        sunrise_utc=fmt_utc(sunrise),
        sunset_utc=fmt_utc(sunset),
    )


