"""Sunrise and sunset estimation for a calendar day."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from astral import LocationInfo
from astral.sun import sun


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


def parse_video_day(video_path: str, fallback: date | None = None) -> date:
    import os
    import re

    name = os.path.basename(video_path)
    for pattern in (
        re.compile(r"(20\d{2})(\d{2})(\d{2})"),
        re.compile(r"(20\d{2})-(\d{2})-(\d{2})"),
        re.compile(r"(20\d{2})_(\d{2})_(\d{2})"),
    ):
        match = pattern.search(name)
        if match:
            y, m, d = (int(match.group(i)) for i in range(1, 4))
            return date(y, m, d)
    if fallback:
        return fallback
    mtime = os.path.getmtime(video_path)
    return datetime.fromtimestamp(mtime).date()