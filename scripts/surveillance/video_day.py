"""Infer calendar day from a surveillance video filename or mtime."""

from __future__ import annotations

import os
import re
from datetime import date, datetime


def parse_video_day(video_path: str, fallback: date | None = None) -> date:
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