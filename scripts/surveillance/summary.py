"""Human-readable daily summary from structured events."""

from __future__ import annotations

from typing import Any

from surveillance.day_night import DayNightTimes
from surveillance.events import SurveillanceEvent


def build_summary_text(
    day_night: DayNightTimes,
    events: list[SurveillanceEvent],
    motion_stats: dict[str, float],
) -> str:
    dawn = day_night.dawn or "не определён"
    dusk = day_night.dusk or "не определён"
    lines = [
        f"Сводка за {day_night.date}",
        "",
        f"Переход на день (цвет): {dawn}",
        f"Переход на ночь (ч/б): {dusk}",
        f"Метод: {day_night.method}",
        "",
        "События:",
    ]
    if not events:
        lines.append("- Значимых событий не обнаружено.")
    else:
        for event in events:
            lines.append(f"- {event.start_time[:5]}–{event.end_time[:5]}: {event.description_ru}")
    lines.extend(
        [
            "",
            "Техническая статистика:",
            f"- Сегментов движения (после фильтра ветра): {int(motion_stats.get('segments_merged', 0))}",
            f"- Отброшено как ветка/бельё/шум: {int(motion_stats.get('segments_rejected_wind', 0))}",
            f"- Доля ночных сэмплов: {day_night.night_sample_ratio:.2f}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_summary_json(
    video_path: str,
    day_night: DayNightTimes,
    events: list[SurveillanceEvent],
    motion_stats: dict[str, float],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "video_path": video_path,
        "day_night": {
            "date": day_night.date,
            "method": day_night.method,
            "dawn": day_night.dawn,
            "dusk": day_night.dusk,
            "night_sample_ratio": day_night.night_sample_ratio,
            "color_sample_ratio": day_night.color_sample_ratio,
        },
        "motion": motion_stats,
        "motion_segments": segments,
        "events": [
            {
                "type": e.event_type,
                "start_sec": e.start_sec,
                "end_sec": e.end_sec,
                "start_time": e.start_time,
                "end_time": e.end_time,
                "description_ru": e.description_ru,
                "details": e.details,
            }
            for e in events
        ],
        "summary_text": build_summary_text(day_night, events, motion_stats),
    }