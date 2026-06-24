"""Rule-based event extraction from YOLO tracks inside motion segments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from surveillance.roi import NormalizedBox, zone_hits

VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle"}
BICYCLE_LABELS = {"bicycle"}
PERSON_LABELS = {"person"}
HERD_LABELS = {"sheep", "cow", "horse"}


@dataclass
class TrackObservation:
    track_id: int
    label: str
    timestamp_sec: float
    zones: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class SurveillanceEvent:
    event_type: str
    start_sec: float
    end_sec: float
    start_time: str
    end_time: str
    description_ru: str
    details: dict[str, Any]


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _fmt_clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours:02d}:{minutes:02d}"


def collect_track_observations(
    frames: list[dict[str, Any]],
    zones: dict[str, NormalizedBox],
) -> dict[int, list[TrackObservation]]:
    by_track: dict[int, list[TrackObservation]] = defaultdict(list)
    for frame in frames:
        ts = float(frame["timestamp_sec"])
        for det in frame.get("detections", []):
            track_id = det.get("track_id")
            if track_id is None:
                continue
            tid = int(track_id)
            label = str(det.get("label", ""))
            hits = zone_hits(det, zones)
            by_track[tid].append(TrackObservation(track_id=tid, label=label, timestamp_sec=ts, zones=hits))
    return by_track


def _dominant_label(observations: list[TrackObservation]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for obs in observations:
        counts[obs.label] += 1
    return max(counts, key=counts.get)


def _zone_span(observations: list[TrackObservation], zone_name: str) -> tuple[float, float] | None:
    hits = [o.timestamp_sec for o in observations if zone_name in o.zones]
    if not hits:
        return None
    return min(hits), max(hits)


def infer_events(
    frames: list[dict[str, Any]],
    zones: dict[str, NormalizedBox],
    day_start_offset_sec: float = 0.0,
) -> list[SurveillanceEvent]:
    tracks = collect_track_observations(frames, zones)
    events: list[SurveillanceEvent] = []

    road_zone = "road" in zones
    gate_zone = "gate" in zones
    property_zone = "property" in zones

    herd_frames: list[tuple[float, dict[str, int]]] = []
    for frame in frames:
        ts = float(frame["timestamp_sec"])
        counts: dict[str, int] = defaultdict(int)
        for det in frame.get("detections", []):
            label = str(det.get("label", ""))
            if label in HERD_LABELS and (not road_zone or "road" in zone_hits(det, zones)):
                counts[label] += 1
        herd_total = sum(counts.values())
        if herd_total >= 3:
            herd_frames.append((ts, dict(counts)))

    if herd_frames:
        start = herd_frames[0][0]
        end = herd_frames[-1][0]
        mix = herd_frames[len(herd_frames) // 2][1]
        labels = ", ".join(f"{k}×{v}" for k, v in sorted(mix.items()))
        events.append(
            SurveillanceEvent(
                event_type="herd",
                start_sec=start,
                end_sec=end,
                start_time=_fmt_time(start),
                end_time=_fmt_time(end),
                description_ru=f"Прошло стадо ({labels})",
                details={"counts": mix},
            )
        )

    vehicles_near_gate: list[tuple[int, float, float]] = []
    for tid, observations in tracks.items():
        label = _dominant_label(observations)
        start_ts = observations[0].timestamp_sec
        end_ts = observations[-1].timestamp_sec

        if label in VEHICLE_LABELS and road_zone:
            if any("road" in o.zones for o in observations):
                events.append(
                    SurveillanceEvent(
                        event_type="vehicle_road",
                        start_sec=start_ts,
                        end_sec=end_ts,
                        start_time=_fmt_time(start_ts),
                        end_time=_fmt_time(end_ts),
                        description_ru="По дороге проехал транспорт",
                        details={"label": label, "track_id": tid},
                    )
                )

        if label in BICYCLE_LABELS and road_zone:
            if any("road" in o.zones for o in observations):
                events.append(
                    SurveillanceEvent(
                        event_type="bicycle_road",
                        start_sec=start_ts,
                        end_sec=end_ts,
                        start_time=_fmt_time(start_ts),
                        end_time=_fmt_time(end_ts),
                        description_ru="По дороге проехал велосипед",
                        details={"track_id": tid},
                    )
                )

        if label in PERSON_LABELS:
            if any("road" in o.zones for o in observations):
                events.append(
                    SurveillanceEvent(
                        event_type="person_walk",
                        start_sec=start_ts,
                        end_sec=end_ts,
                        start_time=_fmt_time(start_ts),
                        end_time=_fmt_time(end_ts),
                        description_ru="Прошёл человек",
                        details={"track_id": tid},
                    )
                )
            if property_zone:
                entered = [o for o in observations if "property" in o.zones]
                if entered:
                    events.append(
                        SurveillanceEvent(
                            event_type="person_on_property",
                            start_sec=entered[0].timestamp_sec,
                            end_sec=entered[-1].timestamp_sec,
                            start_time=_fmt_time(entered[0].timestamp_sec),
                            end_time=_fmt_time(entered[-1].timestamp_sec),
                            description_ru=f"Человек #{tid} на территории",
                            details={"track_id": tid},
                        )
                    )

        if label in VEHICLE_LABELS and gate_zone:
            gate_span = _zone_span(observations, "gate")
            if gate_span:
                vehicles_near_gate.append((tid, gate_span[0], gate_span[1]))
                events.append(
                    SurveillanceEvent(
                        event_type="vehicle_gate",
                        start_sec=gate_span[0],
                        end_sec=gate_span[1],
                        start_time=_fmt_time(gate_span[0]),
                        end_time=_fmt_time(gate_span[1]),
                        description_ru="Машина подъехала к воротам",
                        details={"track_id": tid, "label": label},
                    )
                )

    for vehicle_tid, gate_start, gate_end in vehicles_near_gate:
        people = []
        for tid, observations in tracks.items():
            if _dominant_label(observations) not in PERSON_LABELS:
                continue
            person_start = observations[0].timestamp_sec
            person_end = observations[-1].timestamp_sec
            if gate_start - 30 <= person_start <= gate_end + 30:
                people.append(tid)
        if len(people) >= 2:
            events.append(
                SurveillanceEvent(
                    event_type="boarding_vehicle",
                    start_sec=gate_start,
                    end_sec=gate_end,
                    start_time=_fmt_time(gate_start),
                    end_time=_fmt_time(gate_end),
                    description_ru=(
                        f"У ворот люди (треки {people}) сели в машину (трек {vehicle_tid}) и уехали"
                    ),
                    details={
                        "vehicle_track_id": vehicle_tid,
                        "person_track_ids": people,
                    },
                )
            )

    person_property_events = [e for e in events if e.event_type == "person_on_property"]
    by_track: dict[int, list[SurveillanceEvent]] = defaultdict(list)
    for event in person_property_events:
        by_track[int(event.details["track_id"])].append(event)
    for tid, track_events in by_track.items():
        ordered = sorted(track_events, key=lambda e: e.start_sec)
        if len(ordered) >= 2:
            leave = ordered[0]
            return_ev = ordered[-1]
            events.append(
                SurveillanceEvent(
                    event_type="person_departure_return",
                    start_sec=leave.start_sec,
                    end_sec=return_ev.end_sec,
                    start_time=_fmt_clock(leave.start_sec + day_start_offset_sec),
                    end_time=_fmt_clock(return_ev.end_sec + day_start_offset_sec),
                    description_ru=(
                        f"Человек #{tid} ушёл в {leave.start_time[:5]} и вернулся в {return_ev.end_time[:5]}"
                    ),
                    details={
                        "track_id": tid,
                        "left_at": leave.start_time,
                        "returned_at": return_ev.end_time,
                    },
                )
            )

    dedup: dict[tuple[str, int, str], SurveillanceEvent] = {}
    for event in events:
        key = (event.event_type, int(event.start_sec), event.description_ru)
        dedup[key] = event
    return sorted(dedup.values(), key=lambda e: (e.start_sec, e.event_type))