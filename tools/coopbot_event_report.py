#!/usr/bin/env python3
"""Summarise CoopBot JSONL telemetry for reproducible episode comparisons."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, TextIO


FIELD_RE = re.compile(
    r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)='
    r'(?:(?:"(?P<quoted>[^"]*)")|(?P<bare>\([^)]*\)|[^\s]+))'
)


def read_records(stream: TextIO) -> tuple[list[dict], int]:
    records: list[dict] = []
    invalid = 0
    for line_number, line in enumerate(stream, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if not isinstance(record, dict):
            invalid += 1
            continue
        record["_line"] = line_number
        records.append(record)
    return records, invalid


def message_fields(message: object) -> dict[str, str]:
    """Extract stable key=value fields from the legacy event message."""
    if not isinstance(message, str):
        return {}

    fields: dict[str, str] = {}
    for match in FIELD_RE.finditer(message):
        fields[match.group("key")] = (
            match.group("quoted")
            if match.group("quoted") is not None
            else match.group("bare")
        )
    return fields


def number(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def telemetry_summary(records: Iterable[dict]) -> dict:
    states: Counter[str] = Counter()
    shot_kinds: Counter[str] = Counter()
    snapshot_count = 0
    snapshot_distances: list[float] = []
    visible_enemies: list[float] = []
    target_visible = 0
    damage_requested = 0.0
    damage_applied = 0.0
    damage_attempts = 0
    damage_events = 0
    player_contacts = 0
    player_block_candidates = 0
    encounter_events = 0
    target_acquired = 0
    target_lost = 0

    for record in records:
        fields = message_fields(record.get("message"))
        event = record.get("event")

        if event == "bot_snapshot":
            snapshot_count += 1
            state = fields.get("state")
            if state:
                states[state] += 1
            distance = number(fields.get("distance_to_player"))
            if distance is not None and distance >= 0.0:
                snapshot_distances.append(distance)
            visible = number(fields.get("visible_enemies"))
            if visible is not None:
                visible_enemies.append(visible)
            if fields.get("target_visible") == "1":
                target_visible += 1

        if event == "shot":
            shot_kinds[fields.get("kind", "<unknown>")] += 1

        if event == "damage_attempt":
            damage_attempts += 1

        if event == "damage_applied":
            damage_events += 1
            requested = number(fields.get("requested"))
            applied = number(fields.get("applied"))
            if requested is not None:
                damage_requested += requested
            if applied is not None:
                damage_applied += applied

        if event == "player_contact":
            player_contacts += 1

        if event == "player_block_candidate":
            player_block_candidates += 1

        if event == "encounter_observed":
            encounter_events += 1

        if event == "target_acquired":
            target_acquired += 1

        if event == "target_lost":
            target_lost += 1

    return {
        "bot_snapshots": {
            "records": snapshot_count,
            "states": dict(sorted(states.items())),
            "distance_to_player": {
                "samples": len(snapshot_distances),
                "average": (
                    sum(snapshot_distances) / len(snapshot_distances)
                    if snapshot_distances
                    else None
                ),
                "maximum": max(snapshot_distances) if snapshot_distances else None,
            },
            "visible_enemies": {
                "samples": len(visible_enemies),
                "maximum": max(visible_enemies) if visible_enemies else None,
                "target_visible_samples": target_visible,
            },
        },
        "shots": {
            "records": sum(shot_kinds.values()),
            "kinds": dict(sorted(shot_kinds.items())),
        },
        "damage": {
            "attempts": damage_attempts,
            "applied_events": damage_events,
            "requested_total": damage_requested,
            "applied_total": damage_applied,
        },
        "coop": {
            "player_contacts": player_contacts,
            "player_block_candidates": player_block_candidates,
            "encounter_observed": encounter_events,
            "target_acquired": target_acquired,
            "target_lost": target_lost,
        },
    }


def summarise(records: Iterable[dict], invalid_lines: int, source: str) -> dict:
    event_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    episodes: defaultdict[str, dict] = defaultdict(
        lambda: {"records": 0, "events": {}, "maps": [], "first_time": None, "last_time": None}
    )
    maps: defaultdict[str, Counter[str]] = defaultdict(Counter)
    record_count = 0

    for record in records:
        record_count += 1
        event = str(record.get("event", "unknown"))
        severity = str(record.get("severity", "unknown"))
        episode_id = str(record.get("episode_id", "unknown"))
        map_name = str(record.get("map", "unknown"))
        event_counts[event] += 1
        severity_counts[severity] += 1
        maps[map_name][event] += 1

        episode = episodes[episode_id]
        episode["records"] += 1
        episode["events"][event] = episode["events"].get(event, 0) + 1
        if map_name not in episode["maps"]:
            episode["maps"].append(map_name)
        timestamp = record.get("time")
        if isinstance(timestamp, (int, float)):
            if episode["first_time"] is None or timestamp < episode["first_time"]:
                episode["first_time"] = timestamp
            if episode["last_time"] is None or timestamp > episode["last_time"]:
                episode["last_time"] = timestamp

    return {
        "schema_version": 1,
        "source": source,
        "records": record_count,
        "invalid_lines": invalid_lines,
        "events": dict(sorted(event_counts.items())),
        "severities": dict(sorted(severity_counts.items())),
        "telemetry": telemetry_summary(records),
        "maps": {
            name: {"records": sum(counter.values()), "events": dict(sorted(counter.items()))}
            for name, counter in sorted(maps.items())
        },
        "episodes": {
            episode_id: episodes[episode_id]
            for episode_id in sorted(episodes)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSONL path, or '-' for stdin")
    parser.add_argument("-o", "--output", help="write summary JSON to this path")
    args = parser.parse_args()

    if args.input == "-":
        records, invalid = read_records(sys.stdin)
        source = "<stdin>"
    else:
        path = Path(args.input)
        with path.open("r", encoding="utf-8") as stream:
            records, invalid = read_records(stream)
        source = str(path)

    result = json.dumps(
        summarise(records, invalid, source),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        sys.stdout.write(result)
    return 0 if invalid == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
