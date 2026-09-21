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
BOTLIB_EVENT_RE = re.compile(r"\bcoopbot_(?P<event>[a-z_]+)\b")


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


def read_botlib_events(stream: TextIO) -> Counter[str]:
    """Count CoopBot records emitted by the legacy botlib text log."""
    events: Counter[str] = Counter()
    for line in stream:
        for match in BOTLIB_EVENT_RE.finditer(line):
            events[match.group("event")] += 1
        if "coopbot_action" in line:
            fields = message_fields(line)
            action = fields.get("action", "unknown").lower()
            phase = fields.get("phase", "unknown").lower()
            events[f"action_{action}_{phase}"] += 1
        if "coopbot_area_transition" in line:
            fields = message_fields(line)
            actor = fields.get("actor", "unknown").lower()
            events[f"area_{actor}_transition"] += 1
        if "coopbot_objective" in line:
            fields = message_fields(line)
            objective = fields.get("objective", "unknown").lower()
            phase = fields.get("phase", "unknown").lower()
            events[f"objective_{objective}_{phase}"] += 1
        if "coopbot_path_failure" in line:
            # A botlib route failure is the lower-level equivalent of the
            # game-side stuck event and must not be reported as idle follow.
            events["stuck"] += 1
            if re.search(r"\bphase=regroup\b", line):
                events["regroup_path_failure"] += 1
        if "coopbot_regroup" in line and re.search(r"\btraveltype=11\b", line):
            events["elevator_regroup"] += 1
    return events


def read_botlib_map_model(stream: TextIO) -> dict | None:
    """Extract the optional AAS map-model summary from the botlib log."""
    model: dict[str, object] | None = None
    controls: Counter[str] = Counter()
    area_records = 0
    edge_records = 0
    elevator_edges = 0
    control_links = 0
    unresolved_controls = 0

    for line in stream:
        if "coopbot_map_model" in line:
            fields = message_fields(line)
            model = {
                "map": fields.get("map", "<unknown>"),
                "areas": int(number(fields.get("areas")) or 0),
                "clusters": int(number(fields.get("clusters")) or 0),
                "reachabilities": int(number(fields.get("reachabilities")) or 0),
                "elevators": int(number(fields.get("elevators")) or 0),
            }
        elif "coopbot_map_area" in line:
            area_records += 1
        elif "coopbot_map_edge" in line:
            fields = message_fields(line)
            edge_records += 1
            if fields.get("traveltype") == "11":
                elevator_edges += 1
        elif "coopbot_map_control_link" in line:
            control_links += 1
        elif "coopbot_map_control_unresolved" in line:
            unresolved_controls += 1
        elif "coopbot_map_control" in line:
            fields = message_fields(line)
            controls[fields.get("class", "<unknown>")] += 1

    if model is None:
        return None
    model["area_records"] = area_records
    model["edge_records"] = edge_records
    model["elevator_edge_records"] = elevator_edges
    model["controls"] = dict(sorted(controls.items()))
    model["control_links"] = control_links
    model["unresolved_controls"] = unresolved_controls
    return model


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
    regroup_events = 0
    stuck_events = 0

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

        if event == "regroup":
            regroup_events += 1

        if event == "stuck":
            stuck_events += 1

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
            "regroup": regroup_events,
            "stuck": stuck_events,
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
    parser.add_argument(
        "--botlib-log",
        help="optional botlib.log path; counts CoopBot records emitted by botlib",
    )
    parser.add_argument(
        "--require-elevator-edge",
        action="store_true",
        help="fail unless the botlib map model contains a TRAVEL_ELEVATOR edge",
    )
    parser.add_argument(
        "--require-elevator-regroup",
        action="store_true",
        help="fail unless botlib logged a regroup frame using TRAVEL_ELEVATOR",
    )
    parser.add_argument(
        "--require-regroup-path-failure",
        action="store_true",
        help="fail unless botlib logged a failed regroup/path traversal",
    )
    parser.add_argument(
        "--require-regroup-complete",
        action="store_true",
        help="fail unless botlib logged regroup completion",
    )
    parser.add_argument(
        "--require-player-area-transition",
        action="store_true",
        help="fail unless botlib logged a player AAS-area transition",
    )
    parser.add_argument(
        "--require-bot-area-transition",
        action="store_true",
        help="fail unless botlib logged a bot AAS-area transition",
    )
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

    result_object = summarise(records, invalid, source)
    botlib_events: Counter[str] = Counter()
    botlib_map_model: dict | None = None
    validation_errors: list[str] = []

    if args.botlib_log:
        botlib_path = Path(args.botlib_log)
        with botlib_path.open("r", encoding="utf-8", errors="replace") as stream:
            botlib_events = read_botlib_events(stream)
        with botlib_path.open("r", encoding="utf-8", errors="replace") as stream:
            botlib_map_model = read_botlib_map_model(stream)
        result_object["botlib_log"] = {
            "source": str(botlib_path),
            "events": dict(sorted(botlib_events.items())),
        }
        if botlib_map_model is not None:
            result_object["botlib_log"]["map_model"] = botlib_map_model

    if args.require_elevator_edge:
        if botlib_map_model is None:
            validation_errors.append("botlib map model is missing")
        elif botlib_map_model.get("elevator_edge_records", 0) < 1:
            validation_errors.append("no TRAVEL_ELEVATOR edge was recorded")

    if args.require_elevator_regroup and botlib_events.get("elevator_regroup", 0) < 1:
        validation_errors.append(
            "no coopbot_regroup frame used TRAVEL_ELEVATOR"
        )

    if (
        args.require_regroup_path_failure
        and botlib_events.get("regroup_path_failure", 0) < 1
    ):
        validation_errors.append(
            "no failed regroup/path traversal was recorded by botlib"
        )

    if (
        args.require_regroup_complete
        and botlib_events.get("regroup_complete", 0) < 1
    ):
        validation_errors.append("no completed coop regroup was recorded by botlib")

    if (
        args.require_player_area_transition
        and botlib_events.get("area_player_transition", 0) < 1
    ):
        validation_errors.append("no player AAS-area transition was recorded")

    if (
        args.require_bot_area_transition
        and botlib_events.get("area_bot_transition", 0) < 1
    ):
        validation_errors.append("no bot AAS-area transition was recorded")

    if (
        args.require_elevator_edge
        or args.require_elevator_regroup
        or args.require_regroup_path_failure
        or args.require_regroup_complete
        or args.require_player_area_transition
        or args.require_bot_area_transition
    ):
        result_object["validation"] = {
            "ok": not validation_errors,
            "errors": validation_errors,
        }

    result = json.dumps(
        result_object,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        sys.stdout.write(result)
    return 0 if invalid == 0 and not validation_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
