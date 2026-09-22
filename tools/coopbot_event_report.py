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


def integer(value: object, default: int = 0) -> int:
    """Parse decimal or C-style hexadecimal fields from botlib diagnostics."""
    try:
        return int(str(value), 0)
    except (TypeError, ValueError):
        parsed = number(value)
        return int(parsed) if parsed is not None else default


def vector3(value: object) -> tuple[float, float, float] | None:
    """Parse the compact ``(x y z)`` vector form used by botlib logs."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
        value,
    )
    if match is None:
        return None
    try:
        components = tuple(float(component) for component in match.groups())
    except ValueError:
        return None
    return components  # type: ignore[return-value]


def format_vector3(value: tuple[float, float, float]) -> str:
    """Keep derived region vectors stable and human-readable in JSON."""
    return "({:.1f} {:.1f} {:.1f})".format(*value)


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
            area_state = fields.get("state", "unknown").lower()
            events[f"area_{actor}_{area_state}"] += 1
        if "coopbot_area_gate" in line:
            fields = message_fields(line)
            phase = fields.get("phase", "unknown").lower()
            events[f"area_gate_{phase}"] += 1
        if "coopbot_objective" in line:
            fields = message_fields(line)
            objective = fields.get("objective", "unknown").lower()
            phase = fields.get("phase", "unknown").lower()
            events[f"objective_{objective}_{phase}"] += 1
        if "coopbot_changelevel_gate" in line:
            fields = message_fields(line)
            phase = fields.get("phase", "unknown").lower()
            events[f"changelevel_gate_{phase}"] += 1
        if "coopbot_player_style" in line:
            fields = message_fields(line)
            events["player_style_updates"] += 1
            if fields.get("confidence") is not None:
                events["player_style_observations"] = max(
                    events.get("player_style_observations", 0),
                    integer(fields.get("observations"), 0),
                )
        if "coopbot_path_failure" in line:
            # A botlib route failure is the lower-level equivalent of the
            # game-side stuck event and must not be reported as idle follow.
            events["stuck"] += 1
            if re.search(r"\bphase=regroup\b", line):
                events["regroup_path_failure"] += 1
        if "coopbot_decision" in line:
            fields = message_fields(line)
            decision = fields.get("decision", "unknown").lower()
            reason = fields.get("reason", "unknown").lower()
            events[f"decision_{decision}"] += 1
            events[f"decision_reason_{reason}"] += 1
        if "coopbot_role" in line:
            fields = message_fields(line)
            role = fields.get("role", "unknown").lower()
            events[f"role_{role}"] += 1
        if "coopbot_regroup" in line and re.search(r"\btraveltype=11\b", line):
            events["elevator_regroup"] += 1
    return events


def read_botlib_map_model(stream: TextIO) -> dict | None:
    """Extract the optional AAS map model and actionable control details."""
    model: dict[str, object] | None = None
    controls: Counter[str] = Counter()
    control_records: list[dict[str, object]] = []
    region_records: list[dict[str, object]] = []
    region_summary_count: int | None = None
    geometry_records: list[dict[str, str]] = []
    control_links: list[dict[str, str]] = []
    unresolved_links: list[dict[str, str]] = []
    map_transitions: list[dict[str, str]] = []
    areas: dict[int, dict[str, object]] = {}
    elevator_edges: list[dict[str, object]] = []
    area_records = 0
    edge_records = 0

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
        elif "coopbot_map_regions" in line:
            fields = message_fields(line)
            region_summary_count = integer(fields.get("count"))
        elif "coopbot_map_region " in line:
            fields = message_fields(line)
            region_records.append({
                "region": integer(fields.get("region")),
                "cluster": integer(fields.get("cluster")),
                "areas": integer(fields.get("areas")),
                "mins": fields.get("mins", ""),
                "maxs": fields.get("maxs", ""),
                "center": fields.get("center", ""),
            })
        elif "coopbot_map_area" in line:
            fields = message_fields(line)
            area_records += 1
            area_number = integer(fields.get("area"))
            if area_number > 0:
                areas[area_number] = {
                    "area": area_number,
                    "cluster": integer(fields.get("cluster"), -1),
                    "flags": integer(fields.get("flags")),
                    "presence": integer(fields.get("presence")),
                    "reachable": integer(fields.get("reachable")),
                    "mins": fields.get("mins", ""),
                    "maxs": fields.get("maxs", ""),
                    "center": fields.get("center", ""),
                }
        elif "coopbot_map_edge" in line:
            fields = message_fields(line)
            edge_records += 1
            edge = {
                "reach": integer(fields.get("reach")),
                "from": integer(fields.get("from")),
                "to": integer(fields.get("to")),
                "traveltype": integer(fields.get("traveltype")),
                "traveltime": integer(fields.get("traveltime")),
                "facenum": integer(fields.get("facenum")),
                "edgenum": integer(fields.get("edgenum")),
                "start": fields.get("start", ""),
                "end": fields.get("end", ""),
            }
            if edge["traveltype"] == 11:
                elevator_edges.append(edge.copy())
        elif "coopbot_map_control_link" in line:
            fields = message_fields(line)
            control_links.append({
                "source": fields.get("source", ""),
                "target": fields.get("target", ""),
                "destination": fields.get("destination", ""),
            })
        elif "coopbot_map_control_unresolved" in line:
            fields = message_fields(line)
            unresolved_links.append({
                "source": fields.get("source", ""),
                "target": fields.get("target", ""),
            })
        elif "coopbot_map_transition" in line:
            fields = message_fields(line)
            map_transitions.append({
                "class": fields.get("class", ""),
                "map": fields.get("map", ""),
                "message": fields.get("message", ""),
                "target": fields.get("target", ""),
            })
        elif "coopbot_map_geometry" in line:
            fields = message_fields(line)
            geometry_records.append({
                "class": fields.get("class", ""),
                "model": fields.get("model", ""),
                "origin": fields.get("origin", ""),
                "model_origin": fields.get("model_origin", ""),
                "mins": fields.get("mins", ""),
                "maxs": fields.get("maxs", ""),
            })
        elif "coopbot_map_control" in line:
            fields = message_fields(line)
            controls[fields.get("class", "<unknown>")] += 1
            control_records.append({
                "class": fields.get("class", ""),
                "model": fields.get("model", ""),
                "target": fields.get("target", ""),
                "targetname": fields.get("targetname", ""),
                "speed": number(fields.get("speed")) or 0.0,
                "height": number(fields.get("height")) or 0.0,
                "lip": number(fields.get("lip")) or 0.0,
                "spawnflags": integer(fields.get("spawnflags")),
                "map": fields.get("map", ""),
                "message": fields.get("message", ""),
                "origin": fields.get("origin", ""),
                "model_origin": fields.get("model_origin", ""),
                "model_mins": fields.get("model_mins", ""),
                "model_maxs": fields.get("model_maxs", ""),
            })

    if model is None:
        return None

    if not region_records and areas:
        # Older botlib builds emit the complete area graph but not the newer
        # aggregate map_region records.  Recover the same coarse regions from
        # the area cluster field so reports remain useful across log versions.
        derived_regions: dict[int, dict[str, object]] = {}
        for area in areas.values():
            cluster = integer(area.get("cluster"), -1)
            if cluster <= 0:
                continue
            region = derived_regions.setdefault(
                cluster,
                {
                    "region": cluster,
                    "cluster": cluster,
                    "areas": 0,
                    "mins": None,
                    "maxs": None,
                    "center_sum": [0.0, 0.0, 0.0],
                },
            )
            region["areas"] = int(region["areas"]) + 1
            mins = vector3(area.get("mins"))
            maxs = vector3(area.get("maxs"))
            center = vector3(area.get("center"))
            if mins is not None:
                current_mins = region["mins"]
                if current_mins is None:
                    region["mins"] = list(mins)
                else:
                    region["mins"] = [
                        min(float(current_mins[index]), mins[index])
                        for index in range(3)
                    ]
            if maxs is not None:
                current_maxs = region["maxs"]
                if current_maxs is None:
                    region["maxs"] = list(maxs)
                else:
                    region["maxs"] = [
                        max(float(current_maxs[index]), maxs[index])
                        for index in range(3)
                    ]
            if center is not None:
                center_sum = region["center_sum"]
                for index in range(3):
                    center_sum[index] += center[index]

        for cluster in sorted(derived_regions):
            region = derived_regions[cluster]
            area_count = int(region["areas"])
            center_sum = region["center_sum"]
            center = tuple(value / area_count for value in center_sum)
            mins = region["mins"]
            maxs = region["maxs"]
            region_records.append({
                "region": cluster,
                "cluster": cluster,
                "areas": area_count,
                "mins": format_vector3(tuple(mins)) if mins is not None else "",
                "maxs": format_vector3(tuple(maxs)) if maxs is not None else "",
                "center": format_vector3(center),
                "derived_from_areas": True,
            })
        if region_records:
            region_summary_count = len(region_records)

    geometry_by_key = {
        (record["class"], record["model"]): record
        for record in geometry_records
    }
    for control in control_records:
        geometry = geometry_by_key.get((control["class"], control["model"]))
        if geometry is not None:
            for field in ("origin", "model_origin", "mins", "maxs"):
                control[field] = geometry[field]
    if not map_transitions:
        # Some Quake II BSP entity lumps omit the target's `map` epair while
        # retaining the full trigger -> target_changelevel graph. Treat that
        # graph as a static transition record instead of reporting a false
        # missing-transition failure.
        for link in control_links:
            if link.get("destination") not in {
                "target_changelevel",
                "trigger_changelevel",
            }:
                continue
            map_transitions.append({
                "class": link.get("destination", ""),
                "map": "",
                "message": "",
                "target": link.get("target", ""),
                "source": "control_link",
            })
    model["area_records"] = area_records
    model["edge_records"] = edge_records
    model["elevator_edge_records"] = len(elevator_edges)
    model["elevator_edges"] = elevator_edges
    elevator_area_ids = {
        area_id
        for edge in elevator_edges
        for area_id in (edge["from"], edge["to"])
        if isinstance(area_id, int) and area_id > 0
    }
    model["elevator_areas"] = [
        areas[area_id]
        for area_id in sorted(elevator_area_ids)
        if area_id in areas
    ]
    region_connectors: list[dict[str, object]] = []
    for edge in elevator_edges:
        from_area = areas.get(edge["from"])
        to_area = areas.get(edge["to"])
        from_cluster = integer(from_area.get("cluster"), -1) if from_area else -1
        to_cluster = integer(to_area.get("cluster"), -1) if to_area else -1
        from_center = vector3(from_area.get("center")) if from_area else None
        to_center = vector3(to_area.get("center")) if to_area else None
        from_height = from_center[2] if from_center is not None else None
        to_height = to_center[2] if to_center is not None else None
        edge["from_cluster"] = from_cluster
        edge["to_cluster"] = to_cluster
        edge["from_height"] = from_height
        edge["to_height"] = to_height
        edge["vertical_delta"] = (
            to_height - from_height
            if from_height is not None and to_height is not None
            else None
        )
        if from_cluster <= 0 or to_cluster <= 0:
            continue
        region_connectors.append({
            "reach": edge["reach"],
            "from_area": edge["from"],
            "to_area": edge["to"],
            "from_cluster": from_cluster,
            "to_cluster": to_cluster,
            "from_height": from_height,
            "to_height": to_height,
            "vertical_delta": edge["vertical_delta"],
            "traveltype": edge["traveltype"],
        })
    model["region_connectors"] = region_connectors
    model["controls"] = dict(sorted(controls.items()))
    model["control_records"] = control_records
    model["regions"] = region_records
    model["region_records"] = len(region_records)
    model["region_summary_count"] = region_summary_count
    model["geometry_records"] = geometry_records
    model["control_links"] = control_links
    model["unresolved_controls"] = len(unresolved_links)
    model["unresolved_control_links"] = unresolved_links
    model["map_transitions"] = map_transitions
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
    bot_entity_ids: set[int] = set()
    bot_projectile_launches = 0
    bot_shots = 0
    bot_monster_shots = 0
    bot_monster_damage_events = 0
    bot_monster_damage_total = 0.0
    player_contacts = 0
    player_block_candidates = 0
    encounter_events = 0
    target_acquired = 0
    target_lost = 0
    regroup_events = 0
    stuck_events = 0
    map_transition_events = 0
    human_player_samples = 0
    human_player_entities: set[int] = set()

    for record in records:
        fields = message_fields(record.get("message"))
        event = record.get("event")

        if event == "bot_snapshot":
            snapshot_count += 1
            state = fields.get("state")
            if state:
                states[state] += 1
            player_entity = integer(fields.get("player"), -1)
            bot_client = integer(fields.get("client"), -1)
            if bot_client >= 0:
                # Quake II entity numbers are client slot + 1.
                bot_entity_ids.add(bot_client + 1)
            player_is_human = integer(fields.get("player_is_human"), 0)
            if player_entity >= 0 and player_is_human == 1:
                human_player_samples += 1
                human_player_entities.add(player_entity)
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
            attacker = integer(fields.get("attacker"), -1)
            if attacker in bot_entity_ids:
                bot_shots += 1
                if fields.get("target_class", "").startswith("monster_"):
                    bot_monster_shots += 1

        if event == "projectile_launch":
            if integer(fields.get("owner"), -1) in bot_entity_ids:
                bot_projectile_launches += 1

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
            if (
                integer(fields.get("attacker"), -1) in bot_entity_ids
                and fields.get("class", "").startswith("monster_")
            ):
                bot_monster_damage_events += 1
                if applied is not None:
                    bot_monster_damage_total += applied

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

        if event == "map_transition":
            map_transition_events += 1

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
        "bot_combat": {
            "entity_ids": sorted(bot_entity_ids),
            "projectile_launches": bot_projectile_launches,
            "shots": bot_shots,
            "monster_shots": bot_monster_shots,
            "monster_damage_events": bot_monster_damage_events,
            "monster_damage_total": bot_monster_damage_total,
        },
        "coop": {
            "human_player_samples": human_player_samples,
            "human_player_entities": sorted(human_player_entities),
            "player_contacts": player_contacts,
            "player_block_candidates": player_block_candidates,
            "encounter_observed": encounter_events,
            "target_acquired": target_acquired,
            "target_lost": target_lost,
            "regroup": regroup_events,
            "stuck": stuck_events,
            "map_transitions": map_transition_events,
        },
        "bot_state_persistence": bot_state_persistence(records),
    }


def bot_state_persistence(records: Iterable[dict]) -> dict:
    """Compare an explicit test marker with the first bot snapshot after transition."""
    markers: dict[int, dict[str, object]] = {}
    comparisons: list[dict[str, object]] = []
    transition_seen = False

    for record in records:
        event = record.get("event")
        fields = message_fields(record.get("message"))
        if event == "bot_state_marker":
            client = integer(fields.get("client"), -1)
            if client >= 0:
                markers[client] = {
                    "client": client,
                    "health": integer(fields.get("bot_health")),
                    "max_health": integer(fields.get("bot_max_health")),
                    "armor": integer(fields.get("bot_armor")),
                    "ammo_index": integer(fields.get("bot_ammo_index")),
                    "ammo": integer(fields.get("bot_ammo")),
                    "weapon": fields.get("weapon", ""),
                }
        elif event == "map_transition":
            transition_seen = True
        elif event == "bot_snapshot" and transition_seen:
            client = integer(fields.get("client"), -1)
            marker = markers.get(client)
            if marker is None or any(
                comparison["client"] == client for comparison in comparisons
            ):
                continue
            snapshot = {
                "client": client,
                "health": integer(fields.get("bot_health")),
                "max_health": integer(fields.get("bot_max_health")),
                "armor": integer(fields.get("bot_armor")),
                "ammo_index": integer(fields.get("bot_ammo_index")),
                "ammo": integer(fields.get("bot_ammo")),
                "weapon": fields.get("weapon", ""),
            }
            fields_to_compare = (
                "health", "max_health", "armor", "ammo_index", "ammo", "weapon"
            )
            matching = all(
                marker[field] == snapshot[field] for field in fields_to_compare
            )
            comparisons.append({
                "client": client,
                "marker": marker,
                "first_post_transition": snapshot,
                "matching": matching,
            })

    return {
        "marker_count": len(markers),
        "comparisons": comparisons,
        "matching_clients": sum(
            1 for comparison in comparisons if comparison["matching"]
        ),
        "ok": bool(markers) and bool(comparisons) and all(
            comparison["matching"] for comparison in comparisons
        ),
    }


def summarise(records: Iterable[dict], invalid_lines: int, source: str) -> dict:
    event_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    episodes: defaultdict[str, dict] = defaultdict(
        lambda: {
            "records": 0,
            "events": {},
            "maps": [],
            "seed": None,
            "first_time": None,
            "last_time": None,
        }
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
        if event == "episode_start":
            episode["seed"] = message_fields(record.get("message")).get("seed")
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
        "--episode-id",
        help="restrict telemetry and validation to one episode_id",
    )
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
        "--require-vertical-elevator-edge",
        action="store_true",
        help="fail unless a TRAVEL_ELEVATOR edge has a non-zero vertical displacement",
    )
    parser.add_argument(
        "--require-elevator-regroup",
        action="store_true",
        help="fail unless botlib logged a regroup frame using TRAVEL_ELEVATOR",
    )
    parser.add_argument(
        "--require-elevator-reacquired",
        action="store_true",
        help="fail unless botlib logged successful elevator reacquisition",
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
        "--require-regroup",
        action="store_true",
        help="fail unless botlib logged a coopbot_regroup frame",
    )
    parser.add_argument(
        "--require-rescue",
        action="store_true",
        help="fail unless botlib logged a player rescue position",
    )
    parser.add_argument(
        "--require-kill-steal-yield",
        action="store_true",
        help="fail unless botlib logged yielding a player-focused target",
    )
    parser.add_argument(
        "--require-target-lost",
        action="store_true",
        help="fail unless runtime telemetry recorded a lost target",
    )
    parser.add_argument(
        "--require-role",
        action="append",
        choices=("REGROUP", "RESCUER", "COVER", "SUPPORT", "FOLLOWER", "VANGUARD", "ANCHOR"),
        help="fail unless botlib selected the named cooperative role (repeatable)",
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
    parser.add_argument(
        "--require-open-path-route",
        action="store_true",
        help="fail unless AAS confirmed a route after OPEN_PATH activation",
    )
    parser.add_argument(
        "--require-open-path-activation",
        action="store_true",
        help="fail unless botlib reached and activated an OPEN_PATH control",
    )
    parser.add_argument(
        "--require-open-path-complete",
        action="store_true",
        help="fail unless botlib completed an OPEN_PATH control objective",
    )
    parser.add_argument(
        "--require-map-transition",
        action="store_true",
        help="fail unless botlib extracted at least one map transition",
    )
    parser.add_argument(
        "--require-runtime-map-transition",
        action="store_true",
        help="fail unless runtime JSONL recorded a map transition",
    )
    parser.add_argument(
        "--require-bot-state-persistence",
        action="store_true",
        help="fail unless marked bot health/armor/ammo/weapon match after transition",
    )
    parser.add_argument(
        "--require-safe-area",
        action="store_true",
        help="fail unless botlib recorded at least one safe area",
    )
    parser.add_argument(
        "--require-decision",
        action="append",
        choices=("role_select", "target_select", "target_yield", "action_select"),
        help="fail unless botlib recorded the named coop decision (repeatable)",
    )
    parser.add_argument(
        "--require-human-player",
        action="store_true",
        help="fail unless bot snapshots reference a real non-bot player entity",
    )
    parser.add_argument(
        "--require-bot-shot",
        action="store_true",
        help="fail unless telemetry contains a projectile/shot owned by the bot",
    )
    parser.add_argument(
        "--require-bot-monster-damage",
        action="store_true",
        help="fail unless the bot applied damage to a monster entity",
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

    if args.episode_id is not None:
        records = [
            record
            for record in records
            if str(record.get("episode_id", "")) == args.episode_id
        ]
        source = f"{source}#episode={args.episode_id}"

    result_object = summarise(records, invalid, source)
    botlib_events: Counter[str] = Counter()
    botlib_map_model: dict | None = None
    validation_errors: list[str] = []

    if (
        args.require_human_player
        and result_object["telemetry"]["coop"]["human_player_samples"] < 1
    ):
        validation_errors.append(
            "no human player entity was observed in bot snapshots"
        )

    if args.require_bot_shot and result_object["telemetry"]["bot_combat"]["shots"] < 1:
        validation_errors.append("no bot-owned shot was recorded")

    if (
        args.require_bot_monster_damage
        and result_object["telemetry"]["bot_combat"]["monster_damage_events"] < 1
    ):
        validation_errors.append("no bot damage to a monster was recorded")

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

    if args.require_vertical_elevator_edge:
        if botlib_map_model is None:
            validation_errors.append("botlib map model is missing")
        elif not any(
            isinstance(edge.get("vertical_delta"), (int, float))
            and abs(edge["vertical_delta"]) > 1.0
            for edge in botlib_map_model.get("elevator_edges", [])
        ):
            validation_errors.append(
                "no vertically separated TRAVEL_ELEVATOR edge was recorded"
            )

    if args.require_elevator_regroup and botlib_events.get("elevator_regroup", 0) < 1:
        validation_errors.append(
            "no coopbot_regroup frame used TRAVEL_ELEVATOR"
        )

    if (
        args.require_elevator_reacquired
        and botlib_events.get("elevator_reacquired", 0) < 1
    ):
        validation_errors.append(
            "no successful elevator reacquisition was recorded by botlib"
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

    if args.require_regroup and botlib_events.get("regroup", 0) < 1:
        validation_errors.append("no coopbot_regroup frame was recorded by botlib")

    if args.require_rescue and botlib_events.get("rescue_position", 0) < 1:
        validation_errors.append(
            "no coopbot_rescue_position frame was recorded by botlib"
        )

    if (
        args.require_kill_steal_yield
        and botlib_events.get("kill_steal_yield", 0) < 1
    ):
        validation_errors.append(
            "no coopbot_kill_steal_yield frame was recorded by botlib"
        )

    if args.require_target_lost and result_object["telemetry"]["coop"]["target_lost"] < 1:
        validation_errors.append("no target_lost frame was recorded")

    for role in args.require_role or []:
        if botlib_events.get(f"role_{role.lower()}", 0) < 1:
            validation_errors.append(
                f"no coopbot_role {role} frame was recorded by botlib"
            )

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
        args.require_open_path_route
        and botlib_events.get("objective_open_path_route_confirmed", 0) < 1
    ):
        validation_errors.append(
            "no AAS route confirmation followed OPEN_PATH activation"
        )

    if (
        args.require_open_path_activation
        and botlib_events.get("objective_open_path_activate", 0) < 1
    ):
        validation_errors.append(
            "no OPEN_PATH control activation was recorded"
        )

    if (
        args.require_open_path_complete
        and botlib_events.get("objective_open_path_complete", 0) < 1
    ):
        validation_errors.append(
            "no completed OPEN_PATH control objective was recorded"
        )

    if args.require_safe_area and botlib_events.get("safe_area", 0) < 1:
        validation_errors.append("no coopbot_safe_area record was emitted")

    for decision in args.require_decision or []:
        if botlib_events.get(f"decision_{decision}", 0) < 1:
            validation_errors.append(
                f"no coopbot_decision {decision.upper()} record was emitted"
            )

    if args.require_map_transition:
        if botlib_map_model is None:
            validation_errors.append("botlib map model is missing")
        elif len(botlib_map_model.get("map_transitions", [])) < 1:
            validation_errors.append("no map transition was extracted")

    if args.require_runtime_map_transition and result_object["telemetry"]["coop"]["map_transitions"] < 1:
        validation_errors.append("no runtime map transition was recorded")

    if args.require_bot_state_persistence:
        persistence = result_object["telemetry"]["bot_state_persistence"]
        if not persistence["ok"]:
            validation_errors.append(
                "marked bot health/armor/ammo/weapon did not persist across transition"
            )

    if (
        args.require_elevator_edge
        or args.require_vertical_elevator_edge
        or args.require_elevator_regroup
        or args.require_elevator_reacquired
        or args.require_regroup_path_failure
        or args.require_regroup_complete
        or args.require_regroup
        or args.require_rescue
        or args.require_kill_steal_yield
        or args.require_target_lost
        or args.require_role
        or args.require_player_area_transition
        or args.require_bot_area_transition
        or args.require_open_path_route
        or args.require_open_path_activation
        or args.require_open_path_complete
        or args.require_safe_area
        or args.require_decision
        or args.require_map_transition
        or args.require_runtime_map_transition
        or args.require_bot_state_persistence
        or args.require_human_player
        or args.require_bot_shot
        or args.require_bot_monster_damage
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
