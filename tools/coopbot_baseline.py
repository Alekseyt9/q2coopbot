"""Aggregate CoopBot report summaries into a reproducible run baseline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


def load_report(path: Path) -> dict:
    """Load one JSON summary produced by coopbot_event_report.py."""
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"report is not an object: {path}")
    if value.get("schema_version") != 1:
        raise ValueError(f"unsupported report schema in {path}")
    return value


def aggregate(
    reports: Iterable[tuple[Path, dict]],
    minimum_runs: int,
    require_human_player: bool = False,
) -> dict:
    """Combine per-run reports without hiding missing or duplicate seeds."""
    run_entries: list[dict] = []
    seeds: list[str] = []
    maps: Counter[str] = Counter()
    events: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    errors: list[str] = []

    for path, report in reports:
        episodes = report.get("episodes")
        if not isinstance(episodes, dict) or not episodes:
            errors.append(f"{path}: no episodes")
            continue
        real_episodes = {
            episode_id: episode
            for episode_id, episode in episodes.items()
            if episode_id != "pending"
        }
        if not real_episodes:
            errors.append(f"{path}: no non-pending episodes")
            continue

        episode_seeds: set[str] = set()
        episode_maps: set[str] = set()
        for episode_id, episode in real_episodes.items():
            if not isinstance(episode, dict):
                errors.append(f"{path}: episode {episode_id} is not an object")
                continue
            seed = episode.get("seed")
            if seed is None or seed == "":
                errors.append(f"{path}: episode {episode_id} has no seed")
            else:
                seed_text = str(seed)
                seeds.append(seed_text)
                episode_seeds.add(seed_text)
            for map_name in episode.get("maps", []):
                map_text = str(map_name)
                maps[map_text] += 1
                episode_maps.add(map_text)

        for event, count in report.get("events", {}).items():
            if isinstance(count, int):
                events[str(event)] += count

        botlib_log = report.get("botlib_log")
        if isinstance(botlib_log, dict):
            for event, count in botlib_log.get("events", {}).items():
                if str(event).startswith("decision_") and isinstance(count, int):
                    decisions[str(event)] += count

        telemetry = report.get("telemetry", {})
        coop = telemetry.get("coop", {}) if isinstance(telemetry, dict) else {}
        human_samples = coop.get("human_player_samples", 0)
        if require_human_player and not isinstance(human_samples, int):
            errors.append(f"{path}: human player sample count is invalid")
            human_samples = 0
        if require_human_player and human_samples < 1:
            errors.append(
                f"{path}: no human player entity was observed in bot snapshots"
            )

        run_entries.append(
            {
                "source": str(path),
                "episodes": len(real_episodes),
                "seeds": sorted(episode_seeds),
                "maps": sorted(episode_maps),
                "records": int(report.get("records", 0)),
                "human_player_samples": int(human_samples),
            }
        )

    duplicate_seeds = sorted(
        seed for seed, count in Counter(seeds).items() if count > 1
    )
    if len(run_entries) < minimum_runs:
        errors.append(
            f"only {len(run_entries)} valid runs were supplied; "
            f"minimum is {minimum_runs}"
        )

    return {
        "schema_version": 1,
        "minimum_runs": minimum_runs,
        "require_human_player": require_human_player,
        "runs": len(run_entries),
        "episodes": sum(entry["episodes"] for entry in run_entries),
        "seeds": sorted(set(seeds)),
        "duplicate_seeds": duplicate_seeds,
        "maps": dict(sorted(maps.items())),
        "events": dict(sorted(events.items())),
        "decisions": dict(sorted(decisions.items())),
        "run_details": run_entries,
        "validation": {"ok": not errors, "errors": errors},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="event report JSON files")
    parser.add_argument(
        "--min-runs",
        type=int,
        default=20,
        help="minimum number of valid reports required (default: 20)",
    )
    parser.add_argument(
        "--require-human-player",
        action="store_true",
        help="reject reports without a real non-bot player entity",
    )
    parser.add_argument("-o", "--output", help="write baseline JSON to this path")
    args = parser.parse_args()
    if args.min_runs < 1:
        parser.error("--min-runs must be positive")

    loaded: list[tuple[Path, dict]] = []
    errors: list[str] = []
    for raw_path in args.reports:
        path = Path(raw_path)
        try:
            loaded.append((path, load_report(path)))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(str(exc))

    result = aggregate(
        loaded,
        args.min_runs,
        require_human_player=args.require_human_player,
    )
    if errors:
        result["validation"]["ok"] = False
        result["validation"]["errors"] = errors + result["validation"]["errors"]

    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if result["validation"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
