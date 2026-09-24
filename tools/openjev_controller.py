"""OpenJev tactical decisions for the UDP test player."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
import json
import math
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

from openjev_navigation import AASNavigator, JUMP_TYPES, horizontal_distance


MODEL = "hf.co/apus-ailab/APUS-OpenJev-v1-4B-GGUF:Q8_0"
ORIGIN_RE = re.compile(r"(player|bot)_origin=\((-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\)")
WORLD_ORIGIN_RE = re.compile(r"\borigin=\((-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\)")
VELOCITY_RE = re.compile(r"\bvelocity=\((-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\)")
FIELD_RE = re.compile(r"\b(player_health|player_armor|player_ammo|bot_health|bot_armor|bot_ammo|distance_to_player|target_visible|visible_enemies)=(-?[\d.]+)")


def _vector(pattern: re.Pattern, message: str) -> tuple[float, float, float] | None:
    match = pattern.search(message)
    return tuple(float(match.group(i)) for i in (1, 2, 3)) if match else None


def parse_snapshot(record: dict) -> dict | None:
    if record.get("event") == "udp_snapshot":
        return dict(record["snapshot"])
    if record.get("event") not in ("bot_snapshot", "openjev_snapshot"):
        return None
    message = str(record.get("message", ""))
    if "player_is_human=1" not in message:
        return None
    origins = {
        match.group(1): tuple(float(match.group(i)) for i in (2, 3, 4))
        for match in ORIGIN_RE.finditer(message)
    }
    if "player" not in origins or "bot" not in origins:
        return None
    values = {key: float(value) for key, value in FIELD_RE.findall(message)}
    weapon = re.search(r'\bweapon="([^"]*)"', message)
    player_weapon = re.search(r'\bplayer_weapon="([^"]*)"', message)
    target = re.search(r'\btarget_class="([^"]*)"', message)
    return {"map": record.get("map"), "player_origin": origins["player"],
            "bot_origin": origins["bot"], "weapon": weapon.group(1) if weapon else None,
            "player_weapon": player_weapon.group(1) if player_weapon else None,
            "target_class": target.group(1) if target else None,
            "enemies": [], "pickups": [], **values}


def parse_world_entity(record: dict) -> dict | None:
    if record.get("event") not in ("openjev_enemy", "openjev_pickup"):
        return None
    message = str(record.get("message", ""))
    entity_id = re.search(r"\bid=(\d+)", message)
    entity_class = re.search(r'\bclass="([^"]+)"', message)
    origin = _vector(WORLD_ORIGIN_RE, message)
    if not entity_id or not entity_class or origin is None:
        return None
    result = {"id": int(entity_id.group(1)), "class": entity_class.group(1),
              "origin": origin}
    if record["event"] == "openjev_enemy":
        health = re.search(r"\bhealth=(\d+)", message)
        result["health"] = int(health.group(1)) if health else 0
        result["velocity"] = _vector(VELOCITY_RE, message) or (0.0, 0.0, 0.0)
    return result


def read_latest_snapshot(path: Path, offset: int, episode_id: str | None,
                         previous: dict | None = None) -> tuple[dict | None, int]:
    latest = previous
    snapshot_time = previous.get("_time") if previous else None
    changed = False
    try:
        with path.open("rb") as stream:
            stream.seek(offset)
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    offset = start
                    break
                offset = stream.tell()
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if episode_id and record.get("episode_id") != episode_id:
                    continue
                snapshot = parse_snapshot(record)
                if snapshot is not None:
                    latest = snapshot
                    snapshot_time = record.get("time")
                    latest["_time"] = snapshot_time
                    changed = True
                    continue
                if latest is None or record.get("time") != snapshot_time:
                    continue
                entity = parse_world_entity(record)
                if entity is not None:
                    if not changed:
                        latest = {**latest, "enemies": list(latest["enemies"]),
                                  "pickups": list(latest["pickups"])}
                        changed = True
                    key = "enemies" if record["event"] == "openjev_enemy" else "pickups"
                    latest[key].append(entity)
    except OSError:
        pass
    return (latest if changed else None), offset


def distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return math.dist(first[:2], second[:2])


class StrategicMemory:
    """Slow, persistent goal selection above OpenJev's tactical choices."""

    def __init__(self) -> None:
        self.map_name: str | None = None
        self.goal = "advance"
        self.goal_since = 0.0
        self.goal_reason = "join the player and help with immediate threats"
        self.events: deque[str] = deque(maxlen=8)
        self.last_seen_enemies: dict[int, tuple[str, float]] = {}
        self.last_human_damage_at = -1e9
        self.previous: dict | None = None
        self.version = 0

    def observe(self, snapshot: dict, now: float) -> list[dict]:
        changes: list[dict] = []
        map_name = snapshot.get("map")
        if map_name != self.map_name:
            self.map_name = map_name
            self.events.clear()
            self.last_seen_enemies.clear()
            self.previous = None
            self.goal = "advance"
            self.goal_since = now
            self.goal_reason = "new map"
            self.version += 1
            self.events.append(f"entered map {map_name or 'unknown'}")
            changes.append({"event": "goal_started", "goal": self.goal,
                            "reason": self.goal_reason})

        previous = self.previous
        if previous is not None:
            if snapshot.get("player_health", 0) < previous.get("player_health", 0):
                self.events.append("I took damage")
            ally_health = snapshot.get("bot_health")
            old_ally_health = previous.get("bot_health")
            if (ally_health is not None and old_ally_health is not None
                    and ally_health < old_ally_health):
                self.last_human_damage_at = now
                self.events.append("the human teammate took damage")

        current_ids = {enemy["id"] for enemy in snapshot.get("enemies", [])}
        previous_ids = ({enemy["id"] for enemy in previous.get("enemies", [])}
                        if previous else set())
        for enemy in snapshot.get("enemies", []):
            self.last_seen_enemies[enemy["id"]] = (enemy["class"], now)
            if enemy["id"] not in previous_ids:
                self.events.append(f"spotted {enemy['class']} #{enemy['id']}")
        if previous_ids - current_ids:
            self.events.append("lost sight of an enemy")
        self.last_seen_enemies = {
            entity_id: value for entity_id, value in self.last_seen_enemies.items()
            if now - value[1] <= 15.0
        }

        separation = distance(snapshot["player_origin"], snapshot["bot_origin"])
        health_pickup = any("health" in item["class"]
                            for item in snapshot.get("pickups", []))
        human_threat = any(
            distance(enemy["origin"], snapshot["bot_origin"]) < 520
            for enemy in snapshot.get("enemies", []))
        human_health = snapshot.get("bot_health")
        if human_health is not None and human_health <= 0:
            desired, reason, urgent = "wait", "the human teammate is awaiting respawn", True
        elif snapshot.get("player_health", 0) < 45 and health_pickup:
            desired, reason, urgent = "recover", "my health is low and health is reachable", True
        elif human_threat or now - self.last_human_damage_at < 7.0:
            desired, reason, urgent = "cover", "the human teammate is threatened", True
        elif separation > 220:
            desired, reason, urgent = "regroup", "the human teammate is far away", separation > 500 or self.goal == "wait"
        else:
            desired, reason, urgent = "advance", "stay with the human and progress together", False

        if desired != self.goal and (urgent or now - self.goal_since >= 7.0):
            completed = (
                self.goal == "regroup" and separation <= 220
                or self.goal == "cover" and not human_threat
                and now - self.last_human_damage_at >= 7.0
                or self.goal == "recover" and snapshot.get("player_health", 0) >= 65
                or self.goal == "wait" and human_health is not None and human_health > 0
            )
            changes.append({"event": "goal_completed" if completed else "goal_aborted",
                            "goal": self.goal,
                            "reason": reason})
            self.goal, self.goal_reason, self.goal_since = desired, reason, now
            self.version += 1
            self.events.append(f"plan: {desired} ({reason})")
            changes.append({"event": "goal_started", "goal": desired,
                            "reason": reason})
        self.previous = snapshot
        return changes

    def context(self, now: float) -> dict:
        return {"goal": self.goal, "goal_reason": self.goal_reason,
                "goal_age_s": round(now - self.goal_since, 1),
                "recent_events": list(self.events),
                "remembered_enemies": [
                    {"id": entity_id, "class": kind,
                     "last_seen_s_ago": round(now - seen_at, 1)}
                    for entity_id, (kind, seen_at) in self.last_seen_enemies.items()
                ]}


def candidates(snapshot: dict, goal: str = "advance") -> list[tuple[str, str]]:
    player = snapshot["player_origin"]
    blocked = set(snapshot.get("blocked_targets", []))
    attackable = [enemy for enemy in snapshot.get("enemies", [])
                  if enemy["id"] not in blocked
                  and horizontal_distance(player, enemy["origin"]) <= 320]
    nearby_threats = [enemy for enemy in attackable
                      if horizontal_distance(player, enemy["origin"]) <= 256]
    if goal == "wait":
        weapon = (snapshot.get("player_weapon") or "").lower()
        defense = [(f"attack:{enemy['id']}",
                    f"defend against visible {enemy['class']} {enemy['id']}")
                   for enemy in attackable[:3]]
        if snapshot.get("player_health", 0) <= 25 or not (
                snapshot.get("player_ammo", 0) > 0 or "blaster" in weapon):
            defense = []
        return defense + [("hold", "wait safely for the human teammate to respawn")]
    result = []
    if snapshot.get("player_health", 0) < 65:
        for pickup in snapshot.get("pickups", []):
            if "health" in pickup["class"]:
                result.append((f"heal:{pickup['id']}",
                               f"move to nearby health pickup {pickup['id']}"))
                break
    weapon = (snapshot.get("player_weapon") or "").lower()
    armed = snapshot.get("player_ammo", 0) > 0 or "blaster" in weapon
    teammate_near = horizontal_distance(player, snapshot["bot_origin"]) <= 320
    if (snapshot.get("player_health", 0) >= 40 and armed and teammate_near
            and nearby_threats):
        return [(f"attack:{enemy['id']}",
                 f"protect the teammate by attacking nearby {enemy['class']} "
                 f"{enemy['id']}; switch target if a wall blocks shots")
                for enemy in nearby_threats[:3]]
    if snapshot.get("player_health", 0) > 25 and (
            armed):
        for enemy in attackable[:3]:
            result.append((f"attack:{enemy['id']}",
                           f"aim, fire and strafe against nearby {enemy['class']} "
                           f"{enemy['id']} with {enemy.get('health') or 'unknown'} health"))
    if snapshot.get("enemies") and snapshot.get("player_health", 0) < 35:
        result.append(("retreat", "move away from the nearest visible enemy"))
    separation = distance(snapshot["player_origin"], snapshot["bot_origin"])
    if goal == "regroup" and separation > 160 and not snapshot.get("enemies"):
        return [("follow", "rejoin the human teammate; this is the active plan")]
    result.append(("follow", "move toward the human teammate"))
    result.append(("hold", "hold position and do not fire"))
    return result[:16]


def choose_action(url: str, model: str, snapshot: dict,
                  timeout: float, strategy: dict | None = None
                  ) -> tuple[str, float, str, list[str]]:
    strategy = strategy or {"goal": "advance", "recent_events": []}
    choices = candidates(snapshot, strategy["goal"])
    state = (
        f"Map: {snapshot.get('map') or 'unknown'}. "
        f"Player health {snapshot.get('player_health', 0):.0f}, "
        f"armor {snapshot.get('player_armor', 0):.0f}, "
        f"weapon {snapshot.get('player_weapon') or 'unknown'}, "
        f"ammo {snapshot.get('player_ammo', 0):.0f}, "
        f"position {snapshot['player_origin']}. "
        f"Human teammate health {snapshot.get('bot_health') if snapshot.get('bot_health') is not None else 'unknown'}, "
        f"position {snapshot['bot_origin']}. "
        f"Enemies in network visibility area (may be behind walls): {json.dumps(snapshot.get('enemies', []))}. "
        f"Nearby active pickups: {json.dumps(snapshot.get('pickups', []))}. "
        "Coordinates are current observations, not a navigable route."
    )
    task = {
        "primitive": "choice",
        "instructions": "You are the fast tactical system. Follow the persistent strategic goal. Protect the human teammate, attack only a nearby enemy offered as an attack option, seek health when hurt, and stay nearby. Network visibility does not prove a clear shot. Do not replace the strategic goal with a momentary observation.",
        "criteria": [
            {"label": chr(65 + index), "description": description}
            for index, (_, description) in enumerate(choices)
        ],
    }
    labels = ", ".join(chr(65 + index) for index in range(len(choices)))
    prompt = ("Shared state:\n" + state + "\nStrategic memory: "
              + json.dumps(strategy, ensure_ascii=False) + "\n\n"
              + json.dumps(task, ensure_ascii=False, sort_keys=True)
              + f"\nReturn only the selected letter: {labels}.\nAnswer:")
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False,
                          "think": False, "options": {"temperature": 0, "num_predict": 1}}).encode()
    request = Request(url.rstrip("/") + "/api/generate", payload,
                      {"Content-Type": "application/json"})
    started = time.monotonic()
    with urlopen(request, timeout=timeout) as response:
        answer = json.load(response).get("response", "")
    match = re.search(r"\b([A-P])\b", answer.strip().upper())
    if not match or ord(match.group(1)) - 65 >= len(choices):
        raise ValueError(f"OpenJev returned no valid action: {answer[:80]!r}")
    return (choices[ord(match.group(1)) - 65][0],
            time.monotonic() - started, answer,
            [name for name, _ in choices])


class OpenJevController:
    def __init__(self, path: Path, episode_id: str | None, offset: int,
                 url: str, model: str, interval: float, timeout: float,
                 trace_path: Path | None = None):
        self.path, self.episode_id, self.offset = path, episode_id, offset
        self.url, self.model, self.interval, self.timeout = url, model, interval, timeout
        self.trace_path = trace_path
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openjev")
        self.pending: Future | None = None
        self.snapshot: dict | None = None
        self.snapshot_at = 0.0
        self.memory = StrategicMemory()
        self.pending_goal_version = -1
        self.action = "hold"
        self.next_decision_at = 0.0
        self.decisions = 0
        self.errors: list[str] = []
        self.navigator: AASNavigator | None = None
        self.navigation_map: str | None = None
        self.waypoints: list[tuple[tuple[float, float, float], int]] = []
        self.waypoint_index = 0
        self.route_target: tuple[float, float, float] | None = None
        self.route_at = 0.0
        self.last_motion_origin: tuple[float, float, float] | None = None
        self.last_motion_at = 0.0
        self.detour_until = 0.0
        self.detour_side = 200
        self.blocked_targets: dict[int, float] = {}
        self.attack_started_at = 0.0
        self.attack_origin: tuple[float, float, float] | None = None
        self.attack_target: int | None = None

    def _block_target(self, entity_id: int, reason: str, now: float) -> None:
        self.blocked_targets[entity_id] = now + 8.0
        self.action = "follow"
        self.next_decision_at = now
        self._trace({"event": "attack_blocked", "target": entity_id,
                     "reason": reason})

    def _observe_wall_impacts(self, snapshot: dict, now: float) -> None:
        if not self.action.startswith("attack:"):
            return
        try:
            entity_id = int(self.action.split(":", 1)[1])
        except ValueError:
            return
        enemy = next((item for item in snapshot.get("enemies", [])
                      if item["id"] == entity_id), None)
        if enemy is None:
            return
        start, end = snapshot["player_origin"], enemy["origin"]
        vector = tuple(end[i] - start[i] for i in range(3))
        length2 = sum(axis * axis for axis in vector)
        if length2 < 1:
            return
        for impact in snapshot.get("wall_impacts", []):
            direction = tuple(impact[i] - start[i] for i in range(3))
            fraction = sum(direction[i] * vector[i] for i in range(3)) / length2
            if not 0.05 < fraction < 0.9:
                continue
            miss = math.sqrt(sum((direction[i] - fraction * vector[i]) ** 2
                                 for i in range(3)))
            if miss < 64:
                self._block_target(entity_id, "shot hit world before target", now)
                break

    def tick(self) -> None:
        snapshot, self.offset = read_latest_snapshot(
            self.path, self.offset, self.episode_id, self.snapshot)
        if snapshot is not None:
            now = time.monotonic()
            self._observe_wall_impacts(snapshot, now)
            self.blocked_targets = {entity_id: until
                                    for entity_id, until in self.blocked_targets.items()
                                    if until > now}
            snapshot["blocked_targets"] = list(self.blocked_targets)
            self.snapshot = snapshot
            self.snapshot_at = now
            for change in self.memory.observe(snapshot, self.snapshot_at):
                self._trace(change)
                if change["event"] == "goal_started":
                    self.action = "hold"
        if self.pending is not None and self.pending.done():
            try:
                action, latency, raw, choices = self.pending.result()
                self.decisions += 1
                if self.pending_goal_version != self.memory.version:
                    self._trace({"event": "stale_tactical_decision",
                                 "discarded_action": action})
                else:
                    self.action = ("follow" if action.startswith("attack:") and
                                   int(action.split(":", 1)[1]) in self.blocked_targets
                                   else action)
                self._trace({"action": self.action, "latency_s": latency,
                             "raw": raw, "candidates": choices,
                             "strategy": self.memory.context(time.monotonic()),
                             "snapshot": self.snapshot})
            except Exception as exc:
                self.errors.append(str(exc))
                self.action = "hold"
                self._trace({"error": str(exc), "snapshot": self.snapshot})
            self.pending = None
        now = time.monotonic()
        if now - self.snapshot_at > 2.5:
            self.action = "hold"
            return
        if self.snapshot and self.pending is None and now >= self.next_decision_at:
            self.pending_goal_version = self.memory.version
            self.pending = self.executor.submit(choose_action, self.url, self.model,
                                                self.snapshot.copy(), self.timeout,
                                                self.memory.context(now))
            self.next_decision_at = now + self.interval

    def command(self, previous_yaw: int, use: bool) -> tuple[int, ...]:
        snapshot = self.snapshot
        if not snapshot or time.monotonic() - self.snapshot_at > 2.5:
            return (0, previous_yaw, 0, 0, 0, 0, 0, 0, 50, 0)
        if snapshot.get("player_health", 0) <= 0:
            # Quake II uses the attack button to leave the death screen.
            return (0, previous_yaw, 0, 0, 0, 0, 1, 0, 50, 0)
        player = snapshot["player_origin"]
        action, _, raw_id = self.action.partition(":")
        if action == "attack" and (int(raw_id) in self.blocked_targets or
                                   not any(str(enemy["id"]) == raw_id and
                                           horizontal_distance(player, enemy["origin"]) <= 320
                                           for enemy in snapshot["enemies"])):
            self.action = "follow"
            action = "follow"
        now = time.monotonic()
        if action == "attack":
            entity_id = int(raw_id)
            if entity_id != self.attack_target:
                self.attack_target = entity_id
                self.attack_origin = player
                self.attack_started_at = now
            elif (self.attack_origin is not None and
                  now - self.attack_started_at > 2.0 and
                  horizontal_distance(player, self.attack_origin) < 12):
                enemy = next((item for item in snapshot["enemies"]
                              if item["id"] == entity_id), None)
                if enemy and horizontal_distance(player, enemy["origin"]) > 160:
                    self._block_target(entity_id, "no movement while attacking", now)
                    action = "follow"
        else:
            self.attack_target = None
        if action == "attack":
            weapon = (snapshot.get("player_weapon") or "").lower()
            if (snapshot.get("player_health", 0) <= 25 or
                    not (snapshot.get("player_ammo", 0) > 0 or "blaster" in weapon)):
                return (0, previous_yaw, 0, 0, 0, 0, 0, 0, 50, 0)
        target = None
        if action == "attack":
            target = next((enemy for enemy in snapshot["enemies"]
                           if str(enemy["id"]) == raw_id), None)
        elif action == "heal":
            target = next((item for item in snapshot["pickups"]
                           if str(item["id"]) == raw_id and "health" in item["class"]), None)
        elif action == "follow":
            target = {"origin": snapshot["bot_origin"]}
        elif action == "retreat" and snapshot["enemies"]:
            enemy = snapshot["enemies"][0]["origin"]
            target = {"origin": (2 * player[0] - enemy[0],
                                 2 * player[1] - enemy[1], player[2])}
        if target is None:
            return (0, previous_yaw, 0, 0, 0, 0, 0, 0, 50, 0)
        goal = target["origin"]
        travel_type = 2
        if action in ("follow", "heal"):
            threshold = 32 if action == "heal" else (
                80 if self.memory.goal == "regroup" else 64)
            if horizontal_distance(player, goal) < threshold:
                return (0, previous_yaw, 0, 0, 0, 0, 0, 0, 50, 0)
            goal, travel_type = self._navigation_goal(player, goal, snapshot)
        dx, dy = goal[0] - player[0], goal[1] - player[1]
        horizontal = math.hypot(dx, dy)
        delta_angles = snapshot.get("delta_angles")
        if delta_angles is not None:
            yaw = int(math.degrees(math.atan2(dy, dx)) * 65536 / 360) - delta_angles[1]
        else:
            yaw = int((math.degrees(math.atan2(-dy, -dx)) % 360) * 65536 / 360)
        yaw = ((yaw + 32768) % 65536) - 32768
        if action == "attack":
            dz = goal[2] + 24 - (player[2] + 22)
            pitch = int(-math.degrees(math.atan2(dz, max(horizontal, 1))) * 65536 / 360)
            if delta_angles is not None:
                pitch -= delta_angles[0]
            pitch = ((pitch + 32768) % 65536) - 32768
            forward = 180 if horizontal > 128 else 0
            return (pitch, yaw, 0, forward, 100, 0, 1, 0, 50, 0)
        side = self.detour_side if time.monotonic() < self.detour_until else 0
        up = 200 if travel_type in JUMP_TYPES and goal[2] - player[2] > 12 else 0
        return (0, yaw, 0, 400, side, up, 2 if use else 0, 0, 50, 0)

    def _navigation_goal(self, player: tuple[float, float, float],
                         target: tuple[float, float, float],
                         snapshot: dict) -> tuple[tuple[float, float, float], int]:
        now = time.monotonic()
        map_name = snapshot.get("map") or ""
        if map_name != self.navigation_map:
            self.navigation_map = map_name
            self.navigator = None
            self.waypoints = []
            if re.fullmatch(r"[A-Za-z0-9_-]+", map_name):
                path = self.path.parent / "baseq2" / "maps" / f"{map_name}.aas"
                try:
                    self.navigator = AASNavigator(path)
                    self._trace({"event": "navigation_loaded", "map": map_name,
                                 "areas": len(self.navigator.areas)})
                except (OSError, ValueError) as exc:
                    self._trace({"event": "navigation_unavailable", "error": str(exc)})
        if self.navigator is None:
            return target, 2

        if self.last_motion_origin is None or horizontal_distance(
                player, self.last_motion_origin) > 12:
            self.last_motion_origin = player
            self.last_motion_at = now
        elif now - self.last_motion_at > 2.5 and now >= self.detour_until:
            self._trace({"event": "navigation_stuck", "origin": player,
                         "waypoint_index": self.waypoint_index})
            self.detour_side = -self.detour_side
            self.detour_until = now + 0.8
            self.last_motion_at = now
            self.waypoints = []

        if (not self.waypoints or self.route_target is None
                or horizontal_distance(target, self.route_target) > 80
                or now - self.route_at > 4.0):
            self.waypoints = self.navigator.route(player, target)
            self.waypoint_index = 0
            self.route_target = target
            self.route_at = now
            self._trace({"event": "navigation_route", "map": map_name,
                         "waypoints": len(self.waypoints),
                         "from": player, "to": target})

        while self.waypoint_index < len(self.waypoints):
            waypoint, travel_type = self.waypoints[self.waypoint_index]
            if horizontal_distance(player, waypoint) <= 10 and abs(
                    player[2] - waypoint[2]) <= 64:
                self.waypoint_index += 1
                continue
            return waypoint, travel_type
        return target, 2

    def _trace(self, record: dict) -> None:
        if self.trace_path:
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
