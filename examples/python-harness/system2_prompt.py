"""Prompt for a slow companion planner selecting from supplied plan options."""

import json
import math


RULES = (
    "You plan for a cooperative Quake II companion. Choose exactly "
    "one option using only the state. Interpret fields literally:\n"
    "- human_distance is the measured distance to a PRESENT human. "
    "A numeric distance proves the human exists.\n"
    "- human_hp=null means health UNKNOWN; it does not mean dead, "
    "missing, or absent. Only numeric human_hp<=0 proves death.\n"
    "- An entry in corpses is a dead ENEMY, never the human.\n"
    "- current_plan is an old plan, not evidence that waiting or "
    "covering is still needed.\n"
    "- events, when present, contains only the highest available priority "
    "tier. Omitted lower-priority events are not proof that they never occurred.\n"
    "Apply these rules in order:\n"
    "1. If human_hp is numeric and <=0, choose wait. Otherwise "
    "never choose wait or hold.\n"
    "2. If bot_hp<45 and a health pickup is nearby, choose recover.\n"
    "3. If a living enemy threatens the human, choose cover: "
    "engage only when clear_shot=true; if clear_shot=false, "
    "reposition. If alive or clear_shot is unknown, treat a nearby "
    "enemy as a possible threat and reposition to assess it; do "
    "not engage without a confirmed living enemy and clear shot.\n"
    "4. If a route has failed repeatedly and an alternative route "
    "exists, choose reroute.\n"
    "5. With no urgent issue, choose regroup only if "
    "human_distance>220; otherwise choose advance and follow the "
    "human. Ignore dead enemies.\n"
    "Return JSON with choice and a reason under 15 words. The reason "
    "must cite a field and its actual value; do not invent facts.\n"
)


def build_prompt(state: dict, options: dict[str, str]) -> str:
    """Explain telemetry semantics before asking for a short justified choice."""
    return (RULES + "State: " + json.dumps(state, ensure_ascii=False)
            + "\nOptions: " + json.dumps(options, ensure_ascii=False))


def _distance(first: list | tuple, second: list | tuple) -> int:
    return round(math.dist(first[:2], second[:2]))


def compact_world_state(snapshot: dict, strategy: dict | None = None,
                        max_events: int = 3) -> dict:
    """Send core facts and up to `max_events` from the highest available tier.

    The UDP protocol calls the companion's entity `player` and its human peer
    `bot`. The names here reflect their actual roles for the planner.
    """
    if max_events < 1:
        raise ValueError("max_events must be positive")
    bot = snapshot["player_origin"]
    human = snapshot.get("bot_origin")
    bot_hp = snapshot.get("player_health")
    human_hp = snapshot.get("bot_health")
    human_distance = _distance(bot, human) if human else None
    state = {
        "map": snapshot.get("map"),
        "bot_hp": bot_hp,
        "human_hp": human_hp,
        "human_distance": human_distance,
        "bot_weapon": snapshot.get("player_weapon"),
        "bot_ammo": snapshot.get("player_ammo"),
    }
    if strategy:
        state["current_plan"] = strategy.get("goal")
    high: list[tuple[int, int, dict]] = []
    medium: list[tuple[int, int, dict]] = []
    low: list[tuple[int, int, dict]] = []
    if isinstance(human_hp, (int, float)) and human_hp <= 0:
        high.append((0, 0, {"type": "human_dead", "human_hp": human_hp}))

    health = []
    resources = []
    for pickup in snapshot.get("pickups", []):
        origin = pickup.get("origin")
        if not origin:
            continue
        item = {"id": pickup["id"], "class": pickup.get("class"),
                "distance": _distance(bot, origin)}
        if "health" in (pickup.get("class") or ""):
            health.append(item)
        elif any(word in (pickup.get("class") or "")
                 for word in ("ammo", "weapon")):
            resources.append(item)
    health.sort(key=lambda item: item["distance"])
    resources.sort(key=lambda item: item["distance"])
    nearest_health = health[0] if health else None
    if isinstance(bot_hp, (int, float)) and bot_hp < 45:
        event = {"type": "bot_low_health", "bot_hp": bot_hp}
        if nearest_health and nearest_health["distance"] <= 200:
            event["health_pickup"] = nearest_health
        high.append((1, int(bot_hp), event))

    blocked = set(snapshot.get("blocked_targets", []))
    for enemy in snapshot.get("enemies", []):
        if enemy.get("death_animation") or enemy.get("alive") is False:
            continue
        origin = enemy.get("origin")
        if not origin:
            continue
        health = enemy.get("health")
        alive = (enemy.get("alive") if isinstance(enemy.get("alive"), bool)
                 else health > 0 if isinstance(health, (int, float)) else None)
        item = {"type": "nearby_enemy", "id": enemy["id"],
                "class": enemy.get("class"),
                "distance_bot": _distance(bot, origin),
                "distance_human": _distance(human, origin) if human else None,
                "alive": alive,
                "clear_shot": enemy.get("clear_shot") if isinstance(
                    enemy.get("clear_shot"), bool) else None}
        if enemy["id"] in blocked:
            item["recent_shot_blocked"] = True
        threat_distance = (item["distance_human"]
                           if item["distance_human"] is not None
                           else item["distance_bot"])
        if threat_distance <= 200 or item["distance_bot"] <= 256:
            high.append((2, threat_distance, item))
        else:
            item["type"] = "enemy_visible_far"
            medium.append((3, threat_distance, item))

    if strategy and strategy.get("route_failed"):
        high.append((3, 0, {"type": "route_blocked",
                            "alternative_route": strategy.get(
                                "available_alternative_route")}))
    if human_distance is not None and human_distance > 220:
        medium.append((0, human_distance,
                       {"type": "human_far", "distance": human_distance}))
    ammo = snapshot.get("player_ammo")
    weapon = (snapshot.get("player_weapon") or "").lower()
    if isinstance(ammo, (int, float)) and ammo <= 5 and "blaster" not in weapon:
        medium.append((1, int(ammo),
                       {"type": "bot_low_ammo", "bot_ammo": ammo}))
    if nearest_health and nearest_health["distance"] <= 300:
        medium.append((2, nearest_health["distance"],
                       {"type": "health_available", **nearest_health}))
    for item in resources[:2]:
        if item["distance"] <= 300:
            medium.append((4, item["distance"],
                           {"type": "resource_available", **item}))
    low.append((0, 0, {"type": "stay_with_human" if human_distance is not None
                       else "human_location_unknown"}))

    for tier_name, events in (("high", high), ("medium", medium), ("low", low)):
        if events:
            events.sort(key=lambda entry: (entry[0], entry[1]))
            state["event_tier"] = tier_name
            state["events"] = [entry[2] for entry in events[:max_events]]
            if len(events) > max_events:
                state["events_omitted"] = len(events) - max_events
            break
    return state
