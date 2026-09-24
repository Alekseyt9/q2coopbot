"""Prompt for a slow companion planner selecting from supplied plan options."""

import json


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
