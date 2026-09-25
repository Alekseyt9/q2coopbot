"""Priority and unknown-value checks for compact System 2 telemetry."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from system2_prompt import compact_world_state


def snapshot(**changes):
    base = {"map": "base1", "player_origin": [0, 0, 0],
            "bot_origin": [300, 0, 0], "player_health": 100,
            "player_weapon": "shotgun", "player_ammo": 20,
            "enemies": [], "pickups": []}
    return {**base, **changes}


class CompactWorldStateTests(unittest.TestCase):
    def test_urgent_event_displaces_medium_and_preserves_unknown_health(self):
        state = compact_world_state(snapshot(
            enemies=[{"id": 7, "class": "monster_soldier",
                      "origin": [310, 0, 0], "health": None}],
            pickups=[{"id": 8, "class": "ammo_shells",
                      "origin": [20, 0, 0]}]))
        self.assertIsNone(state["human_hp"])
        self.assertEqual(state["human_distance"], 300)
        self.assertEqual(state["event_tier"], "high")
        self.assertEqual([event["type"] for event in state["events"]],
                         ["nearby_enemy"])
        self.assertIsNone(state["events"][0]["alive"])
        self.assertIsNone(state["events"][0]["clear_shot"])

    def test_medium_used_when_no_urgent_event(self):
        state = compact_world_state(snapshot())
        self.assertEqual(state["event_tier"], "medium")
        self.assertEqual(state["events"][0]["type"], "human_far")

    def test_low_used_for_quiet_nearby_human(self):
        state = compact_world_state(snapshot(bot_origin=[50, 0, 0]))
        self.assertEqual(state["event_tier"], "low")
        self.assertEqual(state["events"], [{"type": "stay_with_human"}])

    def test_priority_and_limit_keep_death_first(self):
        state = compact_world_state(snapshot(
            bot_health=0,
            enemies=[{"id": i, "origin": [300 + i, 0, 0]}
                     for i in range(6)]), max_events=2)
        self.assertEqual(len(state["events"]), 2)
        self.assertEqual(state["events"][0]["type"], "human_dead")
        self.assertEqual(state["events_omitted"], 5)

    def test_zero_event_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            compact_world_state(snapshot(), max_events=0)


if __name__ == "__main__":
    unittest.main()
