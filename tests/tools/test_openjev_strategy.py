"""Focused checks for OpenJev's persistent strategic state."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from openjev_controller import OpenJevController, StrategicMemory, candidates, parse_snapshot
from openjev_navigation import AASNavigator


BASE1_AAS = Path(r"F:\src\quake2\q2coopbot-runtime-bot\baseq2\maps\base1.aas")


def world(*, self_pos=(0.0, 0.0, 0.0), human_pos=(100.0, 0.0, 0.0),
          self_hp=100, human_hp=100, enemies=None, pickups=None):
    return {"map": "base1", "player_origin": self_pos,
            "bot_origin": human_pos, "player_health": self_hp,
            "bot_health": human_hp, "player_weapon": "Blaster",
            "player_ammo": 0, "enemies": enemies or [], "pickups": pickups or []}


class StrategyTests(unittest.TestCase):
    def test_regroup_is_held_until_arrival_and_commitment(self):
        memory = StrategicMemory()
        changes = memory.observe(world(human_pos=(600.0, 0.0, 0.0)), 1.0)
        self.assertEqual(memory.goal, "regroup")
        self.assertIn("goal_started", [item["event"] for item in changes])
        self.assertEqual(candidates(world(human_pos=(600.0, 0.0, 0.0)),
                                    memory.goal)[0][0], "follow")
        memory.observe(world(human_pos=(120.0, 0.0, 0.0)), 3.0)
        self.assertEqual(memory.goal, "regroup")
        changes = memory.observe(world(human_pos=(120.0, 0.0, 0.0)), 9.0)
        self.assertEqual(memory.goal, "advance")
        self.assertEqual(changes[0]["event"], "goal_completed")

    def test_human_damage_interrupts_and_recent_threat_persists(self):
        memory = StrategicMemory()
        memory.observe(world(), 1.0)
        memory.observe(world(human_hp=75), 2.0)
        self.assertEqual(memory.goal, "cover")
        self.assertIn("the human teammate took damage", memory.context(2.0)["recent_events"])
        memory.observe(world(human_hp=75), 5.0)
        self.assertEqual(memory.goal, "cover")
        memory.observe(world(human_hp=75), 10.0)
        self.assertEqual(memory.goal, "advance")

    def test_enemy_memory_expires_without_rewriting_current_observation(self):
        memory = StrategicMemory()
        enemy = {"id": 31, "class": "monster_soldier", "origin": (50.0, 0.0, 0.0)}
        memory.observe(world(enemies=[enemy]), 1.0)
        memory.observe(world(), 2.0)
        self.assertEqual(memory.context(2.0)["remembered_enemies"][0]["id"], 31)
        memory.observe(world(), 17.0)
        self.assertEqual(memory.context(17.0)["remembered_enemies"], [])

    def test_waits_for_human_respawn_then_regroups(self):
        memory = StrategicMemory()
        memory.observe(world(human_hp=-4), 1.0)
        self.assertEqual(memory.goal, "wait")
        self.assertEqual(candidates(world(human_hp=-4), memory.goal)[0][0], "hold")
        memory.observe(world(human_pos=(600.0, 0.0, 0.0)), 2.0)
        self.assertEqual(memory.goal, "regroup")

    def test_negative_health_is_parsed_for_respawn(self):
        snapshot = parse_snapshot({"event": "openjev_snapshot", "map": "base1",
                                   "message": "player_is_human=1 player_health=-999 "
                                   "player_origin=(0 0 0) bot_origin=(10 0 0)"})
        self.assertEqual(snapshot["player_health"], -999)

    @unittest.skipUnless(BASE1_AAS.exists(), "live base1 AAS asset not installed")
    def test_aas_route_goes_around_base1_wall(self):
        navigator = AASNavigator(BASE1_AAS)
        start = (239.9, -293.0, 24.1)
        human = (-58.1, 30.9, 24.1)
        waypoints = navigator.route(start, human)
        self.assertGreaterEqual(len(waypoints), 6)
        self.assertLess(waypoints[0][0][0], start[0])
        self.assertLess(abs(waypoints[0][0][1] - start[1]), 20)
        self.assertGreater(waypoints[2][0][1], waypoints[0][0][1] + 100)

    @unittest.skipUnless(BASE1_AAS.exists(), "live base1 AAS asset not installed")
    def test_follow_command_uses_aas_waypoint(self):
        controller = OpenJevController(BASE1_AAS.parents[2] / "coopbot_debug_events.jsonl",
                                       None, 0, "http://127.0.0.1:11434", "unused", 1, 1)
        try:
            controller.snapshot = world(self_pos=(239.9, -293.0, 24.1),
                                        human_pos=(-58.1, 30.9, 24.1))
            controller.snapshot_at = __import__("time").monotonic()
            controller.action = "follow"
            command = controller.command(0, False)
            self.assertGreaterEqual(len(controller.waypoints), 6)
            self.assertLess(abs(command[1]), 5000)
            self.assertEqual(command[3], 400)
        finally:
            controller.close()

    @unittest.skipUnless(BASE1_AAS.exists(), "live base1 AAS asset not installed")
    def test_narrow_base1_turn_does_not_skip_portal(self):
        controller = OpenJevController(BASE1_AAS.parents[2] / "coopbot_debug_events.jsonl",
                                       None, 0, "http://127.0.0.1:11434", "unused", 1, 1)
        try:
            start = (-803.9, 943.9, -23.9)
            goal = (-690.0, 877.0, 40.1)
            waypoint, _ = controller._navigation_goal(start, goal, world(self_pos=start,
                                                                          human_pos=goal))
            self.assertLessEqual(controller.waypoint_index, 1)
            self.assertGreater(waypoint[0], start[0])
        finally:
            controller.close()

    def test_follows_at_ninety_units_instead_of_idling(self):
        controller = OpenJevController(BASE1_AAS.parents[2] / "coopbot_debug_events.jsonl",
                                       None, 0, "http://127.0.0.1:11434", "unused", 1, 1)
        try:
            controller.snapshot = world(self_pos=(128.0, -320.0, 24.1),
                                        human_pos=(32.0, -224.0, 24.1))
            controller.snapshot_at = __import__("time").monotonic()
            controller.action = "follow"
            self.assertEqual(controller.command(0, False)[3], 400)
        finally:
            controller.close()

    def test_distant_network_enemy_is_not_an_attack_choice(self):
        enemy = {"id": 31, "class": "monster_soldier",
                 "origin": (500.0, 0.0, 0.0), "health": None}
        self.assertNotIn("attack:31", [choice for choice, _ in
                                     candidates(world(enemies=[enemy]))])

    def test_healthy_companion_engages_nearby_threat(self):
        enemy = {"id": 31, "class": "monster_soldier",
                 "origin": (220.0, 0.0, 0.0), "health": None}
        snapshot = world(enemies=[enemy])
        self.assertEqual([choice for choice, _ in candidates(snapshot)],
                         ["attack:31"])
        snapshot["blocked_targets"] = [31]
        self.assertEqual([choice for choice, _ in candidates(snapshot)],
                         ["follow", "hold"])

    def test_wall_impact_blocks_attack_and_follows_human(self):
        controller = OpenJevController(BASE1_AAS.parents[2] / "coopbot_debug_events.jsonl",
                                       None, 0, "http://127.0.0.1:11434", "unused", 1, 1)
        try:
            enemy = {"id": 31, "class": "monster_soldier",
                     "origin": (250.0, 0.0, 0.0), "health": None}
            snapshot = world(enemies=[enemy], human_pos=(0.0, 100.0, 0.0))
            snapshot["wall_impacts"] = [(100.0, 0.0, 0.0)]
            controller.action = "attack:31"
            controller._observe_wall_impacts(snapshot, __import__("time").monotonic())
            self.assertEqual(controller.action, "follow")
            self.assertIn(31, controller.blocked_targets)
        finally:
            controller.close()


if __name__ == "__main__":
    unittest.main()
