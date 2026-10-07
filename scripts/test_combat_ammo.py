import json
import pathlib
import tempfile
import unittest
from report_combat_ammo import episode


class AmmoReportTest(unittest.TestCase):
    def test_death_hud_zero_is_not_spent_ammo(self):
        with tempfile.TemporaryDirectory(dir="F:/src/quake2/q2coopbot-src/workspace/artifacts") as tmp:
            batch = pathlib.Path(tmp)
            run = batch / "worker-0-episode-0"
            run.mkdir()
            rows = [{"frame": f, "health": hp, "ammo": ammo,
                     "weapon": "models/weapons/v_machn/tris.md2"}
                    for f, hp, ammo in [(99, 100, 99), (100, 100, 98),
                                        (101, 2, 97), (102, 0, 0), (103, -1, 0)]]
            (run / "bot.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            result = {"worker": 0, "episode": 0, "seed": 1,
                      "first_life": {"start_frame": 99, "end_frame": 102,
                                     "end_reason": "first_observed_death",
                                     "outgoing_by_target_class": [{"health_damage": 8}]}}
            measured = episode(batch, result)
            self.assertEqual(measured["observed_alive_rounds_consumed"], 2)
            self.assertTrue(measured["terminal_ammo_unknown"])


if __name__ == "__main__":
    unittest.main()
