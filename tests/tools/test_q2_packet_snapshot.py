"""Protocol 34 packet decoding against known wire layouts."""

import struct
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from q2_packet_snapshot import PacketSnapshots


def config(index, value):
    return b"\x0d" + struct.pack("<H", index) + value.encode() + b"\0"


def entity(number, model, origin):
    # U_ORIGIN1 | U_ORIGIN2 | U_MOREBITS1, then U_ORIGIN3 | U_MODEL.
    return (b"\x83\x0a" + bytes([number, model])
            + struct.pack("<hhh", *(round(axis * 8) for axis in origin)))


class PacketSnapshotTests(unittest.TestCase):
    def test_full_frame_uses_network_state_and_model_configstrings(self):
        decoder = PacketSnapshots()
        serverdata = (b"\x0c" + struct.pack("<iiB", 34, 7, 0)
                      + b"baseq2\0" + struct.pack("<h", 0) + b"Outer Base\0")
        decoder.parse(serverdata + config(33, "maps/base1.bsp")
                      + config(30, "4")
                      + config(34, "models/monsters/soldier/tris.md2")
                      + config(35, "models/items/healing/medium/tris.md2")
                      + config(36, "models/weapons/v_blast/tris.md2"))
        stats = (1 << 1) | (1 << 3) | (1 << 5)
        frame = (b"\x14" + struct.pack("<iiBB", 10, -1, 0, 0) + b"\x11"
                 + struct.pack("<HhhhhhhB", 0x1042, 800, -160, 192,
                               0, 8192, 0, 4)
                 + struct.pack("<Ihhh", stats, 83, 0, 25) + b"\x12"
                 + entity(2, 255, (32, -16, 24))
                 + entity(5, 2, (90, 0, 24))
                 + entity(6, 3, (80, 10, 24)) + b"\x00\x00")
        frames = decoder.parse(frame)
        self.assertEqual(len(frames), 1)
        snapshot = decoder.snapshot(frames[0])
        self.assertEqual(snapshot["map"], "base1")
        self.assertEqual(snapshot["player_origin"], (100.0, -20.0, 24.0))
        self.assertEqual(snapshot["bot_origin"], (32.0, -16.0, 24.0))
        self.assertEqual(snapshot["player_health"], 83)
        self.assertEqual(snapshot["player_weapon"], "Blaster")
        self.assertEqual(snapshot["delta_angles"], (0, 8192, 0))
        self.assertEqual(snapshot["enemies"][0]["class"], "monster_soldier")
        self.assertIsNone(snapshot["enemies"][0]["health"])
        self.assertEqual(snapshot["pickups"][0]["class"], "item_health")

    def test_blaster_wall_impact_is_decoded_before_frame(self):
        decoder = PacketSnapshots()
        decoder.parse(b"\x03\x02" + struct.pack("<hhhB", 800, -80, 16, 0))
        self.assertEqual(decoder.wall_impacts, [(100.0, -10.0, 2.0)])


if __name__ == "__main__":
    unittest.main()
