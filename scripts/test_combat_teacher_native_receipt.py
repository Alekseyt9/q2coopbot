import copy
import json
import tempfile
import unittest
from pathlib import Path

from combat_teacher_native_receipt import admitted_terminal_receipt


class AdmittedReceiptTests(unittest.TestCase):
    def fixture(self, root):
        rows = [dict(frame=f, map='cool1', spawncount=4, self_entity=1,
                     arbitration=dict(aim_entity=8), enemies=[dict(id=8, **{'class': 'monster_gunner'})],
                     sent_command=dict(Buttons=int(f == 3))) for f in (1, 2, 3)]
        path = root/'bot.jsonl'
        path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
        result = dict(root=str(root), goal_stop=None, death_stop=None,
                      first_life=dict(start_frame=1, end_frame=3, map='cool1', spawncount=4,
                                      bot_entity=1, minimum_health=70,
                                      aim_by_class=[dict({'class': 'monster_gunner'}, aim_frames=3, attack_command_frames=1)]))
        return rows, result

    def test_only_reconstructed_preparation_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, result = self.fixture(root)
            original = copy.deepcopy(result)
            receipt, proof = admitted_terminal_receipt(result, 2)
            self.assertEqual(result, original)
            self.assertEqual(proof['preparation_counts'][0]['aim_frames'], 1)
            self.assertEqual(receipt['first_life']['aim_by_class'][0]['aim_frames'], 2)
            # Changing an admitted attack remains visible, never normalized away.
            rows[1]['sent_command']['Buttons'] = 1
            (root/'bot.jsonl').write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            result['first_life']['aim_by_class'][0]['attack_command_frames'] = 2
            other, _ = admitted_terminal_receipt(result, 2)
            self.assertNotEqual(receipt, other)

    def test_unexplained_counter_and_missing_frame_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, result = self.fixture(root)
            result['first_life']['aim_by_class'][0]['aim_frames'] = 4
            with self.assertRaises(AssertionError):
                admitted_terminal_receipt(result, 2)
            result['first_life']['aim_by_class'][0]['aim_frames'] = 3
            (root/'bot.jsonl').write_text('\n'.join(json.dumps(r) for r in rows[1:]), encoding='utf-8')
            with self.assertRaises(AssertionError):
                admitted_terminal_receipt(result, 2)


if __name__ == '__main__':
    unittest.main()
