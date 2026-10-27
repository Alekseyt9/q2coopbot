"""Metadata-only admission checks, including the Quake HUD health floor."""
import copy
import tempfile
import unittest
from pathlib import Path

from recover_combat_wide_future_replicates import compare_terminal_pair


class TerminalPairTests(unittest.TestCase):
    def fixture(self,root):
        from test_combat_teacher_end_receipt import NativeEndReceiptTests
        return NativeEndReceiptTests().fixture(root)

    def test_exact_present_client_tags_do_not_require_native_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result=self.fixture(root)
            result['first_life']['minimum_health']=-99
            result['death_stop']=dict(version='combat_first_life_death_stop_v1',reason='combat_first_life_death',
                map='base1',spawncount=4,actor=1,death_frame=3,observed_frame=3,health=-99)
            # No native reconstruction is allowed or needed for an exact pair.
            left,right,difference=compare_terminal_pair(result,copy.deepcopy(result),2,root/'absent',root/'absent')
            self.assertIsNone(difference)
            self.assertNotIn('native_death',left)
            other=copy.deepcopy(result);other['death_stop']['health']=-98
            with self.assertRaises(AssertionError):compare_terminal_pair(result,other,2,root,root)

    def test_only_missing_tag_can_use_exact_native_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);missing=self.fixture(root);present=copy.deepcopy(missing)
            present['death_stop']=dict(version='combat_first_life_death_stop_v1',reason='combat_first_life_death',
                map='base1',spawncount=4,actor=1,death_frame=3,observed_frame=3,health=0)
            left,right,difference=compare_terminal_pair(missing,present,2,root,root)
            self.assertIsNotNone(difference)
            self.assertEqual(left['native_death']['canonical'],right['native_death']['canonical'])
            bad=copy.deepcopy(present);bad['first_life']['minimum_health']=-1
            with self.assertRaises(AssertionError):compare_terminal_pair(missing,bad,2,root,root)
            bad=copy.deepcopy(present);bad['death_stop']['actor']=2
            with self.assertRaises(AssertionError):compare_terminal_pair(missing,bad,2,root,root)


if __name__=='__main__':unittest.main()
