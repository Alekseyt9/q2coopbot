import unittest
from prepare_combat_hjepa_joint_world import clip_windows


class WorldClipAdmission(unittest.TestCase):
    def rows(self,n=16,episode=('sha','life1'),terminal=False):
        return [dict(episode_key=episode,frame=i+10,features=[i],next_features=[i+1],tensor_index=i,terminal=terminal and i==n-1) for i in range(n)]

    def test_contiguous_full_window(self):
        clips,lengths=clip_windows(self.rows())
        self.assertEqual(clips,[list(range(16))]);self.assertEqual(lengths,[16])

    def test_terminal_padding_keeps_true_length_for_effect_mask(self):
        clips,lengths=clip_windows(self.rows(3,terminal=True))
        self.assertEqual(lengths,[3,2,1]);self.assertEqual(clips[0],[0,1,2]+[2]*13)

    def test_nonterminal_short_window_not_padded(self):
        self.assertEqual(clip_windows(self.rows(3)),([],[]))

    def test_source_and_life_cannot_be_concatenated(self):
        a=self.rows(8);b=self.rows(8,episode=('other','life1'))
        self.assertEqual(clip_windows(a+b),([],[]))

    def test_frame_gap_rejected(self):
        rows=self.rows()
        for row in rows[8:]:row['frame']+=1
        self.assertEqual(clip_windows(rows),([],[]))

    def test_duplicate_source_frame_rejected(self):
        rows=self.rows();rows[8]['frame']=rows[7]['frame']
        with self.assertRaisesRegex(ValueError,'duplicate'):clip_windows(rows)

    def test_inconsistent_next_state_breaks_window(self):
        rows=self.rows();rows[8]['features']=[-100]
        self.assertEqual(clip_windows(rows),([],[]))


if __name__=='__main__':unittest.main()
