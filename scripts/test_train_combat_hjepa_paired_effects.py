import unittest
from train_combat_hjepa_paired_effects import pair_indices


class ControlledPairAdmission(unittest.TestCase):
    def row(self,seed,variant,horizon=1,features=None):
        return dict(seed=seed,variant=variant,horizon=horizon,features=[float(seed)] if features is None else features)

    def test_reference_stays_in_same_world(self):
        rows=[self.row(1,'idle'),self.row(2,'fire'),self.row(1,'fire'),self.row(2,'idle'),self.row(2,'left-fire')]
        left,right=pair_indices(rows)
        self.assertEqual(len(left),3)
        for i,j in zip(left,right):self.assertEqual(rows[i]['seed'],rows[j]['seed'])

    def test_missing_reference_is_not_borrowed_from_another_world(self):
        rows=[self.row(1,'idle'),self.row(1,'fire'),self.row(2,'left-fire')]
        self.assertEqual(pair_indices(rows),([1],[0]))

    def test_changed_initial_observation_rejects_pair(self):
        with self.assertRaisesRegex(ValueError,'initial state'):
            pair_indices([self.row(1,'idle'),self.row(1,'fire',features=[2.])])

    def test_horizons_cannot_be_mixed(self):
        with self.assertRaisesRegex(ValueError,'horizon'):
            pair_indices([self.row(1,'idle'),self.row(1,'fire',horizon=4)])

    def test_duplicate_arm_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            pair_indices([self.row(1,'idle'),self.row(1,'fire'),self.row(1,'fire')])

    def test_empty_contrast_set_rejected(self):
        with self.assertRaisesRegex(ValueError,'no controlled pairs'):
            pair_indices([self.row(1,'idle')])


if __name__=='__main__':unittest.main()
