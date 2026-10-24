import unittest

from audit_combat_movement_probe import movement_latent, same_aim_timing


class MovementProbeAcceptanceTests(unittest.TestCase):
    def test_gaussian_when_aim_head_inactive(self):
        sample = dict(latent=[.1, -.2, .3, .4])
        self.assertEqual(movement_latent(sample), [.1, -.2])
        self.assertTrue(same_aim_timing(sample, dict(sample)))

    def test_active_head_uses_original_gaussian(self):
        sample = dict(latent=[.1, -.2, 7, 8], aim_candidate=dict(timing=0, source_latent=[.1, -.2, .3, .4]))
        self.assertEqual(movement_latent(sample), [.1, -.2])
        self.assertTrue(same_aim_timing(sample, sample))
        self.assertFalse(same_aim_timing(sample, dict(latent=[.1, -.2, .3, .4])))
        self.assertFalse(same_aim_timing(sample, dict(aim_candidate=dict(timing=1))))


if __name__ == '__main__':
    unittest.main()
