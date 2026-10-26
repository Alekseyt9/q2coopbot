import unittest

from combat_teacher_root_selection import choose_root_frame


class RootSelectionTests(unittest.TestCase):
    def test_new_and_temporally_separated_contexts(self):
        self.assertEqual(choose_root_frame([153, 115, 115, 140]), 115)
        self.assertEqual(choose_root_frame([115, 140, 144, 145, 153], [115]), 145)
        self.assertEqual(choose_root_frame([115, 145, 175, 180], [145, 115]), 175)
        self.assertIsNone(choose_root_frame([115, 140], [115]))
        self.assertIsNone(choose_root_frame([]))

    def test_invalid_identity_or_gap_rejected(self):
        for frames, used, gap in (([109], [], 30), ([True], [], 30),
                                  ([110], [0], 30), ([110], [], 0)):
            with self.assertRaises(AssertionError):
                choose_root_frame(frames, used, gap)


if __name__ == '__main__':
    unittest.main()
