import unittest

from combat_teacher_folds import map_folds, world_weights


class TeacherFoldTests(unittest.TestCase):
    def test_all_frames_and_seeds_of_map_stay_together(self):
        rows = [dict(map=m, seed=s, source_model_sha256='a'*64)
                for m, s in [('a',1), ('b',2), ('c',3), ('a',1), ('d',4), ('a',5)]]
        folds = map_folds(rows)
        self.assertEqual(folds[0], folds[3])
        self.assertEqual(folds[0], folds[5])
        self.assertEqual(set(folds), {0,1,2,3})
        self.assertEqual(map_folds(list(reversed(rows))), list(reversed(folds)))
        self.assertEqual(world_weights(rows), [.5,1.,1.,.5,1.,1.])


if __name__ == '__main__':
    unittest.main()
