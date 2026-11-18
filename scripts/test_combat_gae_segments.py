"""Numerical boundaries: GAE cuts retain bootstrap and full observation context."""
import copy
import unittest

from combat_gae_segments import segment_rows
from ppo_combat import advantages


def row(index, **changes):
    result = dict(seed=1, index=index, frame=100+index, next_frame=101+index,
                  reward=1.+index, sample=dict(value=10.*(index+1)),
                  next_value=10.*(index+2), terminal=False, truncated=False,
                  bootstrap_zero=False)
    result.update(changes)
    return result


class ShortCreditSegments(unittest.TestCase):
    def test_hand_calculated_returns_and_nonzero_bootstrap(self):
        rows = [row(i) for i in range(3)]
        view, proof = segment_rows(rows, 2)
        # TD deltas [1,-3,-7]. Cut after delta -3, retaining V(next)=30.
        self.assertEqual(advantages(view, .5, 1.), ([-.5, -3., -7.], [9.5, 17., 23.]))
        self.assertEqual(advantages(rows, .5, 1.)[0], [-2.25, -6.5, -7.])
        self.assertEqual(proof['added_cuts'], 1)
        self.assertEqual(proof['segments'][0]['bootstrap_value'], 30.)

    def test_reward_after_cut_cannot_leak_into_previous_segment(self):
        rows = [row(i) for i in range(3)]
        before = advantages(segment_rows(rows, 2)[0], .5, 1.)
        rows[-1]['reward'] += 1000
        after = advantages(segment_rows(rows, 2)[0], .5, 1.)
        self.assertEqual(before[0][:2], after[0][:2])
        self.assertNotEqual(before[0][-1], after[0][-1])

    def test_native_terminal_and_truncation_preserved(self):
        rows = [row(0, terminal=True, bootstrap_zero=True, next_value=999.),
                row(1, truncated=True, next_value=8.), row(2)]
        view, proof = segment_rows(rows, 1)
        self.assertEqual([s['reason'] for s in proof['segments']],
                         ['native_terminal', 'native_truncation', 'dataset_end'])
        self.assertEqual(proof['added_cuts'], 0)
        adv, _ = advantages(view, .5, 1.)
        self.assertEqual(adv[0], -9.)
        self.assertEqual(adv[1], -14.)

    def test_seed_and_context_gaps_do_not_join_segments(self):
        rows = [row(0), row(2), row(0, seed=2)]
        _, proof = segment_rows(rows, 64)
        self.assertEqual([s['rows'] for s in proof['segments']], [1, 1, 1])
        self.assertEqual(advantages(rows, .5, 1.)[0], [1., -7., 1.])

    def test_raw_rows_samples_and_indices_are_not_rewritten(self):
        rows = [row(i) for i in range(5)]
        before = copy.deepcopy(rows)
        view, proof = segment_rows(rows, 2)
        self.assertEqual(rows, before)
        self.assertEqual([s['rows'] for s in proof['segments']], [2, 2, 1])
        for source, credit in zip(rows, view):
            self.assertIs(credit['sample'], source['sample'])
            self.assertEqual({k:v for k,v in source.items() if k!='truncated'},
                             {k:v for k,v in credit.items() if k!='truncated'})
        self.assertFalse(proof['context_reset'])

    def test_estimated_zero_value_allowed_but_unexplained_zero_flag_rejected(self):
        rows = [row(0, next_value=0.), row(1)]
        self.assertEqual(segment_rows(rows, 1)[1]['segments'][0]['bootstrap_value'], 0.)
        rows[0]['bootstrap_zero'] = True
        with self.assertRaises(AssertionError):segment_rows(rows, 1)

    def test_noop_horizon_matches_original_advantages_exactly(self):
        rows = [row(i) for i in range(3)]
        view, proof = segment_rows(rows, 64)
        self.assertEqual(advantages(view, .99, .95), advantages(rows, .99, .95))
        self.assertEqual(proof['added_cuts'], 0)

    def test_invalid_boundaries_fail_before_training(self):
        for rows, limit in [([], 64), ([row(0)], 0), ([row(0)], True),
                            ([row(0), row(0)], 2),
                            ([row(0, next_frame=100)], 2),
                            ([row(0, next_value=float('nan'))], 2)]:
            with self.subTest(limit=limit, rows=rows):
                with self.assertRaises(AssertionError):segment_rows(rows, limit)


if __name__ == '__main__':
    unittest.main()
