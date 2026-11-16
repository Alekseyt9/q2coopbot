import unittest
from prepare_combat_hjepa_branch_preferences import outcome_score, preference, outcome_improvement
from prepare_combat_hjepa_weapon_branches import choices


def result(win=False, death=False, received=0, self_hp=0, dealt=0):
    return dict(goal_stop=win, death_stop=death, generated_start=dict(entities=[dict(health=100, **{'class': 'monster_infantry'})]),
                first_life=dict(damage=dict(received_health_damage=received, self_health_damage=self_hp),
                                outgoing_by_target_class=[dict(health_damage=dealt)]))


class BranchPreferenceTests(unittest.TestCase):
    def test_improvement_requires_outcome_change(self):
        same = {0: dict(win=False, death=False), 4: dict(win=False, death=False)}
        self.assertEqual(outcome_improvement({0: 0, 4: 10}, same)[0], 1)
        same[4]['win'] = True
        self.assertEqual(outcome_improvement({0: 0, 4: 100}, same)[4], 1)
        same[0]['death'] = True
        same[4]['win'] = False
        self.assertEqual(outcome_improvement({0: -100, 4: 0}, same)[4], 1)
    def test_win_dominates_progress_and_survival(self):
        self.assertGreater(outcome_score(result(win=True, received=100, self_hp=100)), outcome_score(result(dealt=100)))
        self.assertGreater(outcome_score(result()), outcome_score(result(death=True, dealt=100)))
        self.assertLess(outcome_score(result(win=True, death=True)), outcome_score(result()))

    def test_mask_and_change_cost(self):
        y = preference({0: 0., 4: 1.}, [i in [0, 4] for i in range(12)])
        self.assertGreater(y[0], y[4])
        self.assertEqual(y[9], 0)
        self.assertAlmostEqual(sum(y), 1.)

    def test_unmeasured_available_action_rejected(self):
        with self.assertRaises(ValueError):
            preference({0: 0}, [i in [0, 4] for i in range(12)])

    def test_ammo_thresholds_and_incumbent(self):
        i = dict(inventory=dict(initial_weapon='Blaster', items=[dict(name='Blaster', count=1),
            dict(name='Super Shotgun', count=1), dict(name='Shells', count=1), dict(name='BFG10K', count=1), dict(name='Cells', count=49)]))
        self.assertEqual(choices(i), [])
        i['inventory']['items'][2]['count'] = 2
        i['inventory']['items'][4]['count'] = 50
        self.assertEqual(choices(i), [3, 10])


if __name__ == '__main__':
    unittest.main()
