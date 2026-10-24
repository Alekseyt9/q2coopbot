import unittest

from audit_combat_grenade_preimpact import context_at, describe


class GrenadePreimpactTests(unittest.TestCase):
    def fixture(self):
        identity = dict(map='base3', spawncount=7, connection=1, life=1, actor=1, frame=100)
        step = dict(index=4, observation=dict(identity=identity, velocity=[0, 0, 0], health=80,
            on_ground=True, visible_projectiles=[], enemies=[]), action=dict(forward=1, side=0),
            applied_action=dict(forward=0, side=0), interventions=[])
        return step, dict(frame=103, game_frame=105, inflictor=9, attacker=8)

    def test_network_clock_and_missing_frame(self):
        step, event = self.fixture()
        self.assertIs(context_at({100: step}, event, 3, step['observation']['identity'])[0], step)
        self.assertEqual(context_at({100: step}, event, 1, step['observation']['identity']), (None, 'missing_frame'))

    def test_connection_and_life_boundaries(self):
        step, event = self.fixture()
        for key in ('life', 'connection', 'spawncount', 'actor'):
            reference = dict(step['observation']['identity'])
            reference[key] += 1
            self.assertEqual(context_at({100: step}, event, 3, reference), (None, 'different_identity'))

    def test_visible_but_filtered_candidate_and_unknown_velocity(self):
        step, event = self.fixture()
        step['observation']['visible_projectiles'] = [dict(id=i, distance=i, **{'class': 'blaster_bolt'}) for i in range(1, 5)]
        step['observation']['visible_projectiles'].append(dict(id=9, distance=10, **{'class': 'grenade'}, observed_velocity=None))
        result = describe(step, event)
        self.assertTrue(result['same_id_candidate_visible'])
        self.assertFalse(result['same_id_candidate_in_slots'])
        self.assertFalse(result['same_id_candidate_velocity_known'])
        self.assertFalse(result['lifetime_identity_proven'])

    def test_stable_equal_distance_order_and_zero_velocity(self):
        step, event = self.fixture()
        step['observation']['visible_projectiles'] = [dict(id=9, distance=10, **{'class': 'grenade'}, observed_velocity=[0, 0, 0])]
        step['observation']['visible_projectiles'] += [dict(id=i, distance=10, **{'class': 'blaster_bolt'}) for i in range(1, 5)]
        result = describe(step, event)
        self.assertEqual(result['same_id_candidate_slot'], 0)
        self.assertTrue(result['same_id_candidate_velocity_known'])
        self.assertTrue(result['request_reduced_near_zero'])


if __name__ == '__main__':
    unittest.main()
