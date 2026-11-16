import unittest
from prepare_combat_hjepa_effects import effect_vectors, effect_action, WEIGHTS
from test_hjepa_weapon_returns import ReturnLedger


class NativeEffectTargets(unittest.TestCase):
    def setUp(self):
        self.fixture=ReturnLedger()
        self.result,self.steps,self.outcomes=self.fixture.fixture()

    def test_terminal_win_has_unique_transition(self):
        vectors,ledger=effect_vectors(self.result,self.steps,self.outcomes)
        self.assertEqual([i for i,v in vectors.items() if v[4]], [2])
        self.assertEqual(sum(sum(a*b for a,b in zip(v,WEIGHTS)) for v in vectors.values()),1.)

    def test_self_damage_retains_received_and_self_penalties(self):
        self.outcomes[0]['events']=[self.fixture.event(1,100,90)]
        self.result['first_life']['damage'].update(received_health_damage=10,self_health_damage=10)
        vectors,_=effect_vectors(self.result,self.steps,self.outcomes)
        self.assertEqual(vectors[0][1:3],[.1,.1])
        self.assertAlmostEqual(sum(a*b for a,b in zip(vectors[0],WEIGHTS)),-.11)

    def test_monster_overkill_uses_actual_health_loss(self):
        event=self.fixture.event(1,10,-40,target=2,kind='monster_soldier')
        self.outcomes[0]['events']=[event]
        self.result['first_life']['outgoing_by_target_class']=[dict(health_damage=10)]
        vectors,_=effect_vectors(self.result,self.steps,self.outcomes)
        self.assertAlmostEqual(vectors[0][0],.1)

    def test_post_endpoint_event_has_no_effect_target(self):
        self.outcomes[3]['events']=[self.fixture.event(5,100,99,attacker=2)]
        vectors,ledger=effect_vectors(self.result,self.steps,self.outcomes)
        self.assertEqual(vectors[3],[0.]*6)
        self.assertEqual(ledger['events_outside_first_life'],1)

    def test_death_does_not_receive_win_label(self):
        self.result['goal_stop']=None
        self.result['death_stop']=dict(observed_frame=4)
        vectors,_=effect_vectors(self.result,self.steps,self.outcomes)
        self.assertEqual(vectors[2][4:],[0.,1.])

    def test_inconsistent_native_counts_reject_labels(self):
        self.result['first_life']['damage']['received_health_damage']=1
        with self.assertRaisesRegex(ValueError,'balance mismatch'):
            effect_vectors(self.result,self.steps,self.outcomes)

    def test_full_action_distinguishes_weapon_request(self):
        action=dict(forward=0.,side=0.,yaw_delta_degrees=0.,pitch_delta_degrees=0.,attack=True,vertical='release')
        keep=effect_action(action)
        rail=effect_action(dict(action,weapon='Railgun'))
        self.assertEqual(len(keep),20)
        self.assertEqual(keep[:8],rail[:8])
        self.assertNotEqual(keep[8:],rail[8:])
        self.assertEqual(sum(rail[8:]),1.)


if __name__=='__main__':unittest.main()
