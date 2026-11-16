import copy
import unittest
from hjepa_weapon_returns import episode_ledger,at_frame


class ReturnLedger(unittest.TestCase):
    def fixture(self):
        result=dict(first_life=dict(bot_entity=1,map='base1',spawncount=7,start_frame=1,end_frame=4,
            damage=dict(received_health_damage=0,self_health_damage=0,teammate_health_damage=0),
            outgoing_by_target_class=[]),generated_start=dict(entities=[{'class':'monster_soldier','health':100}]),
            goal_stop=dict(observed_frame=4),death_stop=None)
        steps=[dict(index=i,observation=dict(identity=dict(frame=i+1)),
            next_observation=dict(identity=dict(frame=i+2)),native_step=dict(begin_frame=i+1,end_frame=i+2,actor=1),
            server_execution=dict(matched=True)) for i in range(5)]
        outcomes=[dict(step=i,available=True,events=[],version='server_step_effects_v1') for i in range(5)]
        return result,steps,outcomes

    def event(self,frame,before,after,attacker=1,target=1,kind='player'):
        return dict(map='base1',spawncount=7,frame=frame,health_before=before,health_after=after,
                    take=before-after,attacker=attacker,target=target,target_class=kind)

    def test_past_self_damage_not_blame_for_future_choice(self):
        r,s,o=self.fixture();o[0]['events']=[self.event(1,100,90)]
        r['first_life']['damage'].update(received_health_damage=10,self_health_damage=10)
        ledger=episode_ledger(r,s,o)
        self.assertAlmostEqual(ledger['scalar_return'],89)
        self.assertAlmostEqual(ledger['suffix'][1],100)
        self.assertEqual(ledger['terminal_step'],2)

    def test_goal_bonus_is_not_given_to_post_ack_choice(self):
        r,s,o=self.fixture();o[3]['events']=[self.event(4,100,99,attacker=2)]
        r['first_life']['damage']['received_health_damage']=1
        ledger=episode_ledger(r,s,o)
        self.assertAlmostEqual(ledger['suffix'][3],-.1)
        self.assertEqual(at_frame(ledger,5),0)
        self.assertAlmostEqual(ledger['scalar_return'],99.9)

    def test_post_endpoint_event_excluded_without_changing_score(self):
        r,s,o=self.fixture();o[3]['events']=[self.event(5,100,99,attacker=2)]
        ledger=episode_ledger(r,s,o)
        self.assertEqual(ledger['events_outside_first_life'],1)
        self.assertEqual(ledger['scalar_return'],100)

    def test_missing_effect_cannot_be_silently_spread_over_steps(self):
        r,s,o=self.fixture();r['first_life']['damage']['received_health_damage']=1
        with self.assertRaisesRegex(ValueError,'balance mismatch'):episode_ledger(r,s,o)

    def test_native_step_phase_cannot_be_guessed_from_frame(self):
        r,s,o=self.fixture();o[0]['events']=[self.event(4,100,99,attacker=2)]
        with self.assertRaisesRegex(ValueError,'causal native'):episode_ledger(r,s,o)

    def test_terminal_needs_unique_observed_transition(self):
        r,s,o=self.fixture();r['goal_stop']['observed_frame']=99
        with self.assertRaisesRegex(ValueError,'terminal observation'):episode_ledger(r,s,o)

    def test_other_stream_finished_before_current_choice_has_no_future_bonus(self):
        r,s,o=self.fixture();r['goal_stop']['observed_frame']=2;r['first_life']['end_frame']=2
        ledger=episode_ledger(r,s,o)
        self.assertEqual(ledger['scalar_return'],100)
        self.assertEqual(at_frame(ledger,3),0)

    def test_death_receipt_overrides_simultaneous_goal(self):
        r,s,o=self.fixture();r['death_stop']=dict(observed_frame=4)
        ledger=episode_ledger(r,s,o)
        self.assertEqual(ledger['scalar_return'],-100)
        self.assertEqual(ledger['reward'][2],-100)

    def test_monster_progress_cap_and_overkill(self):
        r,s,o=self.fixture();r['goal_stop']=None
        o[0]['events']=[self.event(1,100,-50,target=2,kind='monster_soldier')]
        o[1]['events']=[self.event(2,100,0,target=3,kind='monster_soldier')]
        r['first_life']['outgoing_by_target_class']=[dict(health_damage=200)]
        ledger=episode_ledger(r,s,o)
        self.assertEqual(ledger['scalar_return'],10)
        self.assertEqual(ledger['suffix'][1],0)


if __name__=='__main__':unittest.main()
