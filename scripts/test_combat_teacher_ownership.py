import copy
import unittest

from combat_teacher_ownership import passive_goal_stop, rules_ownership


class OwnershipTests(unittest.TestCase):
    def fixture(self):
        identity=dict(frame=147,map='city2',spawncount=4,actor=1,life=1)
        step=dict(owner='rules',terminal=True,truncated=False,reason='combat_goal_complete',
            observation=dict(identity=identity),next_observation=dict(identity=dict(identity,frame=148),health=85),
            native_step=dict(spawncount=4,begin_frame=147,end_frame=148,actor=1,damage_indexes=[]),
            server_execution=dict(matched=True,window_exclusive=True,dispatch_frame=147,recovery_commands=0),
            sent_command=dict(Forward=0,Side=0,Up=0,Buttons=0,Impulse=0))
        goal=dict(version='combat_goal_stop_v2',reason='combat_goal_complete',health=85,kill_frame=147,
            observed_frame=148,spawncount=4,actor=1,
            native_objective=dict(frame=148,map='city2',spawncount=4,alive=0,pending=0))
        return step,goal

    def test_only_proven_quiet_terminal_receipt_is_passive(self):
        step,goal=self.fixture()
        self.assertTrue(passive_goal_stop(step,goal))
        ownership=rules_ownership([step],goal,148)
        self.assertEqual(ownership,dict(rules_combat_rows=0,passive_goal_stop_frames=[147],original_inclusive_rules_rows=1))
        for mutate in (
            lambda s,g:s['sent_command'].update(Buttons=1),
            lambda s,g:s['sent_command'].update(Forward=1),
            lambda s,g:s['sent_command'].update(Impulse=1),
            lambda s,g:s.update(reason='control_handoff'),
            lambda s,g:s['native_step'].update(damage_indexes=[1]),
            lambda s,g:s['server_execution'].update(matched=False),
            lambda s,g:g['native_objective'].update(alive=1),
            lambda s,g:g['native_objective'].update(frame=149),
            lambda s,g:s['next_observation']['identity'].update(frame=149),
        ):
            s,g=copy.deepcopy(step),copy.deepcopy(goal);mutate(s,g)
            self.assertFalse(passive_goal_stop(s,g))
            self.assertEqual(rules_ownership([s],g,148)['rules_combat_rows'],1)


if __name__=='__main__':
    unittest.main()
