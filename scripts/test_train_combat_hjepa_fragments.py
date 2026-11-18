"""Fragment admission must preserve TRAIN, native-tail and source bindings."""
import json
import pathlib
import tempfile
import unittest

import torch
from train_combat_hjepa_fragments import load, sha, family_weights, validate_fire_roots, distribution_without_sampling_seeds, enumerated_fire_roots, validate_root_joint_records


class FragmentAdmission(unittest.TestCase):
    def test_joint_root_iid_leave_one_out_matches_expected_score_gradient(self):
        # Integrate all independent two-draw Bernoulli batches, including equal
        # actions. Action-stratified enumeration would be a different estimator.
        mean = torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        objective = mean*0
        for first in [False,True]:
            for second in [False,True]:
                attack=torch.tensor([first,second])
                returns=torch.tensor([float(first),float(second)],dtype=torch.float64)
                advantage=returns-returns.flip(0)
                lp=torch.nn.functional.logsigmoid(torch.where(attack,mean,-mean))
                objective=objective+.25*((lp-lp.detach()).exp()*advantage).mean()
        objective.backward()
        self.assertAlmostEqual(mean.grad.item(),.25)

    def test_joint_root_admission_requires_prescribed_complete_common_root(self):
        import copy
        row=dict(index=0,features=[0.]*871,actor_raw=[0.]*8,logical_sampling_identity=dict(frame=101))
        records=[('f',1,copy.deepcopy(row)),('f',2,copy.deepcopy(row))]
        validate_root_joint_records(records,2)
        for bad in [records[:1],[records[0],('f',3,row)]]:
            with self.assertRaisesRegex(ValueError,'complete predeclared'):validate_root_joint_records(bad,2)
        with self.assertRaisesRegex(ValueError,'complete predeclared'):validate_root_joint_records(records,4)
        changed=copy.deepcopy(records);changed[1][2]['features'][3]=1.
        with self.assertRaisesRegex(ValueError,'common inputs'):validate_root_joint_records(changed,2)
        changed=copy.deepcopy(records);changed[1][2]['index']=1
        with self.assertRaisesRegex(ValueError,'first decision'):validate_root_joint_records(changed,2)

    def test_joint_root_format_and_peer_credit_are_distinct_from_fire_enumeration(self):
        for path in self.paths:
            t=json.loads(path.read_text());fpath=pathlib.Path(t['prefix']);f=json.loads(fpath.read_text())
            f['version']='hjepa_native_fragment_v5'
            f['sampling_checkpoint'].update(version='hjepa_sampling_checkpoint_v5',sampling_scope='root_joint_only_v1')
            f['transitions'][0]['logical_sampling_identity']=dict(frame=110)
            fpath.write_text(json.dumps(f));t.update(version='hjepa_fragment_mc_targets_v4',sampling_scope='root_joint_only_v1')
            t['sources_sha256'][str(fpath)]=sha(fpath);path.write_text(json.dumps(t))
        _,packed,_,info=load(self.paths)
        self.assertEqual([a for r,a in packed],[.04,-.04])
        validate_root_joint_records([(r['checkpoint_family'],r['sampling_stream'],r) for r,a in packed],2)
        self.assertEqual(info['sampling_scope'],'root_joint_only_v1')

    def test_two_action_mass_matches_exact_bernoulli_expected_return_gradient(self):
        import math
        def row(attack):
            return dict(index=0,logical_sampling_identity=dict(frame=101),features=[0.]*871,
                        actor_raw=[0.]*4+[2*math.log(4)]+[0.]*3,sample=dict(attack=attack,latent=[0.]*4,vertical=0,weapon=0))
        records=[('f',1,row(True)),('f',2,row(False))]
        packed=enumerated_fire_roots(records,{'f':{1:{0:100.},2:{0:0.}}},2.)
        raw=torch.tensor(2*math.log(4),dtype=torch.float64,requires_grad=True)
        attacks=torch.tensor([r['sample']['attack'] for r,a in packed])
        lp=torch.nn.functional.logsigmoid(torch.where(attacks,raw,-raw)/2)
        masses=torch.tensor([r['enumeration_mass'] for r,a in packed],dtype=torch.float64)
        advantage=torch.tensor([a for r,a in packed],dtype=torch.float64)
        objective=(2*masses*(lp-lp.detach()).exp()*advantage).mean()
        objective.backward()
        self.assertAlmostEqual(objective.item(),0.)
        self.assertAlmostEqual(raw.grad.item(),.8*.2/2)
        self.assertAlmostEqual(sum(masses).item(),1.)
        bad=[records[0]]
        with self.assertRaisesRegex(ValueError,'exactly two complete opposite-action'):
            enumerated_fire_roots(bad,{'f':{1:{0:100.}}},2.)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.model = self.write('model.json', dict(weapon_mode='learned_masked_v1',
            weapon_head='combat_masked_weapon_v1', weapon_sampling_seed=7, primitive_sampling=dict(seed=7), planner='bc'))
        self.plan = self.write('plan.json', dict(split='train'))
        self.source = self.root/'source'
        self.source.mkdir()
        self.source_report = self.write('source/report.json', dict(
            generated_fixture=dict(split='train', map='base1', episode_id='episode', engine_seed=1729),
            generated_start=dict(confirmed=True)))
        self.paths = [self.pair(stream, value) for stream, value in [(1, 104.), (2, 100.)]]

    def write(self, name, value):
        p = self.root/name
        p.write_text(json.dumps(value), encoding='utf-8')
        return p

    def pair(self, stream, value):
        fragment = dict(version='hjepa_native_fragment_v2', split='train',
            base_sampling_seed=7,
            training_source_verified=True, training_plan=str(self.plan),
            behavior_model=str(self.model), model_version='hjepa:'+sha(self.model),
            source_directory=str(self.source),
            sources_sha256={str(p): sha(p) for p in [self.model, self.plan, self.source_report]},
            checkpoint_family='same-native-root', sampling_checkpoint=dict(version='hjepa_sampling_checkpoint_v2',model='hjepa:'+sha(self.model),sampling_stream=stream, source_identity=dict(map='base1', frame=110)),
            transitions=[dict(version='hjepa_native_fragment_transition_v2',
                index=0, features=[0.]*871, actor_raw=[0.]*8,sample=dict(sampling_seed=7))])
        prefix = self.write(f'prefix-{stream}.json', fragment)
        tail = self.write(f'tail-{stream}.json', dict(native_terminal=True))
        target = dict(version='hjepa_fragment_mc_targets_v1', optimizer_ready=True,
            bootstrap_proof='native_terminal_in_same_stream_continuation', bootstrap_value=0,
            terminal_reason='native_source_roster_clear', terminal_step=2,
            prefix=str(prefix), tail=str(tail), model_version=fragment['model_version'],
            sources_sha256={str(p): sha(p) for p in [self.model,self.plan,self.source_report,prefix,tail]},
            sampling_stream=stream, checkpoint_family=fragment['checkpoint_family'],
            targets=[dict(index=0, return_to_go=value)])
        return self.write(f'target-{stream}.json', target)

    def seeded_family(self, seed, *, same_condition=False, mutate_model=None):
        model = json.loads(self.model.read_text())
        model['primitive_sampling']['seed'] = model['weapon_sampling_seed'] = seed
        if mutate_model:mutate_model(model)
        model_path = self.write(f'model-{seed}.json',model)
        source = self.root/f'source-{seed}';source.mkdir()
        report = json.loads(self.source_report.read_text())
        if not same_condition:report['generated_fixture']['engine_seed'] += seed
        source_report = self.write(f'source-{seed}/report.json',report)
        paths = []
        for index,path in enumerate(self.paths):
            target = json.loads(path.read_text());fragment = json.loads(pathlib.Path(target['prefix']).read_text())
            fragment.update(behavior_model=str(model_path),model_version='hjepa:'+sha(model_path),base_sampling_seed=seed,
                            source_directory=str(source),checkpoint_family=f'family-{seed}')
            fragment['sampling_checkpoint']['model'] = fragment['model_version']
            fragment['transitions'][0]['sample']['sampling_seed'] = seed
            fragment['sources_sha256'] = {str(f):sha(f) for f in [model_path,self.plan,source_report]}
            prefix = self.write(f'seeded-prefix-{seed}-{index}.json',fragment)
            target.update(prefix=str(prefix),model_version=fragment['model_version'],checkpoint_family=fragment['checkpoint_family'])
            target['sources_sha256'] = dict(fragment['sources_sha256'],**{str(prefix):sha(prefix),target['tail']:sha(pathlib.Path(target['tail']))})
            paths.append(self.write(f'seeded-target-{seed}-{index}.json',target))
        return paths

    def test_sampling_seed_equivalence_is_opt_in_and_keeps_distribution_fields(self):
        extra = self.seeded_family(19)
        with self.assertRaisesRegex(ValueError,'different behavior weights'):load([*self.paths,*extra])
        parent,rows,inputs,admission = load([*self.paths,*extra],True)
        self.assertEqual(admission['rows'],4)
        self.assertEqual(admission['training_worlds'],2)
        self.assertEqual(len(admission['behavior_model_versions']),2)
        self.assertEqual(distribution_without_sampling_seeds(parent),distribution_without_sampling_seeds(json.loads(self.model.read_text())))
        bad = self.seeded_family(23,mutate_model=lambda m:m.update(actor=[dict(weight=[[.5]],bias=[0.])]))
        with self.assertRaisesRegex(ValueError,'different behavior weights'):load([*self.paths,*bad],True)

    def test_equivalent_seed_models_still_require_actual_seed_receipts(self):
        extra = self.seeded_family(19)
        target = json.loads(extra[0].read_text());prefix = pathlib.Path(target['prefix']);fragment = json.loads(prefix.read_text())
        fragment['transitions'][0]['sample']['sampling_seed'] = 7
        self.write(prefix.name,fragment);target['sources_sha256'][str(prefix)] = sha(prefix);self.write(extra[0].name,target)
        with self.assertRaisesRegex(ValueError,'behavior sampling seed binding'):load([*self.paths,*extra],True)

    def test_reserialized_seed_equivalent_model_does_not_hide_duplicate_world_root(self):
        extra = self.seeded_family(7,same_condition=True)
        with self.assertRaisesRegex(ValueError,'same TRAIN condition/root'):load([*self.paths,*extra],True)

    def change_target(self, mutate):
        p = self.paths[0]
        obj = json.loads(p.read_text())
        mutate(obj)
        self.write(p.name, obj)

    def test_paired_returns_use_other_stream_without_counting_rows_as_worlds(self):
        _, packed, _, info = load(self.paths)
        self.assertEqual([adv for _,adv in packed], [.04, -.04])
        self.assertEqual(info['checkpoint_families'], 1)
        self.assertEqual(info['rows'], 2)

    def test_family_balanced_gradient_does_not_overweight_more_rows_in_one_family(self):
        def gradient(values):
            rows = [dict(checkpoint_family=family) for family,value in values]
            parameter = torch.tensor(1., dtype=torch.float64, requires_grad=True)
            score = torch.tensor([value for family,value in values], dtype=torch.float64)*parameter
            (score*torch.tensor(family_weights(rows), dtype=torch.float64)).mean().backward()
            return parameter.grad.item()
        base = [('a', 1.), ('a', 2.), ('b', 4.)]
        duplicated = [('a', 1.), ('a', 2.)]*4+[('b', 4.)]
        self.assertAlmostEqual(gradient(base), 2.75)
        self.assertAlmostEqual(gradient(duplicated), gradient(base))

    def test_observed_phase_mass_is_independent_of_action_outcome_and_population(self):
        def row(family,held):
            features = [0.]*871
            if held:features[856],features[5] = 1.,11/30
            return dict(checkpoint_family=family,features=features,sample=dict(attack=True))
        rows = [row('a',True)]+[row('a',False) for _ in range(9)]+[row('b',False)]
        weights = family_weights(rows,True)
        n = len(rows)
        self.assertAlmostEqual(weights[0]/n,.25)
        self.assertAlmostEqual(sum(weights[1:10])/n,.25)
        self.assertAlmostEqual(weights[-1]/n,.5)
        for r in rows:r['sample']['attack']=False;r['return_to_go']=999
        self.assertEqual(family_weights(rows,True),weights)
        duplicated = rows[:1]+rows[1:10]*10+rows[-1:]
        other_weights = family_weights(duplicated,True)
        score = lambda rs,ws:sum(w*(10 if r['features'][856] else 4 if r['checkpoint_family']=='b' else 0) for r,w in zip(rs,ws))/len(rs)
        self.assertAlmostEqual(score(rows,weights),4.5)
        self.assertAlmostEqual(score(duplicated,other_weights),4.5)

    def test_source_tampering_rejected(self):
        self.plan.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed target source'):
            load(self.paths)

    def test_unhashed_prefix_rejected(self):
        self.change_target(lambda t: t['sources_sha256'].pop(t['prefix']))
        with self.assertRaisesRegex(ValueError, 'bind prefix'):
            load(self.paths)

    def test_repeated_mechanical_run_is_not_another_stream(self):
        with self.assertRaisesRegex(ValueError, 'duplicate family/stream'):
            load([*self.paths,self.paths[0]])

    def test_new_package_id_does_not_make_a_recaptured_world_independent(self):
        target = json.loads(self.paths[0].read_text())
        fragment = json.loads(pathlib.Path(target['prefix']).read_text())
        fragment['checkpoint_family'] = 'new-package-same-condition-and-root'
        prefix = self.write('recaptured-prefix.json', fragment)
        target.update(prefix=str(prefix), checkpoint_family=fragment['checkpoint_family'])
        target['sources_sha256'][str(prefix)] = sha(prefix)
        extra = self.write('recaptured-target.json', target)
        with self.assertRaisesRegex(ValueError, 'same TRAIN condition/root'):
            load([*self.paths, extra])

    def test_single_stream_rejected(self):
        with self.assertRaisesRegex(ValueError, 'two paired streams'):
            load(self.paths[:1])

    def test_fire_scope_must_match_target_and_cannot_mix_joint_sampling(self):
        for idx,path in enumerate(self.paths):
            target = json.loads(path.read_text())
            prefix = pathlib.Path(target['prefix'])
            fragment = json.loads(prefix.read_text())
            fragment['sampling_checkpoint'].update(version='hjepa_sampling_checkpoint_v3',sampling_scope='fire_only_v1')
            fragment['version'] = 'hjepa_native_fragment_v3'
            self.write(prefix.name,fragment)
            target['sources_sha256'][str(prefix)] = sha(prefix)
            self.write(path.name,target)
            with self.assertRaisesRegex(ValueError,'target sampling scope differs'):
                load(self.paths)
            target['sampling_scope'] = 'fire_only_v1'
            target['version'] = 'hjepa_fragment_mc_targets_v2'
            self.write(path.name,target)
            if idx == 0:
                with self.assertRaisesRegex(ValueError,'mixed joint/fire-only'):
                    load(self.paths)
        self.assertEqual(load(self.paths)[3]['sampling_scope'],'fire_only_v1')
        target = json.loads(self.paths[0].read_text());prefix = pathlib.Path(target['prefix'])
        fragment = json.loads(prefix.read_text());fragment['version'] = 'hjepa_native_fragment_v2'
        self.write(prefix.name,fragment);target['sources_sha256'][str(prefix)] = sha(prefix)
        self.write(self.paths[0].name,target)
        with self.assertRaisesRegex(ValueError,'requires new fragment and target versions'):
            load(self.paths)

    def test_conditional_fire_roots_reject_nonfire_changes_and_later_states(self):
        def row(attack):
            return dict(checkpoint_family='root',index=0,logical_sampling_identity=dict(frame=180),features=[0.]*871,
                        sample=dict(latent=[.1,.2,.3,.4],vertical=2,weapon=1,attack=attack))
        validate_fire_roots([row(True),row(False)])
        for mutate in [lambda r:r.update(index=1),lambda r:r['features'].__setitem__(0,1.),
                       lambda r:r['sample']['latent'].__setitem__(2,9.),lambda r:r['sample'].update(vertical=0),
                       lambda r:r['sample'].update(weapon=2)]:
            changed = row(False);mutate(changed)
            with self.assertRaises(ValueError):validate_fire_roots([row(True),changed])

    def test_horizon_cannot_supply_zero_bootstrap(self):
        self.change_target(lambda t: t.update(bootstrap_proof='horizon'))
        with self.assertRaisesRegex(ValueError, 'native terminal targets'):
            load(self.paths)

    def test_nonfinite_return_rejected(self):
        self.change_target(lambda t: t['targets'][0].update(return_to_go=float('nan')))
        with self.assertRaisesRegex(ValueError, 'nonfinite terminal return'):
            load(self.paths)

    def test_diagnostic_cannot_enter_even_with_updated_hash(self):
        p = self.root/'prefix-1.json'
        f = json.loads(p.read_text());f['split']='diagnostic'
        self.write(p.name,f)
        self.change_target(lambda t: t['sources_sha256'].update({str(p):sha(p)}))
        with self.assertRaisesRegex(ValueError, 'verified TRAIN'):
            load(self.paths)

    def test_target_model_mismatch_rejected(self):
        self.change_target(lambda t: t.update(model_version='hjepa:other'))
        with self.assertRaisesRegex(ValueError, 'terminal/model binding'):
            load(self.paths)

    def test_native_objective_requires_zero_counters_and_exact_terminal_identity(self):
        identity = dict(map='base1', spawncount=9, frame=4)
        self.change_target(lambda t: t.update(terminal_reason='native_objective_clear',
            terminal_next_identity=identity, terminal_native_objective=dict(identity, alive=0, pending=0)))
        self.assertEqual(load(self.paths)[3]['rows'], 2)
        for change in [dict(pending=1),dict(alive=1),dict(frame=3),dict(map='base2'),dict(spawncount=8)]:
            self.change_target(lambda t: t.update(terminal_native_objective=dict(identity, alive=0, pending=0)))
            self.change_target(lambda t: t['terminal_native_objective'].update(change))
            with self.assertRaisesRegex(ValueError, 'objective zero state'):
                load(self.paths)

    def test_long_credit_tail_cannot_become_optimizer_prefix(self):
        p = self.root/'prefix-1.json'
        f = json.loads(p.read_text())
        row = f['transitions'][0]
        f['transitions'] = [dict(row, index=i) for i in range(101)]
        self.write(p.name, f)
        self.change_target(lambda t: t.update(terminal_step=101,
            sources_sha256={**t['sources_sha256'], str(p):sha(p)},
            targets=[dict(index=i,return_to_go=0.) for i in range(101)]))
        with self.assertRaisesRegex(ValueError, 'short TRAIN prefix'):
            load(self.paths)


if __name__ == '__main__':
    unittest.main()
