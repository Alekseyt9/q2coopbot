"""Matched own-policy projectile-type PPO; neutral retains projectile presence."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

from evaluate_registered_combat_models import read, sha
from ppo_combat import torch, nn, training_devices, advantages, guarded_actor_step, validate_objective
from ppo_recurrent import prepare_context, sequence, durable_json, durable_write
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model, events, probabilities, validate
from combat_aim_candidates import VERSION
from combat_geometry_ppo import GeometryJoint
from combat_precision_head import selected_means
from combat_weapon_state_residual import optimizer as actor_optimizer
from combat_spatial_aim import export as export_actor_branches
from combat_checkpoint_scope import require_checkpoint_scope
from combat_execution_contract import LEGACY, require_same
from combat_body_features import LEGACY_WIDTH
from combat_projectile_type_features import VERSION as TYPE_FEATURE_VERSION, NEUTRAL_VERSION, PARENT_WIDTH, WIDTH, encode_tail
from combat_projectile_training_scope import SCOPE as DECISION_SCOPE, ProjectileTrainingScope, RATES
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'corpus', 'native-credit', 'advantage-credit', 'protocol', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--control-update', type=Path)
    args = parser.parse_args()
    assert not args.out.exists()
    training_devices()
    torch.set_default_device('cuda')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = pins.get(str(path)) or sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    def sealed(directory):
        seal = read(pin(directory / 'complete.json'))
        assert seal['state'] == 'complete'
        return seal, read(pin(directory / 'report.json', seal['report_sha256']))

    pin(__file__)
    protocol = read(pin(args.protocol))
    assert protocol['version'] == DECISION_SCOPE and protocol['actor_steps'] == 30 and protocol['critic_steps'] == 40
    assert protocol['group_lrs'] == list(RATES) and protocol['target_full_kl'] == .005 and protocol['target_base_kl'] == .0025
    source_seal, source_report = sealed(args.source)
    model = read(pin(args.source / 'weights.json', source_seal['weights_sha256']))
    pin(args.source / 'checkpoint.pt', source_seal['checkpoint_sha256'])
    cp = torch.load(args.source / 'checkpoint.pt', map_location='cuda', weights_only=True)
    original = copy.deepcopy(cp)
    # CPU deserialization only transports RNG bytes and serialized checkpoint metadata.
    serial = torch.load(args.source / 'checkpoint.pt', map_location='cpu', weights_only=True)
    require_checkpoint_scope(cp, DECISION_SCOPE)
    assert cp['projectile_training_scope']['protocol_sha256'] == protocol['initialization_protocol_sha256']
    assert cp['projectile_training_scope']['version'] == DECISION_SCOPE
    contract = require_same(cp['config'].get('combat_execution_contract'),model.get('combat_execution_contract'),
        cp['projectile_training_scope']['combat_execution_contract'],source_report['combat_execution_contract'])
    assert contract == LEGACY, 'Both projectile-type arms use the unchanged legacy execution contract'
    scope_updates = cp['projectile_training_scope']['updates_completed']
    assert cp['projectile_training_scope'].get('training_protocol_sha256') == (None if scope_updates == 0 else sha(args.protocol))
    saved_optimizer = cp['projectile_training_optimizer']
    feature_version = model['feature_version']
    assert feature_version in (TYPE_FEATURE_VERSION, NEUTRAL_VERSION)
    assert source_report['feature_version'] == feature_version == cp['config']['projectile_type_feature_version'] == cp['projectile_training_scope']['feature_version']
    assert cp['projectile_training_scope']['shared_input_bias'] is True and protocol['shared_input_bias'] is True
    assert cp['weights_sha256'] == source_seal['weights_sha256']
    assert cp['config']['objective_reward_sha256'] == protocol['reward_sha256']
    assert cp['config']['gamma']==.99 and cp['config']['lambda']==.95 and cp['config']['clip']==.2
    assert cp['config']['entropy']==.001 and cp['config']['value_lr']==.001
    closed = read(pin(args.corpus / 'complete.json'))
    collection = read(pin(args.corpus / 'protocol.json', closed['protocol_sha256']))
    assert collection['train_only'] and collection['ppo_training_objective']['reward_sha256'] == protocol['reward_sha256']
    assert closed['native_cases'] == closed['cuda_cases'] == protocol['cases_per_update']
    assert len(collection['assignments']) == 1
    assignment = collection['assignments'][0]
    plan = read(pin(assignment['plan'], assignment['plan_sha256']))
    assert assignment['split'] == 'train' and plan['model_sha256'] == source_seal['weights_sha256']
    assert all(task['split'] == 'train' and task['modes'] == ['learned'] for task in plan['tasks'])
    assert {task['episode']['id'] for task in plan['tasks']} == set(protocol['episodes'])
    data = args.corpus / 'processing-train/merged'
    meta = read(pin(data / 'report.json'))
    assert meta['numerical_verification'] == 'cuda_verified_v1' and not meta.get('diagnostic_only')
    assert meta['model_sha256'] == source_seal['weights_sha256'] and meta['action_distribution'] == VERSION
    assert meta['rollout_sha256'] not in cp['consumed_rollouts'], 'Already consumed on this source checkpoint'
    validate_objective(cp['config'], meta)
    assert meta['feature_version'] == feature_version
    require_same(contract,meta.get('combat_execution_contract'),plan.get('combat_execution_contract'),collection['combat_execution_contract'])
    assert plan['checked_wall_slide'] and not plan.get('diagnostic_stopped_crouch') and not plan.get('diagnostic_checked_ground_request')
    proofs = 0
    for path,digest in meta['source_sha256'].items():
        pin(path,digest)
        if Path(path).name == 'cuda-verification.json':
            proof = read(path)
            assert proof['state'] == 'passed' and proof['device'] == 'cuda'
            assert proof['frame_hash_candidate_assignments_verified'] and proof['original_latent_transform_verified']
            proofs += 1
    assert proofs == protocol['cases_per_update']
    _, native = sealed(args.native_credit)
    _, credit = sealed(args.advantage_credit)
    assert native['inputs'][str((args.corpus / 'complete.json').resolve())] == sha(args.corpus / 'complete.json')
    assert credit['inputs'][str((args.native_credit / 'report.json').resolve())] == sha(args.native_credit / 'report.json')
    assert native['totals']['rows'] == credit['rows'] == meta['rows'] and not native['unassigned_native_effects']
    assert meta['handoff_bootstrap'] == 'nonterminal_v1'
    assert all(row['bootstrap_contract_verified'] for row in native['handoffs'])
    with pin(data / 'rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig') as stream:
        rows = list(map(json.loads,stream))
    with pin(data / 'sequence.jsonl',meta['sequence_sha256']).open(encoding='utf-8-sig') as stream:
        context = list(map(json.loads,stream))
    assert len(rows) == meta['rows'] and {row['seed'] for row in rows} == set(assignment['seeds'])
    prepared = prepare_context(context,rows,'cuda')
    x = torch.tensor([row['features'] for row in rows],device='cuda',dtype=torch.float32)
    assert x.shape == (len(rows), WIDTH)
    assert not bool(x[:,LEGACY_WIDTH:PARENT_WIDTH].any()), 'Body information leaked into projectile experiment'
    collection_report = read(pin(args.corpus/'processing-train/report.json'))
    observations = {}
    for case in collection_report['cases']:
        directory = args.corpus/'processing-train'/f"case-{case['job']:04d}"
        cm = read(pin(directory/'cuda/report.json'))
        step_path = next((directory/'native').glob('replay-*/steps.jsonl')).resolve()
        with pin(step_path, cm['source_sha256'][str(step_path)]).open(encoding='utf-8-sig') as stream:
            for step in map(json.loads, stream):
                observations[case['seed'],step['index']] = step['observation']
    observed = [observations[row['seed'],row['index']] for row in rows]
    tail = encode_tail(observed,feature_version)
    assert torch.equal(x[:,PARENT_WIDTH:],tail), 'Go/CUDA type encoding mismatch before fitting'
    typed = encode_tail(observed,TYPE_FEATURE_VERSION).reshape(len(rows),4,5)
    presence = typed.sum(-1)
    projectile_presence_rows = int((presence.sum(-1)>0).sum())
    assert projectile_presence_rows>0, 'No projectile examples in this fixed TRAIN corpus'
    projectile_class_slot_counts = typed.sum((0,1)).to(torch.int64).tolist()
    if feature_version == NEUTRAL_VERSION:
        slots = tail.reshape(len(rows),4,5)
        assert not bool(slots[:,:,:4].any()) and torch.equal(slots[:,:,4],presence)
    category = lambda key: torch.tensor([row['sample'].get(key,0) for row in rows],device='cuda',dtype=torch.long)
    target, mode, vertical, weapon = (category(key) for key in ('target','aim_mode','vertical','weapon'))
    final = torch.tensor([row['sample']['latent'] for row in rows],device='cuda',dtype=torch.float32)
    attack = torch.tensor([float(row['sample']['attack']) for row in rows],device='cuda')
    source_z, choice, timing, active = events(rows,x,target,mode,vertical,weapon)
    core = base_model(model,allow_shared=True)
    actor, value, joint = branch(core,'actor'), branch(core,'value'), GeometryJoint(model)
    std = nn.Parameter(torch.tensor(model['log_std'],device='cuda'))
    assert equal_tree(actor.state_dict(),cp['actor']) and equal_tree(value.state_dict(),cp['value'])
    legacy = actor_optimizer(actor,std,3e-5,.0003,.0003,.0003)
    legacy.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    value_optimizer = torch.optim.Adam(value.parameters(),lr=.001)
    value_optimizer.load_state_dict(copy.deepcopy(cp['projectile_value_optimizer']))

    def likelihood(module, raw, gaussian_std, scorer=joint):
        return probabilities(scorer,raw,gaussian_std,source_z,final,attack,vertical,weapon,x,target,mode,choice,timing,
            module.sampled_motion_aim_branch,module.motor_branch,module.conditional_fire_branch)

    with torch.no_grad():
        old_raw, memory_error = sequence(actor,prepared,32,True,'actor')
        old_value, value_memory = sequence(value,prepared,32,True,'value')
        old_lp = likelihood(actor,old_raw,std)[0]
        old_means = selected_means(old_raw,target,mode,x,source_z[:,:2],actor.sampled_motion_aim_branch,vertical,actor.motor_branch)
        native_lp = torch.tensor([row['sample']['log_probability'] for row in rows],device='cuda')
        replay_error = float((old_lp-native_lp).abs().max())
        assert replay_error < .003 and max(memory_error,value_memory) < 2e-5
    adv, returns = advantages(rows,.99,.95)
    advantage = torch.tensor(adv,device='cuda')
    advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    expected_value = torch.tensor(returns,device='cuda')
    scope = ProjectileTrainingScope(actor,std,legacy,saved_optimizer=saved_optimizer)
    joint.requires_grad_(False)
    args.out.mkdir(parents=True)
    durable_json(args.out/'progress.json',dict(stage='training',device='cuda'))
    torch.set_rng_state(serial['rng'])
    torch.cuda.set_rng_state_all(serial['cuda_rng'])

    def metrics():
        raw,_ = sequence(actor,prepared,32,False,'actor')
        lp,entropy,_,current_active = likelihood(actor,raw,std)
        assert torch.equal(active,current_active)
        delta = lp-native_lp
        ratio = delta.exp()
        loss = -torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean() - .001*entropy.mean()
        base_delta = lp-old_lp
        return loss, (ratio-1-delta).mean(), (base_delta.exp()-1-base_delta).mean(), raw, lp

    def objective():
        loss,full,base,_,_ = metrics()
        return loss,torch.maximum(full,2*base)

    with torch.no_grad():
        initial = metrics()
        assert float((initial[3]-old_raw).abs().max()) < 5e-5
        initial_loss = float(initial[0])
    started, trials = time.perf_counter(), []
    projectile_before = scope.input.projectile.detach().clone()
    old_body_columns = scope.input.body.detach().clone()
    bias_before = scope.input.bias.detach().clone()
    first_gradients = None
    for _ in range(30):
        loss,_ = objective()
        scope.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if first_gradients is None:
            first_gradients = dict(projectile=float(scope.input.projectile.grad.norm()), shared_bias=float(scope.input.bias.grad.norm()))
        assert bool(torch.isfinite(nn.utils.clip_grad_norm_(scope.parameters,.5)))
        trial = guarded_actor_step(scope.parameters,scope.optimizer,objective,loss.detach(),RATES[0],.005,group_lrs=list(RATES))
        trials.append(trial)
        if not trial['accepted']:
            break
    for _ in range(40):
        predicted,_ = sequence(value,prepared,32,False,'value')
        loss = (predicted[:,0]-expected_value).square().mean()
        value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert bool(torch.isfinite(nn.utils.clip_grad_norm_(value.parameters(),.5)))
        value_optimizer.step()
    with torch.no_grad():
        loss,full_kl,base_kl,raw,wrapped_lp = metrics()
        assert float(full_kl) <= .005+1e-6 and float(base_kl) <= .0025+1e-6
        means = selected_means(raw,target,mode,x,source_z[:,:2],actor.sampled_motion_aim_branch,vertical,actor.motor_branch)
        coordinate_kl = torch.distributions.kl_divergence(torch.distributions.Normal(old_means,std.exp()),torch.distributions.Normal(means,std.exp())).mean(0)
        assert bool(torch.isfinite(coordinate_kl).all())
    projectile_changed = not torch.equal(projectile_before,scope.input.projectile)
    shared_bias_changed = not torch.equal(bias_before,scope.input.bias)
    assert torch.equal(old_body_columns,scope.input.body), 'Old zero body channels changed'
    assert first_gradients['projectile']>0 and first_gradients['shared_bias']>0
    assert projectile_changed or not any(t['accepted'] for t in trials)
    scope.materialize()
    scope.verify_frozen()
    assert equal_tree(legacy.state_dict(),scope.legacy_before) and equal_tree(cp,original)
    if args.control_update:
        control_seal, control = sealed(args.control_update)
        assert control['behavior_sha256'] == source_seal['weights_sha256'] and control['rollout_sha256'] == meta['rollout_sha256']
        pin(args.control_update/'checkpoint.pt',control_seal['checkpoint_sha256'])
        matched = torch.load(args.control_update/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert equal_tree(value.state_dict(),matched['value']) and equal_tree(value_optimizer.state_dict(),matched['projectile_value_optimizer'])
    base_a, cell_a = actor.export()
    base_v, cell_v = value.export()
    key = 'attention' if core.get('attention') else 'memory'
    updated_core = copy.deepcopy(core)
    updated_core.update(actor=base_a,value=base_v,log_std=std.detach().tolist())
    updated_core[key]['actor'], updated_core[key]['value'] = cell_a, cell_v
    export_actor_branches(updated_core,actor)
    updated = joint.export({**updated_core,'action_distribution':VERSION,'aim_candidate_head':copy.deepcopy(model['aim_candidate_head'])})
    validate(updated,allow_shared=True)
    assert updated['aim_candidate_head'] == model['aim_candidate_head'] and updated['log_std'] == model['log_std']
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        decoded = branch(base_model(updated,allow_shared=True),'actor')
        assert equal_tree(decoded.state_dict(),actor.state_dict())
        with torch.no_grad():
            decoded_raw,_ = sequence(decoded,prepared,32,False,'actor')
            decoded_lp = likelihood(decoded,decoded_raw,std)[0]
            export_error = float((decoded_lp-wrapped_lp).abs().max())
            assert export_error < 1e-4
    durable_json(args.out/'weights.json',updated)
    done = sum(trial['accepted'] for trial in trials)
    fork = copy.deepcopy(serial)
    record = dict(version=DECISION_SCOPE,source_weights_sha256=source_seal['weights_sha256'],rollout_sha256=meta['rollout_sha256'],actor_steps=done,critic_steps=40)
    scope_metadata = scope.metadata(scope_updates+1)
    scope_metadata['protocol_sha256'] = protocol['initialization_protocol_sha256']
    scope_metadata['training_protocol_sha256'] = sha(args.protocol)
    scope_metadata.update(version=DECISION_SCOPE,combat_execution_contract=contract,feature_version=feature_version,
        initial_source_weights_sha256=cp['projectile_training_scope'].get('initial_source_weights_sha256',source_seal['weights_sha256']),root_source_weights_sha256=protocol['root_source_weights_sha256'])
    fork.update(checkpoint_training_scope=DECISION_SCOPE,projectile_training_scope=scope_metadata,
        projectile_training_optimizer=copy.deepcopy(scope.optimizer.state_dict()),actor_optimizer=copy.deepcopy(legacy.state_dict()),
        weights_sha256=sha(args.out/'weights.json'),actor=actor.state_dict(),value=value.state_dict(),
        projectile_value_optimizer=value_optimizer.state_dict(),log_std=std.detach(),
        consumed_rollouts=cp['consumed_rollouts']+[meta['rollout_sha256']],updates_completed=cp['updates_completed']+1,
        total_actor_steps=cp['total_actor_steps']+done,projectile_training_history=cp.get('projectile_training_history',[])+[record],
        actor_training={**cp['actor_training'],'scope':DECISION_SCOPE},rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    durable_write(args.out/'checkpoint.pt',lambda stream:torch.save(fork,stream))
    loaded = torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    require_checkpoint_scope(loaded,DECISION_SCOPE)
    require_same(contract,loaded['config']['combat_execution_contract'],loaded['projectile_training_scope']['combat_execution_contract'],updated.get('combat_execution_contract'))
    for anchor in ('decision_scope','decision_scope_optimizer','decision_scope_history','execution_contract_scope','execution_contract_optimizer','execution_contract_history','body_input_scope','body_input_optimizer','body_input_history','value_optimizer','body_training_scope','body_training_optimizer','body_training_history','body_value_optimizer','geometry_aim_training'):
        assert equal_tree(cp[anchor],loaded[anchor]),anchor
    for name in ('actor','value','actor_optimizer','projectile_value_optimizer','geometry_aim_head','geometry_aim_optimizer','aim_candidate_head','aim_candidate_optimizer','rng','cuda_rng'):
        expected = cp[name] if name in ('geometry_aim_head','geometry_aim_optimizer','aim_candidate_head','aim_candidate_optimizer','rng','cuda_rng') else fork[name]
        assert equal_tree(expected,loaded[name]),name
    assert loaded['consumed_rollouts'][:-1] == cp['consumed_rollouts']
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        continued = []
        for state in (fork['projectile_training_optimizer'],loaded['projectile_training_optimizer']):
            module = branch(base_model(updated,allow_shared=True),'actor')
            new_std = nn.Parameter(torch.tensor(updated['log_std'],device='cuda'))
            anchor = actor_optimizer(module,new_std,3e-5,.0003,.0003,.0003)
            anchor.load_state_dict(copy.deepcopy(loaded['actor_optimizer']))
            restored = ProjectileTrainingScope(module,new_std,anchor,saved_optimizer=state)
            assert equal_tree(restored.optimizer.state_dict(),loaded['projectile_training_optimizer'])
            before = [parameter.detach().clone() for parameter in restored.parameters]
            raw,_ = sequence(module,prepared,32,False,'actor')
            lp,entropy,_,_ = likelihood(module,raw,new_std)
            ratio = (lp-native_lp).exp()
            probe_loss = -torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()-.001*entropy.mean()
            restored.optimizer.zero_grad(set_to_none=True)
            probe_loss.backward()
            nn.utils.clip_grad_norm_(restored.parameters,.5)
            restored.optimizer.step()
            assert any(not torch.equal(a,b) for a,b in zip(before,restored.parameters))
            continued.append((copy.deepcopy(module.state_dict()),copy.deepcopy(restored.optimizer.state_dict())))
        assert equal_tree(continued[0],continued[1]),'Disposable resumed Adam step differs'
        critic_continued = []
        for state in (fork,loaded):
            module = branch(base_model(updated,allow_shared=True),'value')
            assert equal_tree(module.state_dict(),state['value'])
            optimizer = torch.optim.Adam(module.parameters(),lr=.001)
            optimizer.load_state_dict(copy.deepcopy(state['projectile_value_optimizer']))
            before = {key:tensor.detach().clone() for key,tensor in module.state_dict().items()}
            prediction,_ = sequence(module,prepared,32,False,'value')
            probe_loss = (prediction[:,0]-expected_value).square().mean()
            optimizer.zero_grad(set_to_none=True);probe_loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(),.5);optimizer.step()
            assert any(not torch.equal(before[key],tensor) for key,tensor in module.state_dict().items())
            critic_continued.append((copy.deepcopy(module.state_dict()),copy.deepcopy(optimizer.state_dict())))
        assert equal_tree(critic_continued[0],critic_continued[1]),'Disposable resumed critic Adam step differs'
    for path,digest in pins.items():
        assert sha(path)==digest,path
    frozen = args.out/'trainer-sources'
    frozen.mkdir()
    helpers = {}
    for path in sorted({Path(__file__).resolve()}|{Path(module.__file__).resolve() for module in list(sys.modules.values())
                       if getattr(module,'__file__',None) and Path(module.__file__).resolve().parent==Path(__file__).resolve().parent and Path(module.__file__).suffix=='.py'}):
        shutil.copy2(path,frozen/path.name)
        helpers[path.name] = sha(frozen/path.name)
    report = dict(state='complete',version=DECISION_SCOPE,device='cuda',combat_execution_contract=contract,source=str(args.source.resolve()),
        behavior_sha256=source_seal['weights_sha256'],weights_sha256=sha(args.out/'weights.json'),rows=len(rows),
        rollout_sha256=meta['rollout_sha256'],actor_steps=done,critic_steps=40,trials=trials,
        initial_loss=initial_loss,final_loss=float(loss),full_sample_k3=float(full_kl),base_sample_k3=float(base_kl),
        movement_conditional_kl=float(coordinate_kl[:2].sum()),aim_conditional_kl=float(coordinate_kl[2:].sum()),
        source_lp_native_max_error=replay_error,json_lp_max_error=export_error,scope_updates=scope_updates+1,
        source_checkpoint_exact=True,frozen_original_actor_parameters_exact=True,original_coordinate_rows_exact=True,frozen_legacy_optimizer_exact=True,
        feature_version=feature_version,shared_input_bias=True,projectile_presence_rows=projectile_presence_rows,projectile_class_slot_counts=projectile_class_slot_counts,
        projectile_columns_changed=projectile_changed,body_columns_frozen_exact=True,shared_bias_changed=shared_bias_changed,first_gradients=first_gradients,coordinate_kl=coordinate_kl.tolist(),
        matched_control_critic_exact=bool(args.control_update),checkpoint_roundtrip_exact=True,
        disposable_resume_adam_step_exact=True,disposable_critic_resume_adam_step_exact=True,disposable_resume_step_used_kl_guard=False,
        rng_preserved=True,training_performed=True,promotion=False,quality_verified=False,
        experiment_protocol_sha256=sha(args.protocol),inputs=pins,trainer_sources=helpers,seconds=time.perf_counter()-started,
        scope='Own-policy TRAIN PPO, eight-group projectile input/decision optimizer with shared input bias in both arms. Original v10 input columns and original coordinate rows stay frozen; their outputs may change via new inputs/bias. '
              'Resume proof is one disposable non-guarded Adam continuation, not a fresh on-policy update or battle acceptance.')
    durable_json(args.out/'report.json',report)
    durable_json(args.out/'complete.json',dict(state='complete',weights_sha256=report['weights_sha256'],checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    durable_json(args.out/'progress.json',dict(stage='complete',actor_steps=done))
    print({key:value for key,value in report.items() if key not in ('inputs','trials','trainer_sources')})


if __name__ == '__main__':
    main()
