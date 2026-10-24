"""Explicit resumable BT1/BT4 weapon-only forks; disposable CUDA steps are not fitting."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from evaluate_registered_combat_models import read, sha
from ppo_combat import torch, nn, training_devices
from ppo_recurrent import prepare_context, sequence, durable_json, durable_write
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_weapon_only_scope import SCOPE, WeaponOnlyScope, inherited_optimizer
from combat_checkpoint_scope import require_checkpoint_scope
from combat_projectile_training_scope import SCOPE as PARENT_SCOPE
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'prepared', 'data', 'protocol', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    training_devices()
    torch.set_default_device('cuda'); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        assert expected is None or expected == digest, str(path)
        pins[str(path)] = digest
        return path

    for path in Path(__file__).parent.glob('*.py'):
        pin(path)
    protocol = read(pin(a.protocol))
    assert protocol['version'] == SCOPE and protocol['profiles'] == {'control': 1, 'explore': 4}
    seal = read(pin(a.source/'complete.json'))
    assert seal['weights_sha256'] == protocol['source_weights_sha256']
    for name, key in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')):
        pin(a.source/name, seal[key])
    source = read(a.source/'weights.json')
    cp = torch.load(a.source/'checkpoint.pt', map_location='cuda', weights_only=True)
    original = copy.deepcopy(cp)
    require_checkpoint_scope(cp, PARENT_SCOPE)
    assert cp['weights_sha256'] == seal['weights_sha256']
    assert not cp.get('sequence_retention'), 'Nonempty retention needs a separate explicit migration'
    confirmation = read(pin(protocol['confirmation_report'], protocol['confirmation_report_sha256']))
    assert confirmation['state'] == 'complete' and not confirmation['gate_passed']
    assert not confirmation['training_performed'] and not confirmation['promotion']
    prep_seal = read(pin(a.prepared/'complete.json'))
    prep = read(pin(a.prepared/'report.json', prep_seal['report_sha256']))
    assert prep['device'] == 'cuda'
    meta = read(pin(a.data/'report.json'))
    assert meta['model_sha256'] == seal['weights_sha256'] and meta['numerical_verification'] == 'cuda_verified_v1'
    with pin(a.data/'rollout.jsonl', meta['rollout_sha256']).open(encoding='utf-8-sig') as stream:
        rows = list(map(json.loads, stream))
    with pin(a.data/'sequence.jsonl', meta['sequence_sha256']).open(encoding='utf-8-sig') as stream:
        histories = list(map(json.loads, stream))
    # The specified data are previously verified own-TRAIN, never confirmation captures.
    assert str(a.data.resolve()) == str(Path(protocol['own_train_data']).resolve())
    assert protocol['own_train_split'] == 'train'
    plan = read(pin(protocol['own_train_plan'], protocol['own_train_plan_sha256']))
    assert plan['model_sha256'] == seal['weights_sha256']
    assert plan['tasks'] and all(t['split'] == 'train' and t['modes'] == ['learned'] for t in plan['tasks'])
    seeds = {seed for task in plan['tasks'] for seed in task['seeds']}
    assert rows and {row['seed'] for row in rows} <= seeds
    context = prepare_context(histories, rows, 'cuda')
    old_actor = branch(base_model(source, allow_shared=True), 'actor')
    old_value = branch(base_model(source, allow_shared=True), 'value')
    assert equal_tree(old_actor.state_dict(), cp['actor']) and equal_tree(old_value.state_dict(), cp['value'])
    with torch.no_grad():
        old_raw, err = sequence(old_actor, context, 32, True, 'actor')
        old_v, verr = sequence(old_value, context, 32, True, 'value')
        assert max(err, verr) < 2e-5
    active = inherited_optimizer(cp)
    a.out.mkdir(parents=True)
    summaries = []
    for label, profile in (('control', 'zero'), ('explore', 'blaster4')):
        weights = pin(a.prepared/('models/'+profile+'/weights.json'))
        model = read(weights)
        assert model['weapon_exploration']['blaster_temperature'] == protocol['profiles'][label]
        old_fields, new_fields = copy.deepcopy(source), copy.deepcopy(model)
        old_fields.pop('weapon_exploration'); new_fields.pop('weapon_exploration')
        assert old_fields == new_fields
        actor = branch(base_model(model, allow_shared=True), 'actor')
        value = branch(base_model(model, allow_shared=True), 'value')
        assert equal_tree(actor.state_dict(), cp['actor']) and equal_tree(value.state_dict(), cp['value'])
        with torch.no_grad():
            raw, err = sequence(actor, context, 32, True, 'actor')
            v, verr = sequence(value, context, 32, True, 'value')
            assert max(err, verr) < 2e-5 and torch.equal(v, old_v)
            assert torch.equal(raw[:, :8], old_raw[:, :8]) and torch.equal(raw[:, 20:], old_raw[:, 20:])
            if label == 'control':
                assert torch.equal(raw, old_raw)
        std = nn.Parameter(cp['log_std'].detach().clone())
        scope = WeaponOnlyScope(actor, std, active)
        assert equal_tree(scope.optimizer.state_dict(), active)
        scope.verify_frozen()
        fork = copy.deepcopy(cp)
        metadata = scope.metadata(0)
        metadata.update(initializer_protocol_sha256=sha(a.protocol), profile=label,
                        source_weights_sha256=seal['weights_sha256'], feature_version=model['feature_version'],
                        combat_execution_contract=model.get('combat_execution_contract', 'legacy_v1'))
        fork.update(checkpoint_training_scope=SCOPE, weapon_only_scope=metadata,
                    weapon_only_optimizer=copy.deepcopy(active),
                    weapon_only_value_optimizer=copy.deepcopy(cp['projectile_value_optimizer']),
                    weapon_only_history=[])
        target = a.out/label; target.mkdir()
        shutil.copyfile(weights, target/'weights.json')
        fork['weights_sha256'] = sha(target/'weights.json')
        durable_write(target/'checkpoint.pt', lambda stream: torch.save(fork, stream))
        restored = torch.load(target/'checkpoint.pt', map_location='cuda', weights_only=True)
        assert equal_tree(restored, fork)
        require_checkpoint_scope(restored, SCOPE)
        for incompatible in (None, PARENT_SCOPE, 'combat_decision_scope_ppo_v1'):
            try:
                require_checkpoint_scope(restored, incompatible)
            except ValueError:
                pass
            else:
                raise AssertionError('Incompatible trainer admitted')
        continued = []
        for checkpoint in (fork, restored):
            probe = branch(base_model(model, allow_shared=True), 'actor')
            probe.load_state_dict(checkpoint['actor'])
            ps = WeaponOnlyScope(probe, nn.Parameter(checkpoint['log_std'].clone()), checkpoint['weapon_only_optimizer'])
            before = copy.deepcopy(probe.state_dict())
            prediction, _ = sequence(probe, context, 32, False, 'actor')
            loss = prediction[:, 8:20].square().mean()
            ps.optimizer.zero_grad(set_to_none=True); loss.backward()
            assert bool(torch.isfinite(nn.utils.clip_grad_norm_(ps.parameters, .5)))
            ps.optimizer.step(); ps.verify_frozen()
            assert any(not torch.equal(before[k], v) for k, v in probe.state_dict().items() if k.startswith('weapon_state_branch.'))
            with torch.no_grad():
                updated, _ = sequence(probe, context, 32, False, 'actor')
                assert torch.equal(updated[:, :8], raw[:, :8]) and torch.equal(updated[:, 20:], raw[:, 20:])
            continued.append((copy.deepcopy(probe.state_dict()), copy.deepcopy(ps.optimizer.state_dict())))
        assert equal_tree(*continued), 'Resumed weapon Adam step differs'
        critics = []
        for checkpoint in (fork, restored):
            probe = branch(base_model(model, allow_shared=True), 'value')
            probe.load_state_dict(checkpoint['value'])
            opt = torch.optim.Adam(probe.parameters(), lr=.001)
            opt.load_state_dict(copy.deepcopy(checkpoint['weapon_only_value_optimizer']))
            prediction, _ = sequence(probe, context, 32, False, 'value')
            opt.zero_grad(set_to_none=True); prediction.square().mean().backward()
            assert bool(torch.isfinite(nn.utils.clip_grad_norm_(probe.parameters(), .5)))
            opt.step()
            assert not equal_tree(probe.state_dict(), checkpoint['value'])
            critics.append((copy.deepcopy(probe.state_dict()), copy.deepcopy(opt.state_dict())))
        assert equal_tree(*critics), 'Resumed critic Adam step differs'
        assert equal_tree(cp, original)
        for key, expected in cp.items():
            if key not in ('checkpoint_training_scope', 'weights_sha256'):
                assert equal_tree(expected, restored[key]), key
        report = dict(state='complete', device='cuda', version=SCOPE, profile=label,
                      feature_version=model['feature_version'], combat_execution_contract=metadata['combat_execution_contract'],
                      metadata=metadata, own_train_rows=len(rows), trainable_parameters=sum(p.numel() for p in scope.parameters),
                      source_actor_value_std_exact=True, parent_optimizers_rng_history_exact=True,
                      active_weapon_adam_inherited=True, critic_adam_inherited=True,
                      unchanged_other_outputs_exact=True, actor_critic_resumed_steps_exact=True,
                      checkpoint_roundtrip_exact=True, incompatible_scopes_rejected=True,
                      disposable_steps_only=True, persisted_optimizer_steps=0, training_performed=False, promotion=False)
        durable_json(target/'report.json', report)
        durable_json(target/'complete.json', dict(state='complete', weights_sha256=sha(target/'weights.json'),
            checkpoint_sha256=sha(target/'checkpoint.pt'), report_sha256=sha(target/'report.json')))
        summaries.append(dict(profile=label, weights_sha256=sha(target/'weights.json'), parameters=report['trainable_parameters']))
    assert all(sha(path) == digest for path, digest in pins.items())
    durable_json(a.out/'report.json', dict(state='complete', device='cuda', version=SCOPE,
        profiles=summaries, inputs=pins, training_performed=False, promotion=False))
    durable_json(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json')))
    print(summaries)


if __name__ == '__main__':
    main()
