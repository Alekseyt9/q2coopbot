"""Fork only the explicit reward objective; preserve policy/optimizer/RNG state."""
import argparse
import copy
from pathlib import Path
import shutil
from evaluate_registered_combat_models import read, save, sha
from train_combat_bc import torch, training_devices
from ppo_recurrent import durable_write
from fork_conditional_weapon_exploration_cuda import equal_tree
from train_combat_joint_aim_cuda import cuda_tree


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'registry', 'reference', 'reward', 'parent_protocol', 'protocol', 'out'):
        ap.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    a = ap.parse_args(); assert not a.out.exists() and not a.protocol.exists()
    repo = Path(__file__).resolve().parents[1]; pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest; return path

    seal = read(pin(a.source/'complete.json'))
    for name, key in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')):
        pin(a.source/name, seal[key])
    source_report = read(a.source/'report.json')
    assert source_report['version'] == 'combat_geometry_checkpoint_migration_v1'
    assert source_report['actor_value_std_adam_rng_history_exact'] and source_report['checkpoint_roundtrip_exact']
    assert not source_report['ppo_update_performed']
    reward = read(pin(a.reward)); assert reward['handoff_bootstrap'] == 'nonterminal_v1'
    index = read(pin(a.registry)); descriptors = [(f, read(pin(a.registry.parent/f))) for f in index['files']]
    assert len(descriptors) == 8
    reference_index = read(pin(a.reference))
    references = {e['id']:e for e in (read(pin(a.reference.parent/f)) for f in reference_index['files'])}
    for _, e in descriptors:
        ref = references[e['id']]; assert 'rules' in ref['modes'] and 'learned' in e['modes']
        x, y = copy.deepcopy(e), copy.deepcopy(ref)
        for item in (x,y):
            for key in ('modes','revision'): item.pop(key)
        assert x == y, 'Reference rules fixture differs'
    old_paths = {str((repo/e['recipe']['reward_config']).resolve()) for _, e in descriptors}
    assert len(old_paths) == 1
    previous_reward_path = pin(next(iter(old_paths)))
    previous_reward = read(previous_reward_path); stripped = copy.deepcopy(reward); stripped.pop('handoff_bootstrap')
    assert stripped == previous_reward, 'Reward coefficients or masks changed'
    parent_protocol = read(pin(a.parent_protocol)); assert parent_protocol['initializer_weights_sha256'] == seal['weights_sha256']
    training_devices()
    cp = torch.load(a.source/'checkpoint.pt', map_location='cuda', weights_only=True)
    assert cp['weights_sha256'] == seal['weights_sha256'] and cp['geometry_aim_training']['updates_completed'] == 0
    assert cp['config']['objective_reward_sha256'] == sha(previous_reward_path)
    # CPU deserialization/serialization only; every neural-state comparison uses CUDA.
    fork = torch.load(a.source/'checkpoint.pt', map_location='cpu', weights_only=True)
    fork['config']['objective_reward_sha256'] = sha(a.reward)
    fork['objective_forks'] = fork.get('objective_forks', []) + [dict(version='combat_nonterminal_handoff_objective_v1',
        from_reward_sha256=sha(previous_reward_path), to_reward_sha256=sha(a.reward), handoff_bootstrap='nonterminal_v1',
        source_checkpoint_sha256=seal['checkpoint_sha256'], historical_training_retained=True)]
    a.out.mkdir(parents=True); initial = a.out/'initializer'; initial.mkdir()
    shutil.copy2(a.source/'weights.json', initial/'weights.json')
    durable_write(initial/'checkpoint.pt', lambda f: torch.save(fork, f))
    loaded = torch.load(initial/'checkpoint.pt', map_location='cuda', weights_only=True)
    assert set(loaded) == set(cp) | {'objective_forks'}
    for key in cp:
        if key != 'config': assert equal_tree(cp[key], loaded[key]), 'State changed: '+key
    restored = copy.deepcopy(loaded['config']); restored['objective_reward_sha256'] = cp['config']['objective_reward_sha256']
    assert restored == cp['config']
    assert sha(initial/'weights.json') == seal['weights_sha256']
    relative_reward = a.reward.resolve().relative_to(repo).as_posix()
    for directory, modes in (('train-registry', ['learned']), ('comparison-registry', ['learned', 'rules'])):
        target = a.out/directory; target.mkdir()
        for name, original in descriptors:
            assert Path(name).name == name and 'learned' in original['modes'] and set(original['modes']) <= {'learned','rules'}
            updated = copy.deepcopy(original); updated['recipe']['reward_config'] = relative_reward
            if updated.get('generator') and 'seed_revision' not in updated['generator']:
                updated['generator']['seed_revision'] = original['revision']
            updated['revision'] += 1; updated['modes'] = modes
            check = copy.deepcopy(updated)
            check['recipe']['reward_config'] = original['recipe']['reward_config']
            check['revision'] = original['revision']; check['modes'] = original['modes']
            if check.get('generator'): check['generator'] = copy.deepcopy(original['generator'])
            assert check == original, 'Physical fixture or split changed'
            assert updated.get('generator',{}).get('seed_revision',updated['revision']) == original.get('generator',{}).get('seed_revision',original['revision'])
            save(target/name, updated)
        save(target/'index.json', index)
    protocol = copy.deepcopy(parent_protocol)
    protocol.update(handoff_bootstrap='nonterminal_v1', reward_sha256=sha(a.reward),
        objective_fork='combat_nonterminal_handoff_objective_v1', parent_protocol_sha256=sha(a.parent_protocol),
        initializer=str(initial.resolve()),
        provenance='New fixed experiment after a verified unfinished-handoff failure. Same policy/Adam/RNG initializer; only objective SHA changes. No old B/G endpoint selected or compared across objectives. Fresh ordinary TRAIN; diagnostic data excluded.')
    save(a.protocol, protocol)
    report = dict(state='complete', version='combat_geometry_objective_fork_v1', device='cuda', source=str(a.source.resolve()),
        weights_sha256=seal['weights_sha256'], actor_value_std_adam_rng_history_exact=True, checkpoint_roundtrip_exact=True,
        geometry_head_and_adam_exact=True, reward_coefficients_exact=True, training_performed=False, ppo_update_performed=False,
        source_reward_sha256=sha(previous_reward_path), reward_sha256=sha(a.reward), handoff_bootstrap='nonterminal_v1',
        promotion=False, inputs=pins, protocol_sha256=sha(a.protocol), scope='Explicit new objective; all historical neural state retained, no quality claim.')
    save(initial/'report.json', report)
    save(initial/'complete.json', dict(state='complete', weights_sha256=sha(initial/'weights.json'),
        checkpoint_sha256=sha(initial/'checkpoint.pt'), report_sha256=sha(initial/'report.json')))
    for path, digest in pins.items(): assert sha(Path(path)) == digest, path
    save(a.out/'report.json', dict(**report, physical_fixtures_and_seed_ranges_exact=True,
        registries={d:sha(a.out/d/'index.json') for d in ('train-registry', 'comparison-registry')}))
    save(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json'), protocol_sha256=sha(a.protocol)))
    print(dict(initializer=str(initial), reward_sha256=sha(a.reward), weights_unchanged=True, protocol=str(a.protocol)))


if __name__ == '__main__': main()
