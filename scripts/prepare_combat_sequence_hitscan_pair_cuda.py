"""Prepare equal CUDA PPO starts for a native MG miss-cost ablation.

This does not train or launch servers. Both arms reset Adam and critic output
identically; the control and candidate require the same native shot evidence.
"""
import argparse
import copy
import pathlib
import subprocess
import sys

from process_combat_architecture_pool import read, save, sha


def main():
    ap = argparse.ArgumentParser()
    for name in ('parent', 'registry', 'reference-pilot', 'probe-data', 'out'):
        ap.add_argument('--' + name, type=pathlib.Path, required=True)
    a = ap.parse_args()
    import torch
    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    repo = pathlib.Path(__file__).resolve().parents[1]
    a.out = a.out.resolve()
    assert not a.out.exists()
    parent = a.parent.resolve()
    closed = read(parent / 'complete.json')
    for file, key in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')):
        assert sha(parent / file) == closed[key]
    cp = torch.load(parent / 'checkpoint.pt', map_location='cuda', weights_only=True)
    reference = read(a.reference_pilot)
    episodes = reference['rounds'][0]['episodes']
    assert len(episodes) == 8 and len(set(episodes)) == 8
    objectives = [p for p in (repo / 'scripts/scenarios').glob('*reward*.json')
                  if sha(p) == cp['config']['objective_reward_sha256']]
    assert len(objectives) == 1, 'Resolve the actual pinned parent objective, never a similarly named file'
    legacy = objectives[0]
    assert read(legacy)['version'] == 'combat_reward_v12'
    assert cp['config']['actor_steps'] == 10 and cp['config']['value_steps'] == 40
    a.out.mkdir(parents=True)

    def run(label, arguments):
        with (a.out / (label + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, *map(str, arguments)], cwd=repo,
                stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, str(a.out / (label + '.log'))

    arms = []
    for label, cost, offset, port in (('control', 0., 440, 45900), ('miss-cost', -.02, 444, 45940)):
        reward = copy.deepcopy(read(legacy)); reward.update(version='combat_reward_v13', machinegun_miss=cost)
        suffix = 'control' if label == 'control' else 'cost'
        reward_path = repo / ('scripts/scenarios/combat-reward-sequence-hitscan-' + suffix + '-v13.json')
        assert read(reward_path) == reward, 'Canonical registered recipe differs from the one-factor fork'
        config = copy.deepcopy(cp['config']); config['objective_reward_sha256'] = sha(reward_path)
        config_path = a.out / (label + '-config.json'); save(config_path, config)
        source = a.out / (label + '-source')
        run(label + '-fork', [repo / 'scripts/fork_combat_architecture_objective_cuda.py',
            '--parent', parent, '--config', config_path, '--reward', reward_path,
            '--probe-data', a.probe_data.resolve(), '--out', source])
        save(source / 'complete.json', dict(version='combat_update_complete_v1',
            weights_sha256=sha(source / 'weights.json'), checkpoint_sha256=sha(source / 'checkpoint.pt'),
            report_sha256=sha(source / 'report.json')))
        registry = a.out / (label + '-registry')
        run(label + '-registry', [repo / 'scripts/prepare_combat_objective_registry.py',
            '--registry', a.registry.resolve(), '--config', config_path, '--reward', reward_path,
            '--episodes', ','.join(episodes), '--out', registry])
        pilot = a.out / (label + '-pilot'); pilot.mkdir()
        corpus, fit = a.out / (label + '-corpus'), a.out / (label + '-fit')
        inputs = [parent / 'complete.json', source / 'complete.json', config_path, reward_path,
                  registry / 'index.json', pathlib.Path(reference['anchor']), pathlib.Path(reference['bank'])]
        inputs += [registry / file for file in read(registry / 'index.json')['files']]
        protocol = dict(version='combat_broad_ppo_pilot_v1', source=str(source),
            source_complete_sha256=sha(source / 'complete.json'), registry=str(registry / 'index.json'),
            anchor=reference['anchor'], bank=reference['bank'], rounds=[dict(round=1, episodes=episodes,
                cases=32, train_offset_minimum=offset, port=port, corpus=str(corpus), fit=str(fit))],
            native_training_budget=32, unique_families=8, actor_steps_per_round=10,
            value_steps_per_round=40, sequence_retention_weight=1., cuda_only=True,
            max_instances=16, timescale=2, promotion=False, selection_on_intermediate_validation=False,
            objective_changed=False, inputs=[dict(path=str(p.resolve()), sha256=sha(p)) for p in inputs],
            scope='One fixed own-policy round. Objective stays fixed within this arm; pair changes only actual MG miss cost. Fresh seeds per arm; held-out evaluation is paired separately.')
        save(pilot / 'protocol.json', protocol)
        arms.append(dict(label=label, source=str(source), protocol=str(pilot / 'protocol.json'),
            fit=str(fit), corpus=str(corpus), reward_sha256=sha(reward_path), machinegun_miss=cost))
    left, right = [torch.load(pathlib.Path(arm['source']) / 'checkpoint.pt', map_location='cuda', weights_only=True) for arm in arms]

    def exact(x, y):
        if torch.is_tensor(x):
            assert x.is_cuda and y.is_cuda and torch.equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for key in x: exact(x[key], y[key])
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for v, w in zip(x, y): exact(v, w)
        else: assert x == y

    assert {k:v for k,v in left['config'].items() if k != 'objective_reward_sha256'} == {k:v for k,v in right['config'].items() if k != 'objective_reward_sha256'}
    for key in left:
        if key != 'config': exact(left[key], right[key])
    assert sha(pathlib.Path(arms[0]['source']) / 'weights.json') == sha(pathlib.Path(arms[1]['source']) / 'weights.json')
    assert not left['actor_optimizer']['state'] and not left['value_optimizer']['state']
    save(a.out / 'complete.json', dict(state='complete', version='combat_sequence_hitscan_pair_v1',
        device='cuda', gpu=torch.cuda.get_device_name(), parent=str(parent), parent_weights_sha256=closed['weights_sha256'],
        arms=arms, native_training_budget=64, equal_initial_actor_value_std_optimizers_rng=True,
        zero_critic_output=True, training_performed=False, promotion=False,
        independent_evaluation_required=True, script_sha256=sha(__file__)))
    print('Equal CUDA objective pair ready:', a.out)


if __name__ == '__main__': main()
