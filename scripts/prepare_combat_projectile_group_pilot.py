"""Prepare exact light-soldier/gunner TRAIN conditions; no fitting or quality claim."""
import argparse
from pathlib import Path
import os
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'config', 'reward', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.out.resolve()
    assert not root.exists()
    source, config, reward = (p.resolve() for p in (args.source, args.config, args.reward))
    closed = read(source / 'complete.json')
    assert closed['state'] == 'complete'
    assert sha(source / 'weights.json') == closed['weights_sha256']
    assert sha(source / 'checkpoint.pt') == closed['checkpoint_sha256']
    assert read(config)['objective_reward_sha256'] == sha(reward)
    root.mkdir(parents=True)
    env = dict(os.environ, GOCACHE=str(repo / 'workspace/build/gocache'), GOTOOLCHAIN='auto')

    def run(label, command):
        with (root / (label + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(list(map(str, command)), cwd=repo, env=env, stdout=log, stderr=log,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, str(root / (label + '.log'))

    planner = root / 'q2episode.exe'
    run('build', ['go', 'build', '-o', planner, './cmd/q2episode'])
    generated = root / 'generated.json'
    run('generate', [planner, '--root', repo, '--campaign-sites', 2,
        '--campaign-maps', 'base3,city1,jail1,ware2', '--campaign-group',
        'monster_parasite,monster_soldier_light,monster_gunner', '--campaign-group-loadout', 'weapons', '--out', generated])
    episodes = read(generated)['episodes']
    assert len(episodes) == 8
    raw = root / 'raw-registry'
    raw.mkdir()
    files, names = [], []
    for episode in episodes:
        assert episode['monsters'] == ['monster_parasite', 'monster_soldier_light', 'monster_gunner']
        assert episode['generator']['kind'] == 'campaign-ground-projectile-group-v1'
        assert episode['splits']['train']['start'] >= 1400000000
        name = episode['id']
        names.append(name)
        files.append(name + '.json')
        save(raw / files[-1], episode)
    save(raw / 'index.json', dict(version=1, files=files))
    registry = root / 'registry'
    run('objective', [sys.executable, repo / 'scripts/prepare_combat_objective_registry.py', '--registry', raw / 'index.json',
        '--config', config, '--reward', reward, '--episodes', ','.join(names), '--out', registry])
    save(root / 'history.json', dict(historical_plans=[]))
    report = dict(state='prepared', version='combat_exact_projectile_group_pilot_v1', source=str(source),
        source_weights_sha256=closed['weights_sha256'], registry=str(registry / 'index.json'),
        registry_sha256=sha(registry / 'index.json'), episodes=names, maps=['base3', 'city1', 'jail1', 'ware2'],
        sites_per_map=2, train_count_per_recipe=2, native_budget=16, cuda_budget=16,
        max_instances=16, timescale=2, reward_sha256=sha(reward), fitting=False, promotion=False,
        pre_fit_acceptance='Exact native light soldier20HP/gunner/parasite setup; no infighting; actual observed enemy bolts and grenades with motion estimates. Pilot is TRAIN-only and not a quality comparison.',
        inputs={str(p): sha(p) for p in (Path(__file__), config, reward, generated, source / 'complete.json')})
    save(root / 'prepared.json', report)
    print(dict(recipes=8, native_budget=16, registry=report['registry']))


if __name__ == '__main__':
    main()
