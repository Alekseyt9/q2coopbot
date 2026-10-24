"""Balanced 24-condition fractional curriculum; no native games or neural fitting."""
import argparse
from collections import Counter
import os
from pathlib import Path
import re
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'reward', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    root = a.out.resolve()
    repo = Path(__file__).resolve().parents[1]
    assert not root.exists()
    assert read(a.config)['objective_reward_sha256'] == sha(a.reward)
    root.mkdir(parents=True)
    env = dict(os.environ, GOCACHE=str(repo/'workspace/build/gocache'), GOTOOLCHAIN='auto')

    def run(label, command):
        with (root/(label+'.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(list(map(str, command)), cwd=repo, env=env, stdout=log, stderr=log,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, str(root/(label+'.log'))

    planner = root/'q2episode.exe'
    run('build', ['go', 'build', '-o', planner, './cmd/q2episode'])
    episodes = []
    for loadout_index, loadout in enumerate(('blaster', 'machinegun', 'weapons')):
        for trio in (False, True):
            name = loadout+('-trio' if trio else '-pair')
            generated = root/(name+'.json')
            classes = ('monster_parasite,' if trio else '')+'monster_soldier_light,monster_gunner'
            run(name, [planner, '--root', repo, '--campaign-sites', 2, '--campaign-maps', 'base3,city1,jail1,ware2',
                '--campaign-group', classes, '--campaign-group-loadout', loadout, '--out', generated])
            for episode in read(generated)['episodes']:
                site = int(re.search(r'-site-(\d+)-', episode['id'])[1])-1
                if trio != ((site+loadout_index) % 2 == 0):
                    continue
                assert episode['generator']['kind'] == 'campaign-ground-projectile-group-v1'
                assert 'monster_soldier_light' in episode['monsters'] and 'monster_gunner' in episode['monsters']
                episodes.append(episode)
    assert len(episodes) == 24
    assert Counter(len(e['monsters']) for e in episodes) == {2:12, 3:12}
    assert Counter(e['recipe']['loadout'] for e in episodes) == {'blaster':8, 'machinegun':8, 'weapons':8}
    raw = root/'raw-registry'
    raw.mkdir()
    files = []
    for episode in episodes:
        files.append(episode['id']+'.json')
        save(raw/files[-1], episode)
    save(raw/'index.json', dict(version=1, files=files))
    names = [e['id'] for e in episodes]
    run('objective', [sys.executable, repo/'scripts/prepare_combat_objective_registry.py', '--registry', raw/'index.json',
        '--config', a.config.resolve(), '--reward', a.reward.resolve(), '--episodes', ','.join(names), '--out', root/'registry'])
    lookup = {(e['map'], int(re.search(r'-site-(\d+)-', e['id'])[1]), e['recipe']['loadout']):e for e in episodes}
    screen = []
    for map_name, site, other in (('base3',1,'blaster'), ('city1',1,'machinegun'), ('jail1',2,'blaster'), ('ware2',2,'machinegun')):
        screen += [lookup[map_name,site,'weapons'], lookup[map_name,3-site,other]]
    assert Counter(len(e['monsters']) for e in screen) == {2:4, 3:4}
    assert Counter(e['recipe']['loadout'] for e in screen) == {'weapons':4, 'blaster':2, 'machinegun':2}
    save(root/'prepared.json', dict(state='prepared', version='combat_enemy_projectile_curriculum_v1',
        registry=str(root/'registry/index.json'), registry_sha256=sha(root/'registry/index.json'),
        episodes=names, screen_episodes=[e['id'] for e in screen], reward_sha256=sha(a.reward),
        maps=['base3','city1','jail1','ware2'], sites_per_map=2, pairs=12, trios=12,
        loadouts=dict(Counter(e['recipe']['loadout'] for e in episodes)),
        script_sha256=sha(__file__), fitting=False, native_games=0,
        scope='Predeclared fractional design. Both arms use all24 conditions. Eight-case screen is balanced across maps/sites/composition; per-map/loadout interactions are not independently identified.'))
    print(dict(recipes=24, pairs=12, trios=12, loadouts=3, screen_cases=8))


if __name__ == '__main__':
    main()
