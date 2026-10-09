"""Make one model-independent draft registry with both weapon loadouts."""
import argparse
import copy
import pathlib
import subprocess

from process_combat_architecture_pool import read, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Use a fresh output root'
    search_path = args.search / 'report.json'
    search = read(search_path)
    assert search['state'] == 'static_search_complete' and search['selected']
    planner = pathlib.Path(search['planner'])
    assert sha(planner) == search['planner_sha256']
    registry = args.out / 'registry'
    registry.mkdir(parents=True)
    files, provenance, episodes = [], [], []
    for index, candidate in enumerate(search['selected']):
        frozen = candidate['plans'][0]
        assert sha(frozen['path']) == frozen['sha256']
        plan = read(frozen['path'])
        assert sha(candidate['registry']) == plan['registry_sha256']
        parent = plan['tasks'][0]['episode']
        source_file = pathlib.Path(candidate['registry']).parent / (parent['id'] + '.json')
        assert sha(source_file) == plan['tasks'][0]['episode_sha256']
        for loadout in ('blaster', 'machinegun'):
            episode = copy.deepcopy(parent)
            episode['id'] = parent['id'].replace('-blaster', '-' + loadout)
            episode['title'] = parent['title'] + ' / ' + loadout
            episode['recipe']['loadout'] = loadout
            if loadout == 'machinegun':
                for split in episode['splits'].values():
                    split['start'] += 2000000
            filename = episode['id'] + '.json'
            save(registry / filename, episode)
            files.append(filename)
            episodes.append(episode['id'])
            provenance.append(dict(id=episode['id'], parent=source_file.as_posix(), parent_sha256=sha(source_file),
                                   recipe_sha256=sha(registry / filename), radius=candidate['radius']))
    save(registry / 'index.json', dict(version=1, files=files))
    plans = []
    for split in ('train', 'validation'):
        path = args.out / (split + '-rules-plan.json')
        result = subprocess.run([str(planner), '--registry', str(registry / 'index.json'),
            '--episodes', ','.join(episodes), '--split', split, '--mode', 'rules', '--count', '4',
            '--out', str(path), '--artifacts', str(args.out / (split + '-rules-native'))],
            capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        verified = subprocess.run([str(planner), '--verify-plan', str(path)],
                                  capture_output=True, text=True, timeout=60)
        assert verified.returncode == 0, verified.stderr
        plan = read(path)
        distances = []
        for task in plan['tasks']:
            for instance in task['instances']:
                import math
                distance = math.dist(instance['player'], instance['monsters'][0]['position'])
                assert distance > 512
                distances.append(distance)
        plans.append(dict(split=split, path=str(path), sha256=sha(path), episodes=len(distances),
                          min_distance=min(distances), max_distance=max(distances)))
    save(args.out / 'report.json', dict(state='static_preflight_passed_native_pending',
        search_sha256=sha(search_path), registry=str(registry / 'index.json'), registry_sha256=sha(registry / 'index.json'),
        planner_sha256=sha(planner), recipes=provenance, plans=plans,
        scope='Separate draft registry for universal model training and comparison, two real campaign sites with Blaster/Machinegun. Existing BSP planner regenerated all train/validation plans. No final test, live servers, model execution or training. Native acceptance pending; canonical registry unchanged. Parent wall rays retained as parent-location metadata only.'))
    print(f'Prepared {len(episodes)} recipes, {sum(p["episodes"] for p in plans)} statically verified conditions; native pending', flush=True)


if __name__ == '__main__':
    main()
