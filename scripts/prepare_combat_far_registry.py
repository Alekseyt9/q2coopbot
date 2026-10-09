"""Search draft long-range starts using the existing BSP/native plan validator."""
import argparse
import copy
import json
import math
import pathlib
import subprocess

from process_combat_architecture_pool import read, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--planner', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Use a fresh output root; failed candidates are preserved'
    args.out.mkdir(parents=True)
    planner = args.planner.resolve()
    attempts, selected = [], []
    sources = sorted(pathlib.Path('scripts/scenarios/combat-training').glob('campaign-*-site-*-blaster.json'))
    for site_index, source in enumerate(sources):
        parent = read(source)
        primary = parent['generator']['distributions']['train']['primary']
        center = [(a + b) / 2 for a, b in zip(primary['min'], primary['max'])]
        found = False
        for radius in (640, 768):
            for angle in range(8):
                trial = args.out / 'candidates' / f'site-{site_index:02d}-r{radius}-a{angle}'
                registry = trial / 'registry'
                registry.mkdir(parents=True)
                episode = copy.deepcopy(parent)
                episode['id'] = parent['id'].replace('-blaster', '-far-blaster')
                episode['title'] += ' / draft far start'
                episode['geometry'] = 'Original campaign BSP; new far candidate requires BSP preflight and subsequent native smoke. Site wall_distances describe the original parent player location, not this new start.'
                for split_index, split in enumerate(('train', 'validation', 'test', 'confirmation')):
                    episode['splits'][split] = dict(start=4000000 + site_index * 100000 + split_index * 10000, count=4096)
                    distribution = episode['generator']['distributions'][split]
                    position = [round((center[0] + radius * math.cos(angle * math.pi / 4)) * 8) / 8,
                                round((center[1] + radius * math.sin(angle * math.pi / 4)) * 8) / 8,
                                center[2]]
                    distribution['player'] = dict(min=[position[0] - 4, position[1] - 4, position[2]],
                                                  max=[position[0] + 4, position[1] + 4, position[2]])
                filename = episode['id'] + '.json'
                save(registry / filename, episode)
                save(registry / 'index.json', dict(version=1, files=[filename]))
                record = dict(source=str(source), source_sha256=sha(source), radius=radius,
                              angle_index=angle, registry=str(registry / 'index.json'), plans=[])
                passed = True
                for split in ('train', 'validation'):
                    plan_path = trial / (split + '-plan.json')
                    command = [str(planner), '--registry', str(registry / 'index.json'),
                               '--episodes', episode['id'], '--split', split, '--mode', 'rules',
                               '--count', '4', '--out', str(plan_path), '--artifacts', str(trial / (split + '-native'))]
                    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                    if result.returncode:
                        record.update(state='rejected', error=result.stderr.strip())
                        passed = False
                        break
                    verified = subprocess.run([str(planner), '--verify-plan', str(plan_path)],
                                              capture_output=True, text=True, timeout=60)
                    assert verified.returncode == 0, verified.stderr
                    plan = read(plan_path)
                    instances = plan['tasks'][0]['instances']
                    distances = [math.dist(v['player'], v['monsters'][0]['position']) for v in instances]
                    assert len(distances) == 4 and min(distances) > 512
                    record['plans'].append(dict(split=split, path=str(plan_path), sha256=sha(plan_path),
                                                distances=distances))
                attempts.append(record)
                if passed:
                    record['state'] = 'static_preflight_passed_native_pending'
                    selected.append(record)
                    found = True
                save(args.out / 'progress.json', dict(attempts=len(attempts), selected=len(selected), current_source=str(source)))
                if found:
                    break
            if found:
                break
        print(json.dumps(dict(source=str(source), far_preflight_found=found)), flush=True)
    report = dict(state='static_search_complete', planner=str(planner), planner_sha256=sha(planner),
                  selected=selected, attempts=attempts,
                  scope='Draft generator conditions only. Four train and four validation seeds per accepted site, regenerated and statically verified by existing Go BSP planner. No servers, model execution or training. Final-test/confirmation seeds untouched. Native settling, observation visibility, mover state and live suitability remain unverified. Canonical registry unchanged; parent wall rays are not new-position rays.')
    save(args.out / 'report.json', report)
    print(json.dumps(dict(attempts=len(attempts), selected=len(selected))), flush=True)


if __name__ == '__main__':
    main()
