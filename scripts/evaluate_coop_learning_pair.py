"""Held-out paired comparison in the single-site coop pilot, 16 total slots."""
import argparse
import pathlib
import subprocess
import sys
from process_combat_architecture_pool import read, save, sha


def main():
    ap = argparse.ArgumentParser()
    for name in ('before', 'after', 'out'):
        ap.add_argument('--'+name, type=pathlib.Path, required=True)
    ap.add_argument('--control',type=pathlib.Path,help='Optional old feature/ownership baseline, on the same seeds')
    ap.add_argument('--episodes', type=int, default=32)
    ap.add_argument('--seed', type=int, default=186400)
    a = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parent.parent
    root = a.out.resolve()
    root.mkdir(exist_ok=False)
    client=root/'q2coopbot.exe'
    subprocess.run(['go','build','-o',str(client),'./cmd/q2coopbot'],cwd=repo,check=True)
    save(root/'protocol.json', dict(version='coop_pilot_paired_evaluation_v1',
        client_binary=str(client),client_sha256=sha(client),
        seed=a.seed, episodes=a.episodes, slots=16, timescale=2,
        before_sha256=sha(a.before), after_sha256=sha(a.after),control_sha256=sha(a.control) if a.control else None,
        scope='Held-out from update collection. One base1 Soldier/Blaster site and scripted moving peer; no general campaign or human coop acceptance.'))
    branches=[('before',a.before),('after',a.after)]
    if a.control: branches.append(('control',a.control))
    for name, model in branches:
        with (root/(name+'.log')).open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, str(repo/'scripts/run_coop_learning_pool.py'),
                '--out', str(root/name), '--model', str(model.resolve()),
                '--client-binary',str(client),
                '--episodes', str(a.episodes), '--slots', '16', '--seed', str(a.seed), '--port', '30200'],
                cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
            capture = read(root/name/'report.json')
            assert capture['accepted'] == a.episodes, 'Invalid captures; comparison incomplete'
            subprocess.run([sys.executable, str(repo/'scripts/process_coop_learning_pool_cuda.py'),
                '--pool', str(root/name)], cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(name+' complete', flush=True)
    before, after = [read(root/n/'processing-report.json') for n in ('before','after')]
    left = {r['seed']:r for r in before['results']}
    right = {r['seed']:r for r in after['results']}
    assert left.keys() == right.keys()
    assert read(root/'before/manifest.json')['client_sha256'] == read(root/'after/manifest.json')['client_sha256']
    comparison=dict(version='coop_pilot_paired_evaluation_v1',
        episodes=a.episodes, before={k:before[k] for k in ('joint_deaths','learner_kills','rows')},
        after={k:after[k] for k in ('joint_deaths','learner_kills','rows')},
        pairs=[dict(seed=s, before_death=left[s]['joint_death'], after_death=right[s]['joint_death'],
            before_kills=left[s]['learner_monster_kills'], after_kills=right[s]['learner_monster_kills'],
            before_damage=left[s]['learner_monster_damage'], after_damage=right[s]['learner_monster_damage']) for s in sorted(left)],
        scope='Single-site paired pilot only. Surviving horizon is not automatically victory. No promotion or general superiority claim.')
    if a.control:
        control=read(root/'control/processing-report.json')
        old={r['seed']:r for r in control['results']}
        assert old.keys()==left.keys()
        assert read(root/'control/manifest.json')['client_sha256']==read(root/'before/manifest.json')['client_sha256']
        comparison['control']={k:control[k] for k in ('joint_deaths','learner_kills','rows')}
        for pair in comparison['pairs']:
            r=old[pair['seed']]
            pair.update(control_death=r['joint_death'],control_kills=r['learner_monster_kills'],control_damage=r['learner_monster_damage'])
    save(root/'comparison.json',comparison)


if __name__ == '__main__':
    main()
