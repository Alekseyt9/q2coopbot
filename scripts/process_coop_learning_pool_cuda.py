"""Prepare and CUDA-finalize every paired primary learner; no CPU NN path."""
import argparse
import json
import pathlib
import subprocess
from process_combat_architecture_pool import read, save, sha
from finalize_combat_cuda_rollout import finalize


def verify_memory_export(trace, dataset):
    """Do not let a legacy exporter erase inputs hidden by zero initial weights."""
    identities=('map','connection','spawncount','actor','life','frame')
    own={}
    for line in trace.read_text().splitlines():
        o=json.loads(line).get('combat_policy',{}).get('observation')
        if o is not None:
            own[tuple(o['identity'][k] for k in identities)]=o
    for line in (dataset/'steps.jsonl').read_text().splitlines():
        s=json.loads(line)
        for name in ('observation','next_observation'):
            o=s.get(name)
            if o is None: continue
            raw=own[tuple(o['identity'][k] for k in identities)]
            assert o.get('remembered_threats',[])==raw.get('remembered_threats',[]), 'Exporter changed/dropped observed threat memory'


def run(command, log, cwd):
    with log.open('w', encoding='utf-8') as output:
        result = subprocess.run([str(v) for v in command], cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
    assert result.returncode == 0, str(log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=pathlib.Path, required=True)
    parser.add_argument('--revision',default='',help='Fresh export revision; preserve existing sealed datasets')
    args = parser.parse_args()
    assert not args.revision or args.revision.isalnum(), 'Revision must be alphanumeric'
    suffix = '-'+args.revision if args.revision else ''
    root = args.pool.resolve()
    repo = pathlib.Path(__file__).resolve().parent.parent
    report, manifest = read(root/'report.json'), read(root/'manifest.json')
    assert report['accepted'] == report['completed'] == manifest['episodes']
    model = pathlib.Path(manifest['model'])
    assert sha(model) == manifest['model_sha256']
    model_feature_version=read(model)['feature_version']
    exporter = root/('q2combat-export'+suffix+'.exe')
    if not exporter.exists():
        # Build the frozen exporter from current source, rather than trusting
        # a differently named global binary that can silently drop new fields.
        run(['go','build','-o',exporter,'./cmd/q2combat-export'],root/('exporter-build'+suffix+'.log'),repo)
    adapter = repo/'workspace/build/q2paired-data.exe'
    results = []
    for case in report['results']:
        directory = pathlib.Path(case['root'])
        seed = case['seed']
        reset = dict(version='observed_fixture_reset_v1', map='base1', seed=seed,
                     position=[-48,32,24.125], health=100, armor=0, weapon='Blaster', ammo=0,
                     enemy_class='monster_soldier', enemy_position=[200,-224,24.125],
                     inventory=[dict(name='Blaster',count=1),dict(name='Shotgun',count=1),dict(name='Shells',count=20)])
        save(directory/'reset-role-0.json', reset)
        dataset = directory/('dataset'+suffix)
        if not dataset.exists():
            run([exporter,'--trace',directory/'PairLearner.jsonl','--paired-peer-trace',directory/'PairLeader.jsonl',
             '--paired-role','0','--paired-stop-on-death','--paired-experimental-reward',
             '--reward-config',repo/'scripts/scenarios/combat-reward-coop-navigation-v11.json',
             '--server-log',directory/'server.log','--client-name','PairLearner',
             '--reset-expectation',directory/'reset-role-0.json','--synchronous','--require-execution',
                 '--worker',f'paired-{seed}','--episode',f'paired-{seed}','--out',dataset], directory/'export.log', repo)
        native, cuda = directory/('native'+suffix), directory/('cuda'+suffix)
        if model_feature_version=='combat_features_v9':
            verify_memory_export(directory/'PairLearner.jsonl',dataset)
        if cuda.exists():
            meta = read(cuda/'report.json')
            assert meta['paired_training_ready'] and meta['model_sha256'] == manifest['model_sha256']
            assert sha(cuda/'rollout.jsonl') == meta['rollout_sha256']
            for path, digest in meta['source_sha256'].items(): assert sha(path) == digest
            verification = read(cuda/'cuda-verification.json')
        else:
            if native.exists() and not (native/'report.json').exists():
                native = directory/'native-retry-1'
            if not native.exists():
                run([adapter,'--case',directory,'--dataset',dataset,'--model',model,'--exporter',exporter,
                     '--out',native,'--allow-horizon'], directory/'prepare-retry.log', repo)
            verification = finalize(model, native, cuda)
        meta = read(cuda/'report.json')
        if meta['feature_version']=='combat_features_v9':
            rollout_rows=[json.loads(line) for line in (cuda/'rollout.jsonl').read_text().splitlines()]
            assert any(any(row['features'][881:]) for row in rollout_rows), 'V9 memory missing from exported learning features'
        outcomes = [json.loads(line) for line in (dataset/'server_outcomes.jsonl').read_text().splitlines()]
        results.append(dict(seed=seed, root=str(directory), rows=meta['rows'], terminals=meta['terminals'],
                            joint_death=case['joint_death'], learner_monster_kills=sum(o['monster_kills'] for o in outcomes),
                            learner_monster_damage=sum(o['monster_health_damage'] for o in outcomes),
                            score=read(dataset/'report.json')['reward_sum'], cuda=str(cuda), verification=verification))
        save(root/'processing-progress.json', dict(completed=len(results), total=len(report['results']), results=results))
        print(f"CUDA verified seed={seed} rows={meta['rows']} joint_death={case['joint_death']}", flush=True)
    save(root/'processing-report.json', dict(version='coop_learning_cuda_pool_v1', completed=len(results),
                                            rows=sum(r['rows'] for r in results),
                                            joint_deaths=sum(r['joint_death'] for r in results),
                                            learner_kills=sum(r['learner_monster_kills'] for r in results),
                                            results=results, training_performed=False))


if __name__ == '__main__': main()
