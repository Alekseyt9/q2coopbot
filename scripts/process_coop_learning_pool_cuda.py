"""Prepare and CUDA-finalize every paired primary learner; no CPU NN path."""
import argparse
import json
import pathlib
import shutil
import subprocess
from process_combat_architecture_pool import read, save, sha
from finalize_combat_cuda_rollout import finalize


def run(command, log, cwd):
    with log.open('w', encoding='utf-8') as output:
        result = subprocess.run([str(v) for v in command], cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
    assert result.returncode == 0, str(log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.pool.resolve()
    repo = pathlib.Path(__file__).resolve().parent.parent
    report, manifest = read(root/'report.json'), read(root/'manifest.json')
    assert report['accepted'] == report['completed'] == manifest['episodes']
    model = pathlib.Path(manifest['model'])
    assert sha(model) == manifest['model_sha256']
    exporter = root/'q2combat-export.exe'
    if not exporter.exists():
        shutil.copy2(repo/'workspace/build/q2combat-paired-export.exe', exporter)
    assert sha(exporter) == sha(repo/'workspace/build/q2combat-paired-export.exe')
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
        dataset = directory/'dataset'
        if not dataset.exists():
            run([exporter,'--trace',directory/'PairLearner.jsonl','--paired-peer-trace',directory/'PairLeader.jsonl',
             '--paired-role','0','--paired-stop-on-death','--paired-experimental-reward',
             '--reward-config',repo/'scripts/scenarios/combat-reward-coop-navigation-v11.json',
             '--server-log',directory/'server.log','--client-name','PairLearner',
             '--reset-expectation',directory/'reset-role-0.json','--synchronous','--require-execution',
                 '--worker',f'paired-{seed}','--episode',f'paired-{seed}','--out',dataset], directory/'export.log', repo)
        native, cuda = directory/'native', directory/'cuda'
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
