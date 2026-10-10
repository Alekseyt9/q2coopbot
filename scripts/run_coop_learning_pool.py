"""Work-conserving 16-slot paired collection pilot; no training/CPU NN checks."""
import argparse
import concurrent.futures
import json
import pathlib
import os
import queue
import subprocess
import time
from process_combat_architecture_pool import read, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--model', type=pathlib.Path, required=True)
    parser.add_argument('--client-binary',type=pathlib.Path,help='Reuse one frozen client for paired model comparisons')
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--slots', type=int, default=16)
    parser.add_argument('--seed', type=int, default=186240)
    parser.add_argument('--port', type=int, default=30000)
    args = parser.parse_args()
    assert 1 <= args.slots <= 16 and args.episodes > 0
    root, model = args.out.resolve(), args.model.resolve()
    assert not root.exists(), 'Fresh pool output required'
    root.mkdir()
    repo = pathlib.Path(__file__).resolve().parent.parent
    client = root/'q2coopbot.exe'
    protocol_path=root.parent/'protocol.json'
    if args.client_binary is None and protocol_path.exists():
        protocol=read(protocol_path)
        if protocol.get('version')=='coop_pilot_paired_evaluation_v1' and protocol.get('client_binary'):
            args.client_binary=pathlib.Path(protocol['client_binary'])
            assert sha(args.client_binary)==protocol['client_sha256'], 'Frozen evaluation client changed'
    if args.client_binary:
        os.link(args.client_binary.resolve(),client)
    else:
        subprocess.run(['go', 'build', '-o', str(client), './cmd/q2coopbot'], cwd=repo, check=True)
    manifest = dict(version='coop_learning_collection_pool_v1', slots=args.slots,
                    timescale=2, episodes=args.episodes, first_seed=args.seed,
                    model=str(model), model_sha256=sha(model), client_sha256=sha(client),
                    scope='One base1 Soldier/Blaster geometry, seeded moving peer; primary policy stochastic. Pilot collection, not architecture evaluation.')
    save(root/'manifest.json', manifest)
    leases = queue.Queue()
    for slot in range(args.slots): leases.put(slot)
    started = time.time()

    def collect(index):
        slot = leases.get()
        seed = args.seed+index
        case = root/f'seed-{seed}'
        try:
            command = ['pwsh', '-NoProfile', '-File', str(repo/'scripts/run_coop_lockstep_smoke.ps1'),
                       '-OutputRoot', str(case), '-Port', str(args.port+slot), '-Seed', str(seed),
                       '-ReleaseGameFrame', '100', '-CombatFixture', '-MovingPeer',
                       '-StopOnParticipantDeath', '-Model', str(model), '-ClientBinary', str(client)]
            with (root/f'seed-{seed}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            report = read(case/'report.json') if (case/'report.json').exists() else {}
            return dict(seed=seed, slot=slot, root=str(case), returncode=result.returncode,
                        accepted=result.returncode == 0 and report.get('accepted', False),
                        joint_death=report.get('live_joint_stop_verified', False),
                        stop_frame=report.get('stop_frame'), pairs=report.get('completed_pairs'))
        finally:
            leases.put(slot)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.slots) as executor:
        futures = [executor.submit(collect, i) for i in range(args.episodes)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            save(root/'progress.json', dict(completed=len(results), total=args.episodes, results=results))
            print(json.dumps(result), flush=True)
    assert sha(model) == manifest['model_sha256'] and sha(client) == manifest['client_sha256']
    save(root/'report.json', dict(version=manifest['version'], completed=len(results),
                                  accepted=sum(r['accepted'] for r in results), elapsed_seconds=time.time()-started,
                                  results=sorted(results, key=lambda r:r['seed']), training_performed=False))


if __name__ == '__main__': main()
