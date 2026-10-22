"""Join real MG rays to declared observed targets using exclusive command windows."""
import argparse
import collections
import json
import pathlib

from process_combat_architecture_pool import read, save, sha
from report_combat_machinegun_hits import scan
from report_combat_machinegun_aim import measure, summary


def main():
    ap = argparse.ArgumentParser()
    for name in ('evaluation', 'spec', 'out'):
        ap.add_argument('--'+name, type=pathlib.Path, required=True)
    args = ap.parse_args()
    root = args.evaluation.resolve()
    assert not args.out.exists()
    seal = read(root/'complete.json')
    spec = read(args.spec)
    assert seal['state'] == 'complete' and not seal['training_performed']
    assert sha(args.spec) == seal['spec_sha256']
    pool = read(root/'pool/report.json')
    assert pool['state'] == 'complete' and pool['source_unchanged']
    assert sha(root/'pool/report.json') == seal['pool_sha256']
    pins = {root/'complete.json': sha(root/'complete.json'), args.spec: sha(args.spec),
            root/'pool/report.json': seal['pool_sha256']}
    variants = {}
    for index, variant in enumerate(spec['variants']):
        if variant['mode'] != 'learned':
            continue
        label = variant['label']
        ep = root/('execution-'+label+'.json')
        execution = read(ep)
        assert execution['state'] == 'complete' and execution['pool_sha256'] == seal['pool_sha256']
        assert execution['selected_plan_index'] == index
        pins[ep] = sha(ep)
        pp = root/('processing-'+label)/'report.json'
        processed = read(pp)
        assert processed['device'] == 'cuda' and processed['state'] == 'complete'
        assert processed['pool_sha256'] == seal['pool_sha256'] and processed['selected_plan_index'] == index
        pins[pp] = sha(pp)
        bindings = {c['job']: c for c in execution['cases']}
        assert len(bindings) == len(processed['cases'])
        cases = []
        for job in pool['jobs']:
            if job['plan_index'] != index:
                continue
            binding = bindings[job['job']]
            cp = pathlib.Path(job['root'])/'report.json'
            capture = read(cp)
            assert sha(cp) == binding['capture_sha256']
            result = capture['results'][0]
            log = pathlib.Path(result['root'])/'server.log'
            assert sha(log) == binding['server_log_sha256']
            steps = root/('processing-'+label)/f"case-{job['job']:04d}"/'native/replay-0/steps.jsonl'
            for p in (cp, log, steps):
                pins[p] = sha(p)
            windows = {}
            with steps.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    s = json.loads(line)
                    obs = s['observation']
                    native = s.get('native_step')
                    dispatch = s.get('server_execution') or {}
                    if (not native or s['owner'] != 'provider' or obs['identity']['life'] != 1
                            or obs['health'] <= 0 or obs['identity']['frame'] < binding['network_release_frame']):
                        continue
                    if not dispatch.get('matched') or not dispatch.get('window_exclusive') or dispatch.get('recovery_commands') != 0:
                        continue
                    assert native['spawncount'] == binding['release_generation']
                    key = native['spawncount'], native['begin_frame'], native['actor'], native['sequence']
                    assert key not in windows
                    windows[key] = s
            with log.open(encoding='utf-8-sig') as stream:
                shots = scan(stream, set(windows))
            measured = []
            statuses = collections.Counter()
            eligible = 0
            for shot in shots:
                if not shot['eligible']:
                    continue
                eligible += 1
                value = measure(shot, windows[shot['window']])
                statuses[value['status']] += 1
                if value['status'] == 'measured':
                    measured.append(value)
            cases.append(dict(job=job['job'], episode=binding['episode'], seed=job['seed'],
                eligible_shots=eligible, statuses=dict(statuses), measurements=measured,
                server_log_sha256=pins[log], steps_sha256=pins[steps]))
        assert len(cases) == len(bindings)
        groups = {}
        for group in ('all', 'ordinary', 'recovery'):
            selected = [c for c in cases if group == 'all' or c['episode'].endswith('-recovery') == (group == 'recovery')]
            rows = [m for c in selected for m in c['measurements']]
            groups[group] = dict(cases=len(selected), eligible_shots=sum(c['eligible_shots'] for c in selected),
                statuses=dict(sum((collections.Counter(c['statuses']) for c in selected), collections.Counter())),
                summary=summary(rows),
                moving=summary([r for r in rows if r['native_horizontal_speed'] >= 10]),
                stationary=summary([r for r in rows if r['native_horizontal_speed'] < 10]),
                strong_recoil=summary([r for r in rows if abs(r['native_recoil_pitch']) > 6]))
        variants[label] = dict(cases=cases, groups=groups)
    for p, digest in pins.items():
        assert sha(p) == digest, str(p)
    save(args.out, dict(state='complete', version='registered_machinegun_selected_aim_v1',
        variants=variants, inputs=[dict(path=str(p), sha256=d) for p, d in pins.items()],
        script_sha256=sha(__file__), helper_sha256=sha(pathlib.Path(__file__).with_name('report_combat_machinegun_aim.py')),
        training_performed=False, promotion=False,
        direction_semantics='native_input_aim_before_integer_degree_vectoangles_and_spread',
        scope='Native MG input aim joined by exclusive command windows to the explicitly selected client-observed bbox. fire_lead subsequently truncates angles to integer degrees and applies spread; these angular errors are not final ballistic-ray accuracy. No nearest-enemy substitution, new neural execution, hidden-state inputs, counterfactual or fitting. Unavailable/undeclared targets remain separate statuses.'))
    print('Registered MG aim:', len(variants), 'variants')


if __name__ == '__main__':
    main()
