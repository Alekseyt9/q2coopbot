"""Audit actual MG outcomes against pre-command target availability; no NN execution."""
import argparse
import collections
import json
import pathlib

from process_combat_architecture_pool import read, save, sha
from report_combat_machinegun_hits import scan, summarize
from report_combat_machinegun_aim import measure


def confirmed_waste(shot):
    # Instantaneous native contact. Damageable contacts without health damage
    # remain ambiguous (armor/invulnerability); they are not penalized here.
    f = shot['fields']
    return bool(int(f['sky']) or float(f['fraction']) >= 1 or
                not int(f['damageable']) or int(f['health_before']) <= 0)


def summarize_bucket(shots):
    result = summarize(shots)
    creditable = [s for s in shots if s['creditable'] and confirmed_waste(s)]
    result['confirmed_waste'] = sum(confirmed_waste(s) for s in shots)
    result['creditable_confirmed_waste'] = len(creditable)
    result['proposed_reward_deltas'] = {
        str(p): len(creditable)*p for p in (-.005, -.01, -.02)
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=pathlib.Path, required=True)
    a = ap.parse_args()
    root = a.root.resolve()
    protocol = read(root/'protocol.json')
    proof = read(root/'recovery/verified-members.json')
    reference = read(root/'machinegun-hits.json')
    assert proof['state'] == 'complete'
    assert proof['protocol_sha256'] == reference['protocol_sha256'] == sha(root/'protocol.json')
    reference_sources = {s['server_log']: s for s in reference['sources']}
    groups = {}
    for entry in protocol['evaluations']:
        buckets = collections.defaultdict(list)
        episodes = []
        for path in sorted(pathlib.Path(entry['root']).glob('case-*/s-*/report.json')):
            assert proof['members'][str(path.parent)]['report_sha256'] == sha(path)
            report = read(path)
            assert report['capture_complete'] and report['provenance_valid']
            worker = pathlib.Path(report['results'][0]['root'])
            steps, log = worker/'dataset/steps.jsonl', worker/'server.log'
            source = reference_sources[str(log)]
            assert source['steps_sha256'] == sha(steps) and source['server_sha256'] == sha(log)
            windows = {}
            with steps.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    step = json.loads(line)
                    o, n = step['observation'], step.get('native_step')
                    if (n and o['identity']['life'] == 1 and o['identity']['frame'] > 100
                            and o['health'] > 0 and step.get('server_execution', {}).get('matched')):
                        key = (n['spawncount'], n['begin_frame'], n['actor'], n['sequence'])
                        assert key not in windows
                        windows[key] = step
            with log.open(encoding='utf-8-sig') as stream:
                shots = scan(stream, set(windows))
            local = collections.defaultdict(list)
            for shot in shots:
                if not shot['eligible']:
                    continue
                step = windows[shot['window']]
                execution = step['server_execution']
                shot['creditable'] = bool(execution.get('window_exclusive') and
                                          execution.get('recovery_commands', 0) == 0)
                status = measure(shot, step)['status']
                local[status].append(shot)
                buckets[status].append(shot)
                buckets['all'].append(shot)
            episodes.append(dict(seed=report['results'][0]['seed'], report=str(path),
                steps_sha256=source['steps_sha256'], server_sha256=source['server_sha256'],
                strata={k: summarize_bucket(v) for k, v in local.items()}))
        assert len(episodes) == protocol['episodes_per_model']
        name = entry['model']+'-'+entry['label']
        assert summarize_bucket(buckets['all'])['counts'] == reference['groups'][name]['counts']
        groups[name] = dict(episodes=episodes,
                           strata={k: summarize_bucket(v) for k, v in buckets.items()})
        print(json.dumps({name: groups[name]['strata']}), flush=True)
    save(root/'machinegun-waste-audit.json', dict(version='native_mg_waste_audit_v1',
        state='complete', protocol_sha256=sha(root/'protocol.json'),
        member_proof_sha256=sha(root/'recovery/verified-members.json'),
        reference_sha256=sha(root/'machinegun-hits.json'), groups=groups,
        scope='Actual first-life alive MG fires, not attack frames. Target status from client '
              'pre-command observation; outcomes from native synchronous shot/damage. '
              'Credit proposals require exclusive unrecovered command windows. Sky, geometry, '
              'no contact and corpse contacts are confirmed waste; other damageable contacts '
              'without health loss remain ambiguous. Missing visible target alone is not a miss. '
              'No stored rewards, weights, policy inputs or runtime actions changed.'))


if __name__ == '__main__':
    main()
