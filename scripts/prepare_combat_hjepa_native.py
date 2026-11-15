"""Replace visibility proxy with causal accumulated native outcome labels.

Server outcomes are training targets only. Inputs stay client observations.
No outcome, reward, seed or native monster HP is appended to feature vectors.
"""
import argparse
import hashlib
import json
import pathlib


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(episode, state):
    return (episode, *(state[k] for k in ('map', 'connection', 'spawncount', 'actor', 'life')))


def native_increment(outcome, ammo_spent):
    # One confirmed kill earns 2; losing 100 HP costs 3 plus death costs 5.
    # Visible presence/disappearance has no term. Tick has no positive reward.
    keys = ('monster_health_damage','monster_kills','received_health_damage','self_health_damage','teammate_health_damage','deaths')
    if any(not isinstance(outcome[k], int) or outcome[k] < 0 for k in keys) or ammo_spent < 0:
        raise ValueError('invalid native counts')
    return (.002 * outcome['monster_health_damage'] + 2 * outcome['monster_kills']
            - .03 * outcome['received_health_damage'] - .05 * outcome['self_health_damage']
            - .1 * outcome['teammate_health_damage'] - 5 * outcome['deaths']
            - .005 * ammo_spent)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--terminal-targets',action='store_true',help='also supervise next-state utility, including death cost')
    args = ap.parse_args()
    meta = json.loads(pathlib.Path(str(args.data) + '.meta.json').read_text())
    labels, sources = {}, dict(meta['sources_sha256'])
    for source, digest in meta['sources_sha256'].items():
        path = pathlib.Path(source)
        if sha(path) != digest:
            raise ValueError('changed steps: ' + source)
        outcome_path = path.with_name('server_outcomes.jsonl')
        report_path = path.with_name('report.json')
        report = json.loads(report_path.read_text())
        if not (report['command_proof']['accepted'] and report['synchronous_confirmed']
                and report['observed_reset_confirmed']):
            raise ValueError('unverified native commands')
        sources[str(outcome_path)] = sha(outcome_path)
        sources[str(report_path)] = sha(report_path)
        outcomes = {}
        for line in outcome_path.read_text().splitlines():
            out = json.loads(line)
            key = (out['worker'], out['episode'], out['step'])
            if key in outcomes:
                raise ValueError('duplicate native outcome')
            outcomes[key] = out
        totals, complete = {}, {}
        previous = None
        for line in path.read_text().splitlines():
            s = json.loads(line)
            o, n = s['observation'], s['next_observation']
            key = identity(s['episode'], o['identity'])
            frame_key = (digest,) + key + (o['identity']['frame'],)
            if frame_key in labels:
                raise ValueError('duplicate identity/frame across sources')
            totals.setdefault(key, 0.)
            complete.setdefault(key, True)
            if previous is not None and previous[0] == key and previous[1] + 1 != o['identity']['frame']:
                complete[key] = False
            out = outcomes.get((s['worker'], s['episode'], s['index']))
            native = s.get('native_step')
            valid = (out and out['available'] and out['version'] == 'server_step_effects_v1'
                     and native and native['spawncount'] == o['identity']['spawncount']
                     and native['actor'] == o['identity']['actor']
                     and native['end_frame'] == native['begin_frame'] + 1)
            if complete[key] and valid:
                # Label the state BEFORE this action, using only past native effects.
                before=totals[key]
                ammo = max(0, o['ammo'] - n['ammo']) if n and n['weapon'] == o['weapon'] else 0
                totals[key] += native_increment(out, ammo)
                behavior=float(s['owner']=='rules' and not s.get('interventions'))
                labels[frame_key]=(before,behavior,s['owner'],totals[key])
            else:
                complete[key] = False
            previous = key, o['identity']['frame']
    rows = []
    for line in args.data.read_text().splitlines():
        r = json.loads(line)
        episode, raw_id = r['episode'].split(':', 1)
        suffix = identity(episode, json.loads(raw_id)) + (r['frame'],)
        if r.get('source_sha256'):
            key=(r['source_sha256'],)+suffix
        else:
            matches=[key for key in labels if key[1:]==suffix]
            if len(matches)>1:raise ValueError('ambiguous source trajectory')
            key=matches[0] if matches else None
        if key not in labels:
            continue
        r['utility'], r['behavior_weight'], owner, after = labels[key]
        if args.terminal_targets:r['next_utility']=after
        # Prevent clips crossing control transfers even with adjacent frames.
        r['episode'] = episode+'@'+key[0][:16]+':'+raw_id+':'+owner
        rows.append(r)
    if not rows:
        raise ValueError('no native-supervised rows')
    with args.out.open('x', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, allow_nan=False) + '\n')
    meta.update(rows=len(rows), sources_sha256=sources, objective_version='native_progress_v3' if args.terminal_targets else 'native_progress_v2',
                data_sha256=sha(args.out), parent_data_sha256=sha(args.data),
                preparer_sha256=sha(pathlib.Path(__file__)),
                scope='Causal accumulated native damage/kills/deaths/ammo targets only. '
                      'Visible-enemy count excluded. Rules-only BC mask. Breaks on missing native proof/control transfers.')
    pathlib.Path(str(args.out) + '.meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(dict(rows=len(rows), objective=meta['objective_version'])))


if __name__ == '__main__':
    main()
