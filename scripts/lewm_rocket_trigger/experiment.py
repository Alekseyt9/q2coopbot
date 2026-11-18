"""Fit a small family of optional trigger policies by paired native save replays."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from build import CANDIDATES, REPO


BANK = REPO / 'workspace/artifacts'
V1 = BANK / 'lewm-weapon-saves-v1'
V2 = BANK / 'lewm-weapon-damage-v2'
TOOL = BANK / 'fragment-weapon-roots-20261117-034744/q2checkpoint.exe'
WINDOW_END = 41


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2))


def root_obs(trace, frame):
    rows = subprocess.check_output([V1/'features.exe', trace, str(frame), str(frame)])
    return json.loads(rows)


class InitialObservationMismatch(ValueError):
    pass


def blast_hp(run):
    total=0
    for line in run['events']:
        event=dict(re.findall(r'(\w+)=([^ ]+)',line))
        if (event['attacker']==event['target'] and event['target_class']=='player'
                and (int(event['mod']) & ~0x8000000) in (7,9,13,16,24)):
            total+=max(0,int(event['health_before']))-max(0,int(event['health_after']))
    return total


def plan():
    candidates = read(V1/'plan-verified.json')['roots'] + read(V2/'fresh-plan.json')
    selected = []
    for r in candidates:
        obs = root_obs(Path(r['source'])/'bot.jsonl', r['frame'])
        if obs['owner'] == 'provider' and obs['available'][7]:
            selected.append(dict(r, split=r.get('split', 'fresh'),
                                 feature_hash=hashlib.sha256(json.dumps(obs['features']).encode()).hexdigest()))
    if len(selected) != len({r['seed'] for r in selected}):
        raise ValueError('Duplicate world')
    return selected


def summarize(directory, r, fixture_rocket=True, window_end=WINDOW_END, score_objective=False):
    report = read(directory/'report.json')
    assert all(report[k] for k in ('accepted', 'native_rng_restored', 'native_phase_restored'))
    assert hashlib.sha256((directory/'checkpoint/manifest.json').read_bytes()).hexdigest() == r['manifest_hash']
    restored = read(directory/'control.json.restored.json')['combat']
    root = restored['decision_root']
    assert root['frame'] == 1 and restored['source_root']['frame'] == r['frame']
    obs = root_obs(directory/'bot.jsonl', 1)
    if hashlib.sha256(json.dumps(obs['features']).encode()).hexdigest() != r['feature_hash']:
        src = root_obs(Path(r['source'])/'bot.jsonl',r['frame'])['features']
        differences = [(i,a,b) for i,(a,b) in enumerate(zip(src,obs['features'])) if a != b]
        raise InitialObservationMismatch(f"seed={r['seed']} differing features={differences[:8]}")
    assert obs['owner'] == 'provider'
    if fixture_rocket:
        assert obs['proposed']['weapon'] == 'Rocket Launcher'
    end = window_end
    own = monster = rocket_monster = other = 0
    events = []
    clear_frames = set()
    objective_frames = set()
    for line in (directory/'server.log').read_text(errors='replace').splitlines():
        if score_objective and line.startswith('sv_test_objective '):
            obj = dict(re.findall(r'(\w+)=([^ ]+)',line))
            f = int(obj['server_frame'])
            if int(obj['spawncount']) == root['spawncount'] and 1 < f <= end:
                assert obj['map'] == r['map'] and f not in objective_frames
                objective_frames.add(f)
                if int(obj['alive']) == 0 and int(obj['pending']) == 0:
                    clear_frames.add(f)
            continue
        if not line.startswith('sv_test_damage ') or ' g_test_damage version=1 ' not in line:
            continue
        e = dict(re.findall(r'(\w+)=([^ ]+)', line))
        frame = int(e['server_frame'])
        if int(e['spawncount']) != root['spawncount'] or not 1 < frame <= end:
            continue
        hp = max(0, int(e['health_before'])) - max(0, int(e['health_after']))
        mod = int(e['mod']) & ~0x8000000
        attacker = int(e['attacker']) == root['actor']
        if attacker and e['target_class'].startswith('monster_'):
            monster += hp
            if mod in (8, 9):
                rocket_monster += hp
        if int(e['target']) == root['actor']:
            if attacker and mod == 9:
                own += hp
            else:
                other += hp
            if int(e['health_after']) <= 0:
                end = frame
        if hp and (attacker or int(e['target']) == root['actor']):
            events.append(line)
    previous = None
    spent = commands = last = 0
    first_clear = None
    with (directory/'bot.jsonl').open() as trace:
        for line in trace:
            row = json.loads(line)
            frame = row['frame']
            if frame > end:
                break
            last = frame
            cp = row.get('combat_policy') or {}
            ob = cp.get('observation') or {}
            if ob.get('health', 0) <= 0:
                continue
            if first_clear is None and frame in clear_frames and r['initial_objective_count'] > 0:
                first_clear = frame
            count = next((item['count'] for item in (ob.get('inventory') or [])
                          if item['name'].lower() == 'rockets'), None)
            # Missing inventory differs from a known zero. Ammo use is an
            # observational proxy and is reported separately from native damage.
            if count is not None and previous is not None:
                spent += max(0, previous - count)
            previous = count
            cmd = cp.get('proposed') or {}
            if frame < end and ob.get('weapon', '').lower() in ('rocket launcher', 'models/weapons/v_rocket/tris.md2'):
                commands += int(bool(cmd.get('attack')))
    assert last >= end or end < window_end
    if score_objective and not objective_frames:
        raise ValueError(f"Missing native objective telemetry for world {r['seed']}")
    stderr = (directory/'bot.err').read_text(errors='replace').splitlines()
    audit = [line for line in stderr
             if line.startswith('lewm_gate ') and int(line.split()[1].split('=')[1]) < end]
    return dict(seed=r['seed'], map=r['map'], split=r['split'], self_hp=own, monster_hp=monster,
                rocket_monster_hp=rocket_monster, other_hp=other, rocket_ammo_spent=spent,
                rocket_attack_proposals=commands, deferred=sum('defer=true' in s for s in audit),
                alive=end == window_end and not any('target_class=player' in s and
                int(re.search(r'health_after=(-?\d+)', s)[1]) <= 0 for s in events),
                events=events, gate_audit=audit, first_life_clear_frame=first_clear,
                modes=[line for line in stderr if line.startswith('lewm_mode ')
                       and int(line.split()[1].split('=')[1]) < end],
                refreshes=[line for line in stderr if line.startswith('lewm_refresh ')
                           and int(line.split()[1].split('=')[1]) < end])


def collect(out, roots, policies, workers, *, fixture_rocket=True, port=49110, audit_duration=False,
            followup_frames=0, final_frames=0, score_objective=False):
    if followup_frames and not WINDOW_END <= followup_frames <= 498:
        raise ValueError('Follow-up must extend the default window within native frame limits')
    if final_frames and not 0 < followup_frames < final_frames <= 498:
        raise ValueError('Final window must extend the follow-up window')
    tasks = [(r, k) for r in roots for k in policies]
    def slot(i):
        for r, k in tasks[i::workers]:
            target = out / f"{r['seed']}-{k}.json"
            if target.exists():
                cached=read(target)
                if ((not followup_frames or cached.get('followup_frames')==followup_frames) and
                    (not final_frames or cached.get('final_frames')==final_frames) and
                    (not score_objective or cached.get('objective_verified'))):
                    continue
            with tempfile.TemporaryDirectory(prefix='native-', dir=out) as tmp:
                branch = Path(tmp)/'branch'
                env = dict(os.environ, Q2_LEWM_FORCE_FIXTURE_ROCKET='1' if fixture_rocket else '0',
                           Q2_LEWM_ROCKET_AUDIT='1')
                env.pop('Q2_LEWM_ROCKET_POLICY', None)
                env.pop('Q2_LEWM_TEMPORAL_POLICY', None)
                env['Q2_LEWM_DURATION_AUDIT'] = '1' if audit_duration else '0'
                if k != 'base':
                    key = 'Q2_LEWM_ROCKET_POLICY' if fixture_rocket else 'Q2_LEWM_TEMPORAL_POLICY'
                    env[key] = str(out/f'{k}.json')
                args = [shutil.which('pwsh'), '-NoProfile', '-File', REPO/'scripts/run_combat_checkpoint_branch.ps1',
                        '-SourceRoot', r['source'], '-OutputRoot', branch, '-Client', out/'rocket-client.exe',
                        '-CheckpointTool', TOOL, '-Port', str(port+i), '-Frames', str(2+(final_frames or followup_frames or WINDOW_END-1))]
                proc = subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if proc.returncode:
                    raise RuntimeError(proc.stdout.decode(errors='replace'))
                try:
                    value = summarize(branch, r, fixture_rocket, score_objective=score_objective)
                except InitialObservationMismatch as first:
                    # Recreate the entire native process in a fresh directory.
                    # Score only the retry after verifying its initial observation.
                    retry = branch.with_name('retry')
                    args[args.index('-OutputRoot')+1] = retry
                    proc = subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    if proc.returncode:
                        raise RuntimeError(f'Failed to restore {r["seed"]} for retry') from first
                    value = summarize(retry, r, fixture_rocket, score_objective=score_objective)
                    value['rejected_initial_observation'] = str(first)
                    branch = retry
                if followup_frames:
                    value['followup_frames'] = followup_frames
                    value['longer'] = summarize(branch,r,fixture_rocket,followup_frames+1,score_objective)
                if final_frames:
                    value['final_frames'] = final_frames
                    value['final'] = summarize(branch,r,fixture_rocket,final_frames+1,score_objective)
                if score_objective:
                    value['objective_verified'] = True
                save(target, value)
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        list(pool.map(slot, range(workers)))


def totals(out, roots, policies):
    result = {}
    for policy in policies:
        runs = [read(out/f"{r['seed']}-{policy}.json") for r in roots]
        result[policy] = {k: sum(r[k] for r in runs) for k in
                          ('self_hp', 'monster_hp', 'rocket_monster_hp', 'other_hp', 'rocket_ammo_spent',
                           'rocket_attack_proposals', 'deferred', 'alive')}
        result[policy]['seeds_hurt'] = sum(r['self_hp'] > 0 for r in runs)
    return result


def run(out, workers):
    roots = plan()
    train = [r for r in roots if r['split'] == 'train']
    test = [r for r in roots if r['split'] != 'train']
    protocol = dict(roots=roots, candidates=CANDIDATES, horizon_frames=40,
                    minimum_train_rocket_damage_fraction=.8, minimum_train_rocket_attack_fraction=.5)
    if (out/'plan.json').exists() and read(out/'plan.json') != json.loads(json.dumps(protocol)):
        raise ValueError('Refusing to mix protocols')
    save(out/'plan.json', protocol)
    policies = ['base', *CANDIDATES]
    collect(out, train, policies, workers)
    training = totals(out, train, policies)
    base = training['base']
    eligible = [k for k in policies if training[k]['rocket_monster_hp'] >= .8 * base['rocket_monster_hp']
                and training[k]['rocket_attack_proposals'] >= .5 * base['rocket_attack_proposals']]
    best = min(eligible, key=lambda k: (training[k]['self_hp'], -training[k]['rocket_monster_hp']))
    save(out/'selection.json', dict(best=best, training=training))
    # Evaluate the best feasible learned candidate even if baseline wins training.
    challengers = [k for k in eligible if k != 'base'] or list(CANDIDATES)
    candidate = min(challengers, key=lambda k: (training[k]['self_hp'], -training[k]['rocket_monster_hp']))
    collect(out, test, ['base', candidate], workers)
    report = dict(best_on_train=best, candidate=candidate, train_seeds=len(train), heldout_seeds=len(test),
                  train=training, heldout=totals(out, test, ['base', candidate]))
    save(out/'evaluation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    run(args.output.resolve(), args.workers)
