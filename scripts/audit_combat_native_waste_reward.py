"""Calibrate reward v9 on sealed captures, preserving v8 components; no NN execution."""
import argparse
import collections
import concurrent.futures
import json
import math
import pathlib
import time

from process_combat_architecture_pool import read, save as atomic_save, sha, run, compact_closed_exports


def save(path, value):
    # Windows readers can briefly hold a handle without FILE_SHARE_DELETE.
    # Preserve atomic replacement; retry the same write, never drop status.
    for attempt in range(20):
        try:
            return atomic_save(path, value)
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def main():
    ap = argparse.ArgumentParser()
    for name in ('evaluation', 'exporter', 'reward', 'out'):
        ap.add_argument('--'+name, type=pathlib.Path, required=True)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    a.out = a.out.resolve()
    assert a.resume or not a.out.exists()
    a.out.mkdir(parents=True, exist_ok=a.resume)
    protocol = read(a.evaluation/'protocol.json')
    proof = read(a.evaluation/'recovery/verified-members.json')
    assert proof['state'] == 'complete' and proof['protocol_sha256'] == sha(a.evaluation/'protocol.json')
    mg = read(a.evaluation/'machinegun-waste-audit.json')
    blaster = read(a.evaluation/'miss-reward-audit-after-frame-100.json')
    assert mg['protocol_sha256'] == blaster['protocol_sha256'] == proof['protocol_sha256']
    exporter_sha, reward_sha = sha(a.exporter), sha(a.reward)
    reward_config = read(a.reward)
    assert reward_config['version'] == 'combat_reward_v9'
    expected = {(k,e['server_sha256'],e['steps_sha256']) for k,g in mg['groups'].items() for e in g['episodes']}
    tasks = [(e['model']+'-'+e['label'],p) for e in protocol['evaluations']
             for p in sorted(pathlib.Path(e['root']).glob('case-*/s-*/report.json'))]

    def process(item):
        label, path = item
        assert proof['members'][str(path.parent)]['report_sha256'] == sha(path)
        report = read(path)
        assert report['capture_complete'] and report['provenance_valid']
        result = report['results'][0]
        worker = pathlib.Path(result['root'])
        steps, log = worker/'dataset/steps.jsonl', worker/'server.log'
        assert (label,sha(log),sha(steps)) in expected
        old_rewards = [json.loads(s) for s in (worker/'dataset/rewards.jsonl').read_text(encoding='utf-8-sig').splitlines()]
        old_steps = [json.loads(s) for s in steps.read_text(encoding='utf-8-sig').splitlines()]
        first = old_steps[0]
        member = a.out/(label+'-'+path.parent.parent.name+'-'+path.parent.name)
        binding = dict(source_report_sha256=sha(path), server_sha256=sha(log),
                       steps_sha256=sha(steps), old_rewards_sha256=sha(worker/'dataset/rewards.jsonl'),
                       exporter_sha256=exporter_sha, reward_sha256=reward_sha)
        output = None
        if a.resume and (member/'complete.json').exists():
            saved = read(member/'complete.json')
            assert saved['binding'] == binding
            assert saved['report_sha256'] == sha(pathlib.Path(saved['export'])/'report.json')
            if saved.get('summary_version') == 2:
                return saved['summary']
            output = pathlib.Path(saved['export'])
        member.mkdir(exist_ok=a.resume)
        attempt = len(list(member.glob('export-*')))
        reused = output is not None
        output = output or member/f'export-{attempt}'
        command = [a.exporter, '--trace',worker/'bot.jsonl', '--out',output,
                   '--worker',first['worker'], '--episode',first['episode'],
                   '--server-log',log, '--client-name','SoloRetreatBot',
                   '--require-execution','--synchronous', '--reset-expectation',worker/'reset-expectation.json',
                   '--reward-config',a.reward]
        if result.get('goal_stop'):
            command += ['--end-reason','combat_goal_complete','--goal-observed-frame',result['goal_stop']['observed_frame']]
        elif result.get('death_stop'):
            save(member/'death-stop.json', result['death_stop'])
            command += ['--end-reason','combat_first_life_death','--death-stop',member/'death-stop.json']
        else:
            command += ['--end-reason','game_frame_limit']
        if not reused:
            run(command,member/f'export-{attempt}.log')
        rewards = [json.loads(s) for s in (output/'rewards.jsonl').read_text(encoding='utf-8-sig').splitlines()]
        outcomes = [json.loads(s) for s in (output/'server_outcomes.jsonl').read_text(encoding='utf-8-sig').splitlines()]
        assert len(rewards) == len(old_rewards) == len(old_steps) == len(outcomes)
        counts = collections.Counter()
        total_cost = 0.
        for old,new,step,outcome in zip(old_rewards,rewards,old_steps,outcomes):
            assert old['version'] == 'combat_reward_v8'
            assert (old['worker'],old['episode'],old['step'],old['available'],old.get('reason')) == (new['worker'],new['episode'],new['step'],new['available'],new.get('reason'))
            if not old['available']:
                continue
            extra = {k:new['components'][k] for k in ('blaster_miss','machinegun_miss')}
            assert old['components'] == {k:v for k,v in new['components'].items() if k not in extra}
            assert old.get('aim_reference') == new.get('aim_reference')
            cost = sum(extra.values())
            assert math.isclose(new['score'],old['score']+cost,rel_tol=0,abs_tol=1e-12)
            counts['available_steps'] += 1
            if step['observation']['identity']['frame'] <= 100:
                continue
            for field,key,coefficient in [('hitscan_misses','machinegun_misses','machinegun_miss'),
                                          ('projectile_misses','blaster_misses','blaster_miss')]:
                misses = outcome[field]['misses']
                n = len(misses)
                counts[key] += n if field == 'hitscan_misses' else sum(m['launch']['begin_frame']>100 for m in misses)
                if field == 'projectile_misses':
                    counts['blaster_early_launch_late_misses'] += sum(m['launch']['begin_frame']<=100 for m in misses)
                assert math.isclose(extra[coefficient],n*reward_config[coefficient],rel_tol=0,abs_tol=1e-12)
            total_cost += cost
        summary = dict(label=label, source_member=str(path.parent), counts=dict(counts), after_frame100_cost=total_cost)
        if not reused:
            compact_closed_exports(output)
        save(member/'complete.json', dict(binding=binding,export=str(output),report_sha256=sha(output/'report.json'),summary=summary,summary_version=2))
        return summary

    completed, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process,t):t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:
                completed.append(future.result())
            except Exception as e:
                errors.append(dict(member=str(futures[future][1]),error=str(e)))
            save(a.out/'progress.json',dict(stage='calibrating',done=len(completed),errors=errors,total=len(tasks)))
    if errors:
        raise RuntimeError('Calibration failed; inspect progress and member export logs')
    groups = {}
    for label in mg['groups']:
        rows = [r for r in completed if r['label']==label]
        assert len(rows) == protocol['episodes_per_model']
        counts = collections.Counter()
        for row in rows:
            counts.update(row['counts'])
        assert counts['machinegun_misses'] == mg['groups'][label]['strata']['all']['creditable_confirmed_waste']
        assert counts['blaster_misses'] == blaster['groups'][label]['counts'].get('creditable_misses',0)
        groups[label] = dict(episodes=len(rows),counts=dict(counts),total_cost=sum(r['after_frame100_cost'] for r in rows))
    implementation = ('internal/learningenv/reward.go','internal/learningenv/hitscan_miss.go',
                      'internal/learningenv/projectile_miss.go','cmd/q2combat-export/main.go')
    save(a.out/'report.json',dict(state='complete',episodes=len(completed),groups=groups,
        reward_sha256=reward_sha,exporter_sha256=exporter_sha,
        implementation_sha256={p:sha(repo/p) for p in implementation},members=completed,
        scope='Historical calibration only, no training. V8 components and selected target references '
              'unchanged; V9 cost from exactly joined actual shots. Comparable weapon counts use launch/end frame>100; '
              'earlier launches ending later have a separate counter and still contribute to reward cost. '
              'Native labels never enter policy observations.'))
    save(a.out/'progress.json',dict(stage='complete',episodes=len(completed),report_sha256=sha(a.out/'report.json')))
    print(json.dumps(groups),flush=True)


if __name__ == '__main__':
    main()
