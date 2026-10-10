"""Re-label sealed observed/native episodes for reward calibration, never training."""
import argparse
import concurrent.futures
import json
import pathlib

from process_combat_architecture_pool import read, save, sha, run, compact_closed_exports


def main():
    ap = argparse.ArgumentParser()
    for name in ('evaluation','exporter','reward','out'):
        ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--compress',action='store_true')
    args = ap.parse_args()
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    protocol = read(args.evaluation/'protocol.json')
    pool = read(args.evaluation/'pool/report.json')
    assert pool['state']=='complete' and pool['source_unchanged'] and not any(j['error'] for j in pool['jobs'])
    tasks = []
    for entry in protocol['evaluations']:
        for path in pathlib.Path(entry['root']).glob('case-*/s-*/report.json'):
            report = read(path)
            assert report['capture_complete'] and report['provenance_valid']
            tasks.append((entry['model']+'-'+entry['label'],path,report['results'][0]))
    def process(item):
        label,path,result = item
        worker = pathlib.Path(result['root'])
        output = args.out/(label+'-'+path.parent.parent.name+'-'+path.parent.name)
        command = [args.exporter,'--trace',worker/'bot.jsonl','--out',output,
                   '--worker','audit','--episode',str(result['seed']),'--server-log',worker/'server.log',
                   '--client-name','SoloRetreatBot','--require-execution','--synchronous',
                   '--reset-expectation',worker/'reset-expectation.json','--reward-config',args.reward]
        goal = result.get('goal_stop')
        if goal:
            command += ['--end-reason','combat_goal_complete','--goal-observed-frame',goal['observed_frame']]
        else:
            command += ['--end-reason','game_frame_limit']
        run(command,output.with_suffix('.log'))
        events, counts, cost, available = [], {}, 0., 0
        aim_references = {}
        aim_positive = aim_negative = 0
        with (output/'rewards.jsonl').open(encoding='utf-8-sig') as stream:
            for line in stream:
                r = json.loads(line)
                if not r['available']: continue
                available += 1
                for name,value in r['components'].items():
                    if name in ('target_churn','turn_away','off_target_attack','stalled_movement') and value < 0:
                        counts[name] = counts.get(name,0)+1
                if r.get('target_cycle'):
                    events.append(r['target_cycle']); cost += r['components']['target_churn']
                reference = r.get('aim_reference')
                if reference is not None:
                    reason = reference['reason']
                    aim_references[reason] = aim_references.get(reason,0)+1
                    if reference['applied']:
                        aim_positive += r['components']['aim_potential'] > 0
                        aim_negative += r['components']['aim_potential'] < 0
                    else:
                        assert r['components']['aim_potential']==0,'Masked reference earned reward'
        if args.compress:
            compact_closed_exports(output)
        return dict(label=label,source_member=str(path.parent),source_report_sha256=sha(path),
                    steps=available,counts=counts,target_churn_cost=cost,cycles=events,
                    aim_references=aim_references,aim_positive=aim_positive,aim_negative=aim_negative,
                    reward_report_sha256=sha(output/'report.json'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(process,tasks))
    save(args.out/'report.json',dict(state='complete',episodes=len(rows),
         exporter_sha256=sha(args.exporter),reward_sha256=sha(args.reward),
         implementation_sha256={p:sha(pathlib.Path(__file__).resolve().parents[1]/p) for p in (
             'internal/learningenv/reward.go','internal/learningenv/action_quality_reward.go',
             'internal/learningenv/target_sequence_reward.go','internal/learningenv/selected_aim_reward.go','cmd/q2combat-export/main.go')},
         available_steps=sum(r['steps'] for r in rows),cycle_events=sum(len(r['cycles']) for r in rows),
         total_cycle_cost=sum(r['target_churn_cost'] for r in rows),members=rows,
         aim_applied_steps=sum(r['aim_references'].get('stable_selected_target',0) for r in rows),
         aim_positive=sum(r['aim_positive'] for r in rows),aim_negative=sum(r['aim_negative'] for r in rows),
         scope='Historical development calibration only, not new fights or training. Native command proof retained. '
               'Cycle preference is not proof of uselessness; delayed projectile effects may follow.'))
    print(json.dumps(dict(episodes=len(rows),cycles=sum(len(r['cycles']) for r in rows))))


if __name__=='__main__': main()
