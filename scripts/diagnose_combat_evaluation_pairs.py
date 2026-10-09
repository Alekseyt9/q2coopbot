"""Closed-capture trace diagnosis; snapshot includes only fully matched pairs."""
import argparse, collections, json, pathlib
from rebind_combat_architecture_evaluation import read, sha, save
from analyze_registered_combat_failures import analyze

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--first-seed-only',action='store_true');a=ap.parse_args()
    root=a.root.resolve();protocol=read(root/'protocol.json');plans=[]
    for e in protocol['evaluations']:
        assert sha(e['plan'])==e['plan_sha256'];plans.append((e,read(e['plan'])))
    jobs=[read(p) for p in (root/'pool/jobs').glob('*-result.json')]
    completed={str(pathlib.Path(j['root'])):j for j in jobs if not j['error']}
    rows=[];fingerprints=set();matched=[]
    for i,task in enumerate(plans[0][1]['tasks']):
        seeds=task['seeds'][:1] if a.first_seed_only else task['seeds']
        for seed in seeds:
            members=[]
            for e,p in plans:
                t=p['tasks'][i];assert t['episode']['id']==task['episode']['id'] and t['seeds']==task['seeds']
                folder=pathlib.Path(e['root'])/f"case-{i}-{t['modes'][0]}"/f's-{seed}'
                if str(folder) not in completed:break
                members.append((e,folder))
            if len(members)!=len(plans):continue
            matched.append({'episode':task['episode']['id'],'seed':seed})
            for e,folder in members:
                report=read(folder/'report.json');mf=read(folder/'manifest.json');result=report['results'][0]
                assert report['capture_complete'] and report['provenance_valid'] and result['capture_valid'] and result['seed']==seed
                fingerprints.add(mf['source_fingerprint'])
                diag=analyze(result['root']);diag.update(variant=e['model']+'-'+e['label'],episode=task['episode']['id'],
                    member_report_sha256=sha(folder/'report.json'),server_log_sha256=sha(pathlib.Path(result['root'])/'server.log'),
                    selection_p95_us=result['selection_p95_us'],verified_goal=bool(result.get('goal_stop')))
                rows.append(diag)
    assert rows and len(fingerprints)==1
    summaries=[]
    for variant in sorted({r['variant'] for r in rows}):
        selected=[r for r in rows if r['variant']==variant];counts=collections.Counter()
        for r in selected:counts.update(r['counts'])
        yaw=[r['yaw_error_mean'] for r in selected if r['yaw_error_mean'] is not None]
        latency=[r['selection_p95_us'] for r in selected if r['selection_p95_us'] is not None]
        summaries.append({'variant':variant,'episodes':len(selected),'verified_goals':sum(r['verified_goal'] for r in selected),
            'mean_episode_min_clear_enemy_attack_yaw_error_degrees':sum(yaw)/len(yaw) if yaw else None,
            'visible_fire_fraction':counts['visible_attack_frames']/max(1,counts['visible_frames']),
            'clear_firing_yaw_over_20_fraction':counts['attack_yaw_gt_20']/max(1,counts['visible_attack_frames']),
            'clear_firing_yaw_under_5_fraction':counts['attack_yaw_lt_5']/max(1,counts['visible_attack_frames']),
            'ground_stall_proxy_fraction':counts['ground_move_under_1unit']/max(1,counts['ground_move_pairs']),
            'jump_frames':counts['jump_frames'],'crouch_frames':counts['crouch_frames'],
            'mean_selection_p95_us':sum(latency)/len(latency) if latency else None})
    expected=protocol['total_episodes'];complete=len(rows)==expected
    save(a.out,{'version':'combat_matched_trace_diagnosis_v1','matched_conditions':matched,'summaries':summaries,'rows':rows,
                'episodes':len(rows),'expected_episodes':expected,'coverage_complete':complete,
                'protocol_sha256':sha(root/'protocol.json'),'source_fingerprint':next(iter(fingerprints)),
                'scope':'Closed first-life traces of fully matched condition sets; coverage_complete describes trace coverage, not quality acceptance. Yaw is minimum horizontal command error to any clear observed enemy, not hit accuracy; low displacement is only a stall proxy.'})
    print(json.dumps({'matched_conditions':len(matched),'episodes':len(rows),'summaries':summaries}))

if __name__=='__main__':main()
