"""Observed clear-target decisions, without model replay or causal inference."""
import argparse,collections,hashlib,json,math,pathlib
from process_combat_architecture_pool import read,save,sha

def count_row(row,first_life,counts):
    obs=row['observation'];identity=obs['identity']
    if identity['life']!=first_life or obs['health']<=0:return
    assert row['server_execution']['matched']
    action=row['action'];applied=row['applied_action'];counts['frames']+=1
    visible=[e for e in obs.get('enemies',[]) if e.get('clear_shot')]
    if not visible:return
    counts['clear_frames']+=1
    counts['clear_attack_requested']+=int(action['attack'])
    counts['clear_attack_sent']+=int(applied['attack'])
    counts['clear_request_blocked']+=int(action['attack'] and not applied['attack'])
    counts['clear_no_attack_requested']+=int(not action['attack'])
    if row['owner']=='provider':
        counts['provider_clear_frames']+=1
        counts['provider_clear_without_selected_target']+=int(not action.get('target_entity'))
    requested=math.hypot(action['forward'],action['side'])
    executed=math.hypot(applied['forward'],applied['side'])
    velocity=obs['velocity'];speed=math.hypot(velocity[0],velocity[1])
    assert all(math.isfinite(v) for v in (requested,executed,speed))
    counts['clear_observed_stationary']+=int(speed<40)
    counts['clear_stationary_with_move_request']+=int(speed<40 and requested>.1)
    counts['clear_move_reduced']+=int(requested>.1 and executed+.05<requested)
    counts['clear_crouch_requested']+=int(action['vertical']=='crouch')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True)
    ap.add_argument('--families',nargs='+',required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    root=a.root.resolve();protocol=read(root/'protocol.json');proof=read(root/'recovery/verified-members.json');quality=read(root/'quality-report.json')
    assert proof['state']==quality['state']=='complete'
    assert proof['protocol_sha256']==quality['protocol_sha256']==sha(root/'protocol.json')
    groups={};selected=set(a.families)
    for entry in protocol['evaluations']:
        variant=entry['model']+'-'+entry['label'];plan=read(entry['plan']);assert sha(entry['plan'])==entry['plan_sha256']
        found=set();episodes=[];total=collections.Counter()
        for index,task in enumerate(plan['tasks']):
            if task['episode']['id'] not in selected:continue
            found.add(task['episode']['id'])
            mode='rules' if entry['model']=='rules' else 'learned'
            for seed in task['seeds']:
                member=pathlib.Path(entry['root'])/f'case-{index}-{mode}'/f's-{seed}'
                receipt=proof['members'][str(member)]
                assert sha(member/'report.json')==receipt['report_sha256'] and sha(member/'manifest.json')==receipt['manifest_sha256']
                report=read(member/'report.json');assert len(report['results'])==1 and report['capture_complete'] and report['provenance_valid']
                result=report['results'][0];assert result['seed']==seed
                steps=pathlib.Path(result['root'])/'dataset/steps.jsonl';digest=hashlib.sha256();counts=collections.Counter();first_life=None
                with steps.open('rb') as stream:
                    for line in stream:
                        digest.update(line);row=json.loads(line)
                        if first_life is None and row['observation']['health']>0:first_life=row['observation']['identity']['life']
                        count_row(row,first_life,counts)
                assert counts['frames'];total.update(counts)
                episodes.append(dict(family=task['episode']['id'],seed=seed,steps=str(steps),steps_sha256=digest.hexdigest(),counts=dict(counts)))
        assert found==selected,'Requested family missing from frozen plan'
        groups[variant]=dict(episodes=episodes,counts=dict(total))
    save(a.out,dict(version='combat_visible_target_decisions_v1',state='complete',protocol_sha256=sha(root/'protocol.json'),member_proof_sha256=sha(root/'recovery/verified-members.json'),families=sorted(selected),groups=groups,scope='First observed life, alive pre-command observations, matched native dispatch. clear_shot is observed visibility, not proven safe/ready ballistics. Attack false is actor/rules request; request blocked means final applied false. Stationary uses pre-command XY velocity<40, not causally proven wall trapping; move reduction threshold0.05 ignores small quantization. Provider target intent only, rules missing target metadata not treated as no-target decisions. No NN replay, training, quality promotion or unseen-test claims.'))
    print(json.dumps({k:v['counts'] for k,v in groups.items()}),flush=True)

if __name__=='__main__':main()
