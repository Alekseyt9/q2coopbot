"""Close diagnostic per-case CUDA replays and exact native prefix checks."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from process_combat_architecture_pool import read,save,sha


def normalized(step):
    # Provider content hashes necessarily change with diagnostic headers.
    # All observed state/history, sampled events and native commands stay exact.
    v=copy.deepcopy(step);v.pop('provider',None)
    if v.get('sample'):
        v['sample'].pop('version',None);v['sample'].pop('anchor_probe',None)
    return v


def first_diff(a,b,path=''):
    if type(a) is not type(b):return path+':type'
    if isinstance(a,dict):
        if a.keys()!=b.keys():return path+':keys'
        return next((d for k in a if (d:=first_diff(a[k],b[k],path+'/'+k))),None)
    if isinstance(a,list):
        if len(a)!=len(b):return path+':length'
        return next((d for i,(x,y) in enumerate(zip(a,b)) if (d:=first_diff(x,y,path+'/'+str(i)))),None)
    return None if a==b else path


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve();repo=Path(__file__).resolve().parents[1]
    assert not (root/'complete.json').exists()
    spec=read(root/'spec.json');pool=read(root/'pool/report.json')
    assert spec['version']=='combat_one_frame_observed_mg_anchor_v1'
    assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==12
    for p,h in spec['pins'].items():assert sha(p)==h,'Pinned source changed: '+p
    def run(label,command):
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run(list(map(str,command)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,str(root/(label+'.log'))
    exporter=root/'q2ppo-data.exe';run('build-exporter',['go','build','-o',exporter,'./cmd/q2ppo-data'])
    cases={};tasks=[];pins={};metrics=[]
    for j in sorted(pool['jobs'],key=lambda j:j['plan_index']):
        a=spec['assignments'][j['plan_index']];assert a['plan_index']==j['plan_index'] and a['seed']==j['seed'] and not j['error']
        member=Path(j['root']);report=read(member/'report.json');r=report['results'][0]
        assert report['capture_complete'] and report['provenance_valid'] and r['capture_valid'] and r['dispatch_valid'] and r['seed_confirmed']
        assert r['generated_fixture']['split']=='train' and r['seed']==a['seed']
        path=Path(r['root'])/'dataset/steps.jsonl';steps=[json.loads(line) for line in path.open(encoding='utf-8-sig')]
        assert len(steps)>0 and all(s['server_execution']['matched'] for s in steps)
        cases[(a['context_id'],a['mode'],a['repeat'])]=steps
        d=r['first_life']['damage']
        metrics.append(dict(**a,goal=bool(r['goal_stop']),death=bool(r['death_stop'] or r['first_life']['minimum_health']<=0),
            kills=sum(m['monster_kills'] for m in d['by_mod']),dealt=sum(m['monster_health_damage'] for m in d['by_mod']),
            received=d['received_health_damage'],frames=len(steps)))
        pins[str(path)]=sha(path);pins[str(member/'report.json')]=sha(member/'report.json')
        out=root/('replay-'+str(j['plan_index']));out.mkdir()
        run('export-'+str(j['plan_index']),[exporter,'--batch',member,'--model',a['model'],'--cuda-only','--out',out/'native'])
        tasks.append(dict(model=a['model'],model_sha256=a['model_sha256'],data=str(out/'native'),out=str(out/'cuda'),
            native_report_sha256=sha(out/'native/report.json')))
    save(root/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',tasks=tasks,compact_finalized_streams=True))
    save(root/'progress.json',dict(stage='cuda',cases=len(tasks)))
    run('cuda',[sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',root/'cuda-request.json','--receipt-root',root/'cuda-batch'])
    results=[]
    for ctx in spec['contexts']['contexts']:
        key=ctx['context_id'];frame=ctx['branch_frame'];base=cases[(key,'baseline',0)]
        for mode in ('baseline','identity','observed_mg'):
            for repeat in (0,1):
                other=cases[(key,mode,repeat)]
                prefix=[normalized(s) for s in base if s['observation']['identity']['frame']<frame]
                alt=[normalized(s) for s in other if s['observation']['identity']['frame']<frame]
                diff=first_diff(prefix,alt)
                root_base=next(s for s in base if s['observation']['identity']['frame']==frame)
                roots=[s for s in other if s['observation']['identity']['frame']==frame]
                branch_error='missing_branch' if len(roots)!=1 else None
                if not branch_error:
                    before=normalized(root_base);after=normalized(roots[0])
                    # State before action and all original random decisions must match.
                    branch_error=first_diff(before['observation'],after['observation'],'observation') or first_diff(before['sample'],after['sample'],'sample')
                    x,y=copy.deepcopy(before['action']),copy.deepcopy(after['action'])
                    if mode=='observed_mg':
                        for v in (x,y):v.pop('yaw_delta_degrees');v.pop('pitch_delta_degrees')
                        assert (before['action']['yaw_delta_degrees'],before['action']['pitch_delta_degrees']) != (after['action']['yaw_delta_degrees'],after['action']['pitch_delta_degrees'])
                    branch_error=branch_error or first_diff(x,y,'action')
                full=first_diff([normalized(s) for s in base],[normalized(s) for s in other]) if mode!='observed_mg' else None
                results.append(dict(context_id=key,mode=mode,repeat=repeat,branch_frame=frame,prefix_rows=len(prefix),
                    prefix_difference=diff,source_branch_difference=branch_error,identity_full_difference=full,
                    accepted=diff is None and branch_error is None and full is None))
        # Repeated interventions must reproduce their own complete trajectory.
        a=[normalized(s) for s in cases[(key,'observed_mg',0)]]
        b=[normalized(s) for s in cases[(key,'observed_mg',1)]]
        diff=first_diff(a,b)
        results.append(dict(context_id=key,mode='observed_mg_repeat',difference=diff,accepted=diff is None))
    save(root/'comparison.json',dict(state='passed' if all(r['accepted'] for r in results) else 'rejected',
        comparisons=results,metrics=metrics,source_pins=pins,training_performed=False,promotion=False,
        quality_confirmation=False,scope='Two selected reused TRAIN states. Exact observation/history/source-event/native prefix and repeated identity trajectories. Intervention may change world movement through view-relative controls; no causal general-quality or learned-controller claim.'))
    assert all(r['accepted'] for r in results),str(root/'comparison.json')
    for p,h in {**spec['pins'],**pins}.items():assert sha(p)==h
    cuda=read(root/'cuda-batch/report.json');assert cuda['state']=='complete' and cuda['device']=='cuda' and len(cuda['tasks'])==12
    save(root/'complete.json',dict(state='complete',native_cases=12,cuda_cases=12,unique_train_contexts=2,
        comparison_sha256=sha(root/'comparison.json'),cuda_sha256=sha(root/'cuda-batch/report.json'),
        pool_sha256=sha(root/'pool/report.json'),spec_sha256=sha(root/'spec.json'),
        training_performed=False,promotion=False,quality_confirmation=False))
    print(root/'complete.json')


if __name__=='__main__':main()
