"""Strict prefix/export/CUDA closure for the other-map anchor TRAIN screen."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from finish_combat_anchor_probe import normalized,first_diff
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve();repo=Path(__file__).resolve().parents[1]
    assert not (root/'complete.json').exists()
    spec,pool=read(root/'spec.json'),read(root/'pool/report.json')
    assert spec['version']=='combat_observed_anchor_other_maps_screen_v1'
    assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==24
    for p,h in spec['pins'].items():assert sha(p)==h
    def run(label,command):
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run(list(map(str,command)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,str(root/(label+'.log'))
    def load(member):
        report=Path(member)/'report.json';r=read(report);assert r['capture_complete'] and r['provenance_valid']
        r=r['results'][0];assert r['capture_valid'] and r['dispatch_valid'] and r['seed_confirmed']
        path=Path(r['root'])/'dataset/steps.jsonl'
        return r,[json.loads(line) for line in path.open(encoding='utf-8-sig')],path
    cases={};bindings={};pins={};metrics=[]
    for j in pool['jobs']:
        a=spec['assignments'][j['plan_index']];assert not j['error'] and j['seed']==a['seed']
        r,steps,path=load(j['root']);assert r['generated_fixture']['split']=='train'
        assert all(s['server_execution']['matched'] for s in steps)
        key=(a['context_id'],a['mode']);assert key not in cases;cases[key]=steps;bindings[key]=j
        d=r['first_life']['damage'];metrics.append(dict(**a,goal=bool(r['goal_stop']),
            death=bool(r['death_stop'] or r['first_life']['minimum_health']<=0),
            kills=sum(m['monster_kills'] for m in d['by_mod']),dealt=sum(m['monster_health_damage'] for m in d['by_mod']),
            received=d['received_health_damage'],frames=len(steps)))
        pins[str(path)]=sha(path)
    comparisons=[]
    for ctx in spec['contexts']['contexts']:
        cid,frame=ctx['context_id'],ctx['branch_frame'];base=cases[(cid,'baseline')]
        assert sha(ctx['original_steps'])==ctx['original_steps_sha256']
        _,original,_=load(ctx['baseline_member'])
        old_diff=first_diff([normalized(s) for s in original],[normalized(s) for s in base])
        for mode in ('identity','observed_mg'):
            other=cases[(cid,mode)]
            prefix=lambda rows:[normalized(s) for s in rows if s['observation']['identity']['frame']<frame]
            prefix_diff=first_diff(prefix(base),prefix(other));branch_diff=None
            b=[s for s in base if s['observation']['identity']['frame']==frame]
            c=[s for s in other if s['observation']['identity']['frame']==frame]
            if len(b)!=1 or len(c)!=1:branch_diff='missing_branch'
            else:
                x,y=normalized(b[0]),normalized(c[0])
                branch_diff=first_diff(x['observation'],y['observation'],'observation') or first_diff(x['sample'],y['sample'],'sample')
                xx,yy=copy.deepcopy(x['action']),copy.deepcopy(y['action'])
                if mode=='observed_mg':
                    assert any(xx[k]!=yy[k] for k in ('yaw_delta_degrees','pitch_delta_degrees'))
                    for v in (xx,yy):v.pop('yaw_delta_degrees');v.pop('pitch_delta_degrees')
                branch_diff=branch_diff or first_diff(xx,yy,'action')
            identity_diff=first_diff([normalized(s) for s in base],[normalized(s) for s in other]) if mode=='identity' else None
            comparisons.append(dict(context_id=cid,mode=mode,original_full_difference=old_diff,
                prefix_difference=prefix_diff,branch_difference=branch_diff,identity_full_difference=identity_diff,
                accepted=all(v is None for v in (old_diff,prefix_diff,branch_diff,identity_diff))))
    save(root/'comparison.json',dict(state='passed' if all(v['accepted'] for v in comparisons) else 'rejected',
        comparisons=comparisons,metrics=metrics,pins=pins,training_performed=False,promotion=False,quality_confirmation=False))
    assert all(v['accepted'] for v in comparisons),str(root/'comparison.json')
    exporter=root/'q2ppo-data.exe';run('build-exporter',['go','build','-o',exporter,'./cmd/q2ppo-data'])
    tasks=[]
    for a in spec['assignments']:
        j=bindings[(a['context_id'],a['mode'])];directory=root/('replay-'+str(a['plan_index']));directory.mkdir()
        run('export-'+str(a['plan_index']),[exporter,'--batch',j['root'],'--model',a['model'],'--cuda-only','--out',directory/'native'])
        tasks.append(dict(model=a['model'],model_sha256=a['model_sha256'],data=str(directory/'native'),out=str(directory/'cuda'),
            native_report_sha256=sha(directory/'native/report.json')))
    save(root/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',tasks=tasks,compact_finalized_streams=True))
    save(root/'progress.json',dict(stage='cuda',cases=len(tasks)))
    run('cuda',[sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',root/'cuda-request.json','--receipt-root',root/'cuda-batch'])
    cuda=read(root/'cuda-batch/report.json');assert cuda['state']=='complete' and cuda['device']=='cuda' and len(cuda['tasks'])==24
    for p,h in {**pins,**spec['pins']}.items():assert sha(p)==h
    save(root/'complete.json',dict(state='complete',native_cases=24,cuda_cases=24,unique_train_contexts=8,
        comparison_sha256=sha(root/'comparison.json'),pool_sha256=sha(root/'pool/report.json'),
        cuda_sha256=sha(root/'cuda-batch/report.json'),spec_sha256=sha(root/'spec.json'),
        training_performed=False,promotion=False,quality_confirmation=False))
    # Existing read-only reporters accept the shared assignments/capture schema.
    for name in ('report_combat_anchor_probe.py','report_combat_anchor_continuation.py'):
        run(name.removesuffix('.py'),[sys.executable,repo/'scripts'/name,'--root',root])
    print(root/'complete.json')


if __name__=='__main__':main()
