"""Post-hoc first-life analysis of a REJECTED V1; never rewrite its verdict."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from finish_combat_anchor_probe import normalized,first_diff
from process_combat_architecture_pool import read,save,sha


def first_life(result,steps):
    life=result['first_life'];end=life['end_frame'];goal=result.get('goal_stop')
    if goal:
        assert goal['observed_frame']==end and goal['native_objective']['frame']==end
        assert goal['native_objective']['alive']==goal['native_objective']['pending']==0
    active=[s for s in steps if s['observation']['identity']['life']==1 and
        life['start_frame']<=s['observation']['identity']['frame']<=end]
    assert active and all(s['server_execution']['matched'] for s in active)
    assert all(b['observation']['identity']['frame']==a['observation']['identity']['frame']+1 for a,b in zip(active,active[1:]))
    return active


def terminal_receipt(result):
    receipt=copy.deepcopy({k:result.get(k) for k in ('goal_stop','death_stop','first_life')})
    # The monitor renders visible enemies from a Go map. Order is not semantic
    # in these DEBUG snapshots. Actual policy observations/steps stay unsorted
    # and are still compared byte-for-value, including all IDs and slot order.
    for frame in receipt['first_life'].get('last_frames',[]):
        for name in ('visible_enemies','grenades'):
            if name in frame:
                frame[name].sort(key=lambda entity:entity['id'])
    return receipt


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    original=args.root.resolve();root=args.out.resolve();repo=Path(__file__).resolve().parents[1]
    assert not root.exists() and read(original/'progress.json')['stage']=='failed'
    rejected=read(original/'comparison.json');assert rejected['state']=='rejected'
    spec,pool=read(original/'spec.json'),read(original/'pool/report.json')
    assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==24
    for p,h in spec['pins'].items():assert sha(p)==h
    root.mkdir(parents=True);(root/'pool').mkdir();shutil.copyfile(original/'pool/report.json',root/'pool/report.json')
    shutil.copyfile(original/'spec.json',root/'spec.json')
    save(root/'protocol.json',dict(version='combat_anchor_first_life_posthoc_v3',original_protocol_passed=False,
        original_rejection_sha256=sha(original/'comparison.json'),original_root=str(original),
        scope_correction='Use already-defined verified first_life.end_frame; keep exact IDs/current observations/source events/native commands. Ignore only post-terminal shutdown tail, never active combat. Historical replay mismatches remain explicit and are not a current-pair acceptance claim.',
        debug_snapshot_order='Sort visible_enemies by unchanged ID ONLY in monitor first_life.last_frames, never in policy observations or steps. Preserve all debug fields, IDs and values.',
        includes_all_cases=True,native_cases=24,unique_contexts=8,training_performed=False,promotion=False,quality_confirmation=False))
    def run(label,cmd):
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run(list(map(str,cmd)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,str(root/(label+'.log'))
    try:
        cases={};bindings={};pins={str(Path(__file__).resolve()):sha(__file__)}
        for j in pool['jobs']:
            a=spec['assignments'][j['plan_index']];assert not j['error'] and j['seed']==a['seed']
            report=Path(j['root'])/'report.json';r=read(report)
            assert r['capture_complete'] and r['provenance_valid'];r=r['results'][0]
            assert r['capture_valid'] and r['dispatch_valid'] and r['seed_confirmed'] and r['generated_fixture']['split']=='train'
            path=Path(r['root'])/'dataset/steps.jsonl';steps=[json.loads(line) for line in path.open(encoding='utf-8-sig')]
            active=first_life(r,steps);assert a['branch_frame']<=r['first_life']['end_frame']
            cases[(a['context_id'],a['mode'])]=(r,active,len(steps)-len(active));bindings[a['plan_index']]=j
            pins[str(path)]=sha(path);pins[str(report)]=sha(report)
        comparisons=[]
        for ctx in spec['contexts']['contexts']:
            cid,f=ctx['context_id'],ctx['branch_frame'];br,base,btail=cases[(cid,'baseline')]
            for mode in ('identity','observed_mg'):
                r,other,tail=cases[(cid,mode)]
                norm=lambda rows:[normalized(s) for s in rows]
                prefix=lambda rows:norm([s for s in rows if s['observation']['identity']['frame']<f])
                pd=first_diff(prefix(base),prefix(other))
                b=next(s for s in base if s['observation']['identity']['frame']==f)
                c=next(s for s in other if s['observation']['identity']['frame']==f)
                x,y=normalized(b),normalized(c)
                branch=first_diff(x['observation'],y['observation'],'observation') or first_diff(x['sample'],y['sample'],'sample')
                xx,yy=copy.deepcopy(x['action']),copy.deepcopy(y['action'])
                if mode=='observed_mg':
                    assert any(xx[k]!=yy[k] for k in ('yaw_delta_degrees','pitch_delta_degrees'))
                    for v in (xx,yy):v.pop('yaw_delta_degrees');v.pop('pitch_delta_degrees')
                branch=branch or first_diff(xx,yy,'action')
                identity=first_diff(norm(base),norm(other)) if mode=='identity' else None
                terminal=first_diff(terminal_receipt(br),terminal_receipt(r)) if mode=='identity' else None
                comparisons.append(dict(context_id=cid,mode=mode,prefix_difference=pd,branch_difference=branch,
                    identity_first_life_difference=identity,identity_terminal_difference=terminal,
                    baseline_post_terminal_rows=btail,variant_post_terminal_rows=tail,
                    accepted=all(v is None for v in (pd,branch,identity,terminal))))
        save(root/'comparison.json',dict(state='passed' if all(v['accepted'] for v in comparisons) else 'rejected',
            comparisons=comparisons,metrics=rejected['metrics'],pins=pins,original_protocol_passed=False,
            historical_full_replay_failures=[v for v in rejected['comparisons'] if v['original_full_difference'] is not None],
            posthoc_scope_correction=True,training_performed=False,promotion=False,quality_confirmation=False))
        assert all(v['accepted'] for v in comparisons),str(root/'comparison.json')
        exporter=root/'q2ppo-data.exe';run('build-exporter',['go','build','-o',exporter,'./cmd/q2ppo-data'])
        tasks=[]
        save(root/'progress.json',dict(stage='native-export',cases=24))
        for a in spec['assignments']:
            directory=root/('replay-'+str(a['plan_index']));directory.mkdir()
            run('export-'+str(a['plan_index']),[exporter,'--batch',bindings[a['plan_index']]['root'],'--model',a['model'],'--cuda-only','--out',directory/'native'])
            tasks.append(dict(model=a['model'],model_sha256=a['model_sha256'],data=str(directory/'native'),out=str(directory/'cuda'),native_report_sha256=sha(directory/'native/report.json')))
        save(root/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',tasks=tasks,compact_finalized_streams=True))
        save(root/'progress.json',dict(stage='cuda',cases=24))
        run('cuda',[sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',root/'cuda-request.json','--receipt-root',root/'cuda-batch'])
        receipt=read(root/'cuda-batch/report.json');assert receipt['state']=='complete' and len(receipt['tasks'])==24 and receipt['device']=='cuda'
        for p,h in {**pins,**spec['pins']}.items():assert sha(p)==h
        save(root/'complete.json',dict(state='complete',native_cases=24,cuda_cases=24,unique_train_contexts=8,
            comparison_sha256=sha(root/'comparison.json'),pool_sha256=sha(root/'pool/report.json'),
            cuda_sha256=sha(root/'cuda-batch/report.json'),protocol_sha256=sha(root/'protocol.json'),
            original_protocol_passed=False,posthoc_scope_correction=True,training_performed=False,promotion=False,quality_confirmation=False))
        for name in ('report_combat_anchor_probe.py','report_combat_anchor_continuation.py'):
            run(name.removesuffix('.py'),[sys.executable,repo/'scripts'/name,'--root',root])
        save(root/'progress.json',dict(stage='complete',complete_sha256=sha(root/'complete.json')))
        print(root/'complete.json')
    except BaseException as error:
        save(root/'progress.json',dict(stage='failed',error=str(error),exception=type(error).__name__));raise


if __name__=='__main__':main()
