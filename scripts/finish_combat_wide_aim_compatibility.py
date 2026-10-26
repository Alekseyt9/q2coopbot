"""Verify disabled native identity and first learned-option source/prefix, then CUDA."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from finish_combat_anchor_probe import normalized,first_diff
from recover_combat_anchor_screen import first_life,terminal_receipt
from process_combat_architecture_pool import read,save,sha


def source_sample(step):
    s=copy.deepcopy(step['sample']);s.pop('version')
    w=s.pop('wide_aim',None)
    if w:
        s['latent']=w['source_latent'];s['log_probability']-=w['log_probability']
    return s


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root.resolve();repo=Path(__file__).resolve().parents[1];assert not (root/'complete.json').exists()
    spec,pool=read(root/'spec.json'),read(root/'pool/report.json')
    assert spec['version']=='combat_wide_aim_native_compatibility_v1' and pool['state']=='complete'
    assert pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==9
    for p,h in spec['pins'].items():assert sha(p)==h
    def run(label,cmd):
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run(list(map(str,cmd)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,str(root/(label+'.log'))
    cases={};jobs={};pins={};stats=[]
    for j in pool['jobs']:
        a=spec['assignments'][j['plan_index']];assert not j['error'] and j['seed']==a['seed']
        report=Path(j['root'])/'report.json';r=read(report)
        assert r['capture_complete'] and r['provenance_valid'];r=r['results'][0]
        assert r['capture_valid'] and r['dispatch_valid'] and r['seed_confirmed']
        path=Path(r['root'])/'dataset/steps.jsonl';steps=[json.loads(l) for l in path.open(encoding='utf-8-sig')]
        active=first_life(r,steps);cases[(a['seed'],a['variant'])]=(r,active);jobs[a['plan_index']]=j
        wide=[s for s in active if (s.get('sample') or {}).get('wide_aim')]
        stats.append(dict(**a,frames=len(active),wide_events=len(wide),wide_chosen=sum(s['sample']['wide_aim']['choice'] for s in wide)))
        pins[str(report)]=sha(report);pins[str(path)]=sha(path)
    comparisons=[]
    for seed in sorted({a['seed'] for a in spec['assignments']}):
        br,base=cases[(seed,'legacy')];dr,disabled=cases[(seed,'disabled')];er,enabled=cases[(seed,'enabled')]
        disabled_diff=first_diff([normalized(s) for s in base],[normalized(s) for s in disabled])
        terminal_diff=first_diff(terminal_receipt(br),terminal_receipt(dr))
        chosen=[s for s in enabled if (s.get('sample') or {}).get('wide_aim',{}).get('choice')==1]
        assert chosen,'Enabled compatibility case did not exercise the new option'
        frame=chosen[0]['observation']['identity']['frame'];before={s['observation']['identity']['frame']:s for s in base}
        prefix_error=None;max_lp_error=0.;prefix_rows=0
        for other in enabled:
            f=other['observation']['identity']['frame']
            if f>=frame:break
            assert f in before;b=before[f];x,y=normalized(b),normalized(other)
            if other.get('sample'):
                a,c=source_sample(b),source_sample(other)
                err=abs(a['log_probability']-c['log_probability']);max_lp_error=max(max_lp_error,err)
                if err>1e-9:prefix_error='source_likelihood_difference'
                c['log_probability']=a['log_probability'];x['sample'],y['sample']=a,c
            prefix_error=prefix_error or first_diff(x,y);prefix_rows+=1
        b=before[frame];other=chosen[0]
        # Fire is intentionally conditioned on the NEW final aim. The original
        # Gaussian/movement/target/mode/weapon/25-way event must still match.
        x,y=source_sample(b),source_sample(other)
        for v in (x,y):v.pop('attack');v.pop('log_probability')
        source_diff=first_diff(x,y)
        obs_diff=first_diff(b['observation'],other['observation'])
        ba,aa=copy.deepcopy(b['action']),copy.deepcopy(other['action'])
        for v in (ba,aa):
            for k in ('yaw_delta_degrees','pitch_delta_degrees','attack'):v.pop(k)
        other_action_diff=first_diff(ba,aa)
        comparisons.append(dict(seed=seed,first_wide_frame=frame,prefix_rows=prefix_rows,source_lp_max_rounding_error=max_lp_error,
            disabled_first_life_difference=disabled_diff,disabled_terminal_difference=terminal_diff,
            enabled_prefix_difference=prefix_error,first_wide_source_difference=source_diff,first_wide_observation_difference=obs_diff,
            non_aim_non_fire_action_difference=other_action_diff,
            accepted=all(v is None for v in (disabled_diff,terminal_diff,prefix_error,source_diff,obs_diff,other_action_diff))))
    save(root/'comparison.json',dict(state='passed' if all(c['accepted'] for c in comparisons) else 'rejected',comparisons=comparisons,
        stats=stats,pins=pins,training_performed=False,promotion=False,quality_confirmation=False))
    assert all(c['accepted'] for c in comparisons),str(root/'comparison.json')
    exporter=root/'q2ppo-data.exe';run('build-exporter',['go','build','-o',exporter,'./cmd/q2ppo-data'])
    tasks=[]
    for a in spec['assignments']:
        out=root/('replay-'+str(a['plan_index']));out.mkdir()
        run('export-'+str(a['plan_index']),[exporter,'--batch',jobs[a['plan_index']]['root'],'--model',a['model'],'--cuda-only','--out',out/'native'])
        tasks.append(dict(model=a['model'],model_sha256=a['model_sha256'],data=str(out/'native'),out=str(out/'cuda'),native_report_sha256=sha(out/'native/report.json')))
    save(root/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
    save(root/'progress.json',dict(stage='cuda',cases=9))
    run('cuda',[sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',root/'cuda-request.json','--receipt-root',root/'cuda-batch'])
    batch=read(root/'cuda-batch/report.json');assert batch['state']=='complete' and batch['device']=='cuda' and len(batch['tasks'])==9
    for p,h in {**pins,**spec['pins']}.items():assert sha(p)==h
    save(root/'complete.json',dict(state='complete',native_cases=9,cuda_cases=9,worlds=3,
        comparison_sha256=sha(root/'comparison.json'),cuda_sha256=sha(root/'cuda-batch/report.json'),
        training_performed=False,promotion=False,quality_confirmation=False,
        scope='Reused TRAIN seed compatibility only; no ordinary PPO request or model promotion. Legacy and disabled commands identical; enabled preserves source events before its independent option and conditions fire on actual aim.'))
    print(root/'complete.json')


if __name__=='__main__':main()
