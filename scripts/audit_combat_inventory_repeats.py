"""Strict same-model repeat evidence on current inventory-retry native bindings."""
import argparse
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from native_combat_release import release_boundary
from audit_combat_joint_aim_grid import physical_equal_with_entity_bijection

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();root=a.evaluation.resolve();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path);pins[str(path)]=h;return path
    pin(__file__);seal=read(pin(root/'complete.json'));assert seal['native_cases']==seal['cuda_cases']==32
    fixture=read(pin(root/'fixture-proof.json'));labels={i:Path(p['path']).stem.removesuffix('-plan') for i,p in enumerate(fixture['plans'])}
    assert set(labels.values())=={'source','repeat1','repeat2','repeat3'}
    assert len({m['model_sha256'] for m in fixture['models']})==1
    pool=read(pin(root/'pool/report.json',seal['pool_sha256']));assert pool['source_unchanged'] and len(pool['jobs'])==32
    groups={};cases=[]
    for job in pool['jobs']:
        label=labels[job['plan_index']];seed=job['seed'];folder=root/f'processing-{label}'/f"case-{job['job']:04d}"
        meta=read(pin(folder/'cuda/report.json'));v=read(pin(folder/'cuda/cuda-verification.json',meta['cuda_verification_sha256']));assert v['state']=='passed' and v['device']=='cuda'
        capture=read(pin(Path(job['root'])/'report.json'));r=capture['results'][0];assert capture['capture_complete'] and capture['provenance_valid'] and r['dispatch_valid']
        manifest=read(pin(Path(job['root'])/'manifest.json'));log=pin(Path(r['root'])/'server.log')
        with log.open(encoding='utf-8-sig') as stream:_,release,_=release_boundary(stream,manifest,seed)
        stop=r['goal_stop']['kill_frame'] if r['goal_stop'] else r['first_life']['end_frame']
        sp=next((folder/'native').glob('replay-*/steps.jsonl'));pin(sp,meta['source_sha256'][str(sp.resolve())])
        ss=[s for s in map(json.loads,sp.open(encoding='utf-8-sig')) if s['observation']['identity']['life']==1 and s['observation']['health']>0 and release<=s['observation']['identity']['frame']<stop]
        rows={s['observation']['identity']['frame']:s for s in ss};assert len(rows)==len(ss)
        ep=sp.with_name('server_outcomes.jsonl');pin(ep,meta['source_sha256'][str(ep.resolve())]);effects={e['step']:e for e in map(json.loads,ep.open(encoding='utf-8-sig'))}
        rp=pin(folder/'cuda/rollout.jsonl',meta['rollout_sha256']);features={r['frame']:r['features'] for r in map(json.loads,rp.open(encoding='utf-8-sig'))}
        g=groups.setdefault(seed,{});assert label not in g;g[label]=dict(rows=rows,effects=effects,features=features,release=release,stop=stop,map=r['generated_fixture']['map'])
    assert len(groups)==8
    for seed,g in sorted(groups.items()):
        base=g['source'];assert set(g)==set(labels.values())
        for label in ('repeat1','repeat2','repeat3'):
            other=g[label];forward={};reverse={};known=unknown=providers=0
            assert base['release']==other['release'] and base['stop']==other['stop'] and set(base['rows'])==set(other['rows']),(seed,label,'boundaries')
            for frame,x in sorted(base['rows'].items()):
                y=other['rows'][frame]
                assert x['owner']==y['owner'],(seed,label,frame,'owner')
                for key in ('observation','next_observation'):
                    assert physical_equal_with_entity_bijection(x[key],y[key],forward,reverse),(seed,label,frame,key)
                assert x.get('sample')==y.get('sample'),(seed,label,frame,'sample')
                assert x['sent_command']==y['sent_command'] and x['applied_action']==y['applied_action'],(seed,label,frame,'command')
                assert x['observation']['inventory_age_frames']==y['observation']['inventory_age_frames']==0,(seed,label,frame,'inventory_age')
                assert (frame in base['features'])==(frame in other['features'])
                if frame in base['features']:assert base['features'][frame]==other['features'][frame];providers+=1
                ex,ey=base['effects'].get(x['index']),other['effects'].get(y['index'])
                assert (ex is None)==(ey is None),(seed,label,frame,'effect availability')
                if ex is None or not ex['available']:
                    assert ey is None or not ey['available'];unknown+=1;continue
                assert ey['available']
                for key in ('monster_health_damage','received_health_damage','self_health_damage','teammate_health_damage','monster_kills','deaths'):assert ex[key]==ey[key],(seed,label,frame,key)
                known+=1
            cases.append(dict(seed=seed,map=base['map'],repeat=label,observed_frames=len(base['rows']),provider_features=providers,known_effect_frames=known,unknown_effect_frames=unknown,exact=True))
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);save(a.out/'report.json',dict(state='complete',version='combat_inventory_same_model_repeat_audit_v1',gate_passed=True,native_cases=32,cuda_cases=32,matched_pairs=len(cases),pairs=cases,inputs=pins,
        training_performed=False,neural_execution=False,new_games=False,promotion=False,
        scope='Exposed first-life observations, samples, features, commands and known aggregate per-step effects through task/death/budget boundary. Unknown effects remain unknown. Post-task capture tails and complete hidden simulation state are outside this proof; no combat-quality or model-improvement claim.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')));print(dict(pairs=len(cases),pair_frames=sum(c['observed_frames'] for c in cases),gate_passed=True))

if __name__=='__main__':main()
