"""Stream verified native windows into compressed auxiliary-supervision chunks."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or expected==h,str(path);pins[key]=h;return path
    p=read(pin(a.protocol));assert p['version']=='combat_local_effect_predictor_experiment_v1'
    source=read(pin(Path(p['source'])/'weights.json',p['source_weights_sha256']))
    holdout=set(p['challenge_world_seeds']);horizons=p['horizons_frames'];fields=p['effect_fields']
    assert horizons==[1,3,5,10] and len(holdout)==4 and len(fields)==6
    a.out.mkdir(parents=True);(a.out/'chunks').mkdir();cases=[];world_splits={}
    baseline=Path(p['baseline']);closed=read(pin(baseline/'complete.json'))
    bp=read(pin(baseline/'protocol.json',closed['protocol_sha256']));assignment=bp['assignments'][0]
    plan=read(pin(assignment['plan'],assignment['plan_sha256']))
    pool=read(pin(baseline/'pool/report.json',closed['pool_sha256']))
    for job in pool['jobs']:
        task=plan['tasks'][job['task_index']];seed=job['seed'];slot=task['seeds'].index(seed)
        split='development' if slot==3 else 'train';world_splits[seed]=split
        if seed not in holdout:cases.append(dict(split=split,seed=seed,map=task['episode']['map'],model=Path(p['source'])/'weights.json',
            model_sha256=p['source_weights_sha256'],folder=baseline/'processing-train'/f"case-{job['job']:04d}",capture=Path(job['root'])/'report.json'))
    movement=Path(p['movement_collection']);mc=read(pin(movement/'complete.json'))
    ms=read(pin(movement/'spec.json',mc['spec_sha256']));mp=read(pin(movement/'pool/report.json',mc['pool_sha256']))
    jobs={j['plan_index']:j for j in mp['jobs']}
    for unit in ms['contexts']:
        context=unit['context'];seed=context['seed'];split=context['split']
        assert world_splits[seed]==split
        if seed in holdout:continue
        for variant in unit['variants']:
            i=variant['plan_index'];cases.append(dict(split=split,seed=seed,map=context['instance']['map'],model=Path(variant['model']),
                model_sha256=variant['model_sha256'],folder=movement/f'case-{i:02d}',capture=Path(jobs[i]['root'])/'report.json'))
    chunks=[];counts=collections.Counter();known_counts=collections.Counter();terminal_counts=collections.Counter()
    def arrays(records):
        assert records
        keys=('features','source_action','action','attack','categories','option','effects','known','terminal_kind','world','frame')
        dtypes=(np.float32,np.float32,np.float32,np.float32,np.int64,np.float32,np.float32,np.bool_,np.uint8,np.int64,np.int32)
        return {k:np.asarray([r[k] for r in records],dtype=d) for k,d in zip(keys,dtypes)}
    def root_input(features,sample,probe,frame):
        option=[0.,0.,0.]
        if probe:
            duration=probe.get('duration_frames',1);remaining=probe['frame']+duration-frame
            if probe['frame']<=frame<probe['frame']+duration and any(probe['mean_shift']):option=[*probe['mean_shift'],remaining]
        candidate=sample.get('aim_candidate') or {}
        return dict(features=features,source_action=candidate.get('source_latent') or sample['latent'],action=sample['latent'],
            attack=float(sample['attack']),categories=[sample.get(k,0) for k in ('target','aim_mode','vertical','weapon')],option=option)
    def write_chunk(split,records,case_info):
        if not records:return
        data=arrays(records);assert data['features'].shape==(len(records),1147)
        assert data['effects'].shape==(len(records),4,6) and np.isfinite(data['features']).all() and np.isfinite(data['effects']).all()
        path=a.out/'chunks'/f'{len(chunks):04d}-{split}.npz';np.savez_compressed(path,**data)
        chunks.append(dict(path=str(path.resolve()),sha256=sha(path),rows=len(records),split=split,case=case_info))
        counts[split]+=len(records)
        for hindex,h in enumerate(horizons):
            known_counts[f'{split}_h{h}']+=int(data['known'][:,hindex].sum())
            terminal_counts[f'{split}_h{h}_goal']+=int((data['terminal_kind'][:,hindex]==1).sum())
            terminal_counts[f'{split}_h{h}_death']+=int((data['terminal_kind'][:,hindex]==2).sum())
    for ci,case in enumerate(cases):
        model=read(pin(case['model'],case['model_sha256']));probe=model.pop('diagnostic_movement_probe',None)
        assert model==source,'Unapproved behavior policy in forecast corpus'
        folder=case['folder'];meta=read(pin(folder/'cuda/report.json'))
        assert meta['model_sha256']==case['model_sha256'] and meta['numerical_verification']=='cuda_verified_v1'
        assert meta['feature_version']=='combat_features_v10' and meta['observation_timing_contract']=='frozen_world_v1'
        proof=read(pin(folder/'cuda/cuda-verification.json',meta['cuda_verification_sha256']));assert proof['state']=='passed' and proof['device']=='cuda'
        path=next((folder/'native').glob('replay-*/steps.jsonl'))
        steps=list(map(json.loads,pin(path,meta['source_sha256'][str(path.resolve())]).open(encoding='utf-8-sig')))
        ep=path.with_name('server_outcomes.jsonl')
        effect_list=list(map(json.loads,pin(ep,meta['source_sha256'][str(ep.resolve())]).open(encoding='utf-8-sig')))
        effects={e['step']:e for e in effect_list};assert len(effects)==len(effect_list)
        rows=list(map(json.loads,pin(folder/'cuda/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
        active={s['observation']['identity']['frame']:s for s in steps if s['owner']=='provider' and s.get('native_step') and s['observation']['identity']['life']==1}
        capture=read(pin(case['capture']));assert capture['capture_complete'] and capture['provenance_valid']
        result=capture['results'][0];assert result['seed']==case['seed']
        goal=result['goal_stop'];goal_end=goal['kill_frame'] if goal else None
        records=[]
        for row in rows:
            frame=row['frame'];s=active.get(frame)
            if s is None or not s['observation'].get('enemies') or s['observation']['health']<=0:continue
            if goal_end is not None and frame>=goal_end:continue
            if probe and frame<probe['frame']<frame+max(horizons):continue
            y=[];valid=[];terminals=[]
            for horizon in horizons:
                chunks_=[];known=True;terminal=0
                for f in range(frame,frame+horizon):
                    step=active.get(f)
                    if step is None:known=False;break
                    ns=step['native_step'];dispatch=step.get('server_execution') or {}
                    if not(dispatch.get('matched') and dispatch.get('window_exclusive') and dispatch.get('recovery_commands')==0 and ns['begin_frame']==f and ns['end_frame']==f+1):known=False;break
                    effect=effects.get(step['index']);nxt=step.get('next_observation')
                    if effect is None or not effect['available'] or nxt is None or nxt['identity']['frame']!=f+1:known=False;break
                    chunks_.append(effect)
                    if effect['deaths']>0:terminal=2;break
                    if goal_end is not None and f+1==goal_end:terminal=1;break
                    if nxt['identity']['life']!=1:known=False;break
                valid.append(known);terminals.append(terminal if known else 0)
                # Masked slots are storage placeholders, never training targets.
                y.append([sum(e[k] for e in chunks_) if known else 0. for k in fields])
            if not any(valid):continue
            records.append(dict(**root_input(row['features'],s['sample'],probe,frame),effects=y,known=valid,terminal_kind=terminals,world=case['seed'],frame=frame))
        write_chunk(case['split'],records,dict(folder=str(folder.resolve()),seed=case['seed'],map=case['map']))
        save(a.out/'progress.json',dict(stage='building',cases_done=ci+1,cases=len(cases),rows=dict(counts)))
    challenge=Path(p['challenge']);cc=read(pin(challenge/'complete.json'));cr=read(pin(challenge/'report.json',cc['report_sha256']))
    assert cr['root_and_h1_sanity_passed'] and cr['contexts']==4
    roots=read(pin(challenge/'records.json',cr['artifacts']['records']));challenge_records=[]
    for r in roots:
        assert r['world_seed'] in holdout
        x=r['root_input'];sample=dict(aim_candidate=dict(source_latent=x['source_latent']),latent=x['final_latent'],attack=x['attack'],**x['categories'])
        probe=dict(frame=r['frame'],duration_frames=r['duration'],mean_shift=x['movement_shift'])
        known=[r['measurements'][str(h)]['known'] for h in horizons]
        y=[[r['measurements'][str(h)]['effects'][k] if known[i] else 0. for k in fields] for i,h in enumerate(horizons)]
        challenge_records.append(dict(**root_input(x['features'],sample,probe,r['frame']),effects=y,known=known,terminal_kind=[0]*4,world=r['world_seed'],frame=r['frame']))
    write_chunk('challenge',challenge_records,dict(root_records=str((challenge/'records.json').resolve()),root_records_sha256=sha(challenge/'records.json')))
    worlds={split:sorted({c['seed'] for c in cases if c['split']==split}) for split in ('train','development')}
    assert set(worlds['train']).isdisjoint(worlds['development']) and not(holdout & set(worlds['train']+worlds['development']))
    assert len(worlds['train'])==20 and len(worlds['development'])==8 and counts['challenge']==160
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_local_effect_prediction_dataset_v1',protocol_sha256=sha(a.protocol),source_weights_sha256=p['source_weights_sha256'],
        rows=dict(counts),known=dict(known_counts),terminal_stops=dict(terminal_counts),worlds={**worlds,'challenge':sorted(holdout)},chunks=chunks,
        compressed_bytes=sum(Path(c['path']).stat().st_size for c in chunks),inputs=pins,feature_width=1147,model_input_width=1185,
        horizons=horizons,effect_fields=fields,actor_training=False,ordinary_training_eligible=False,new_games=False,neural_execution=False,
        scope='Auxiliary native-effect supervision only. Whole-world splits and observed stopped-terminal targets; unknown windows masked. Challenge worlds excluded from all fitting/development. No ordinary PPO admission or runtime promotion.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',rows=dict(counts),compressed_bytes=report['compressed_bytes']))
    print(dict(rows=dict(counts),known=dict(known_counts),worlds=report['worlds'],compressed_bytes=report['compressed_bytes']))


if __name__=='__main__':main()
