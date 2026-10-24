"""Prepare fresh H1 forecast audit rows and observable past-damage contexts."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or expected==h,str(path);pins[key]=h;return path
    p=read(pin(a.protocol));assert p['version'] in ('combat_enemy_threat_fresh_corpus_v1','combat_underfire_teacher_corpus_v1')
    teacher=p['version']=='combat_underfire_teacher_corpus_v1'
    if teacher:assert p['world_roles']==dict(train_slots=[0,1,2],development_slots=[3])
    closed=read(pin(a.corpus/'complete.json'));assert closed['native_cases']==closed['cuda_cases']==32
    collection=read(pin(a.corpus/'protocol.json',closed['protocol_sha256']));assignment=collection['assignments'][0]
    assert len(collection['assignments'])==1 and assignment['split']=='train' and assignment['count']==4
    plan=read(pin(assignment['plan'],assignment['plan_sha256']));assert plan['model_sha256']==p['source_weights_sha256']
    pool=read(pin(a.corpus/'pool/report.json',closed['pool_sha256']));assert pool['source_unchanged'] and len(pool['jobs'])==32
    if teacher:
        prior=Path(p['excluded_dataset']);prior_seal=read(pin(prior/'complete.json'))
        prior_report=read(pin(prior/'report.json',prior_seal['report_sha256']))
        assert not {j['seed'] for j in pool['jobs']}.intersection(prior_report['worlds']+p['excluded_world_seeds'])
        names=read(pin(p['episodes_source']))['episodes']
        assert {t['episode']['id'] for t in plan['tasks']}==set(names)
    a.out.mkdir(parents=True);(a.out/'chunks').mkdir();chunks=[];contexts=[];coverage=[];unknown_rows=[];totals=collections.Counter()
    for job in sorted(pool['jobs'],key=lambda j:j['job']):
        task=plan['tasks'][job['task_index']];seed=job['seed'];folder=a.corpus/'processing-train'/f"case-{job['job']:04d}"
        slot=task['seeds'].index(seed)
        split=('train' if slot in p['world_roles']['train_slots'] else 'development') if teacher else 'fresh_audit'
        meta=read(pin(folder/'cuda/report.json'));assert meta['numerical_verification']=='cuda_verified_v1' and meta['model_sha256']==p['source_weights_sha256'] and meta['observation_timing_contract']=='frozen_world_v1'
        proof=read(pin(folder/'cuda/cuda-verification.json',meta['cuda_verification_sha256']));assert proof['state']=='passed' and proof['device']=='cuda'
        sp=next((folder/'native').glob('replay-*/steps.jsonl'))
        steps=list(map(json.loads,pin(sp,meta['source_sha256'][str(sp.resolve())]).open(encoding='utf-8-sig')))
        ep=sp.with_name('server_outcomes.jsonl');effects={e['step']:e for e in map(json.loads,pin(ep,meta['source_sha256'][str(ep.resolve())]).open(encoding='utf-8-sig'))}
        rows=list(map(json.loads,pin(folder/'cuda/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
        byframe={s['observation']['identity']['frame']:s for s in steps if s['observation']['identity']['life']==1}
        active={f:s for f,s in byframe.items() if s['owner']=='provider' and s.get('sample') and s.get('native_step')}
        first=min(active);values=[];candidates=[]
        for row in rows:
            frame=row['frame'];s=active.get(frame)
            if s is None:continue
            o=s['observation'];dispatch=s.get('server_execution') or {};ns=s['native_step'];effect=effects.get(s['index'])
            if o['health']<=0 or not o.get('enemies'):continue
            if not(dispatch.get('matched') and dispatch.get('window_exclusive') and dispatch.get('recovery_commands')==0 and ns['begin_frame']==frame and ns['end_frame']==frame+1):continue
            if effect is None or not effect['available'] or not s.get('next_observation'):continue
            enemies=sorted(o['enemies'],key=lambda e:e['distance'])[:8];slots={e['id']:(i,e['class']) for i,e in enumerate(enemies)}
            incoming=np.zeros(8,dtype=np.float32);unknown=0.;damage_sum=0.
            for e in effect['events']:
                if e['target']!=o['identity']['actor']:continue
                amount=min(max(e['health_before'],0),e['take']);assert amount>=0;damage_sum+=amount
                if e['attacker'] in slots and slots[e['attacker']][1]==e['attacker_class']:incoming[slots[e['attacker']][0]]+=amount
                else:
                    unknown+=amount
                    if amount>0:
                        projectiles=o.get('visible_projectiles') or [];remembered=o.get('remembered_threats') or []
                        unknown_rows.append(dict(seed=seed,map=task['episode']['map'],frame=frame,health_damage=amount,
                            native_attacker_class=e['attacker_class'],attacker_in_memory=any(t['id']==e['attacker'] for t in remembered),
                            observed_projectile_classes=[e['class'] for e in projectiles],observed_enemy_classes=[e['class'] for e in enemies],
                            observed_memory_classes=[e['class'] for e in remembered]))
            assert damage_sum==effect['received_health_damage']==float(incoming.sum()+unknown)
            sample=s['sample'];candidate=sample.get('aim_candidate');source=candidate['source_latent'] if candidate else sample['latent']
            target=np.zeros((4,6),dtype=np.float32);target[0]=[effect[k] for k in ('monster_health_damage','received_health_damage','self_health_damage','teammate_health_damage','monster_kills','deaths')]
            values.append(dict(features=row['features'],source_action=source,action=sample['latent'],attack=float(sample['attack']),
                categories=[sample.get(k,0) for k in ('target','aim_mode','vertical','weapon')],option=[0.,0.,0.],effects=target,known=[True,False,False,False],
                incoming_by_enemy=incoming,unknown_incoming=unknown,world=seed,frame=frame))
            totals['rows']+=1;totals['received_health_damage']+=damage_sum;totals['unknown_health_damage']+=unknown;totals['positive_rows']+=damage_sum>0
            prev=active.get(frame-1)
            if frame>=first+10 and prev and prev['observation']['identity']['actor']==o['identity']['actor'] and prev['observation']['identity']['spawncount']==o['identity']['spawncount']:
                drop=prev['observation']['health']-o['health']
                if drop>0 and o['on_ground'] and sample.get('vertical',0)==0 and candidate:
                    candidates.append(dict(frame=frame,observed_health_drop=drop,observed_enemies=len(enemies)))
        assert values,'No known visible-enemy source rows'
        types=dict(features=np.float32,source_action=np.float32,action=np.float32,attack=np.float32,categories=np.int64,option=np.float32,
            effects=np.float32,known=np.bool_,incoming_by_enemy=np.float32,unknown_incoming=np.float32,world=np.int64,frame=np.int32)
        data={k:np.asarray([v[k] for v in values],dtype=dtype) for k,dtype in types.items()}
        assert data['features'].shape[1]==1147 and np.array_equal(data['incoming_by_enemy'].sum(1)+data['unknown_incoming'],data['effects'][:,0,1])
        path=a.out/'chunks'/f"{job['job']:04d}-fresh.npz";np.savez_compressed(path,**data)
        chunks.append(dict(path=str(path.resolve()),sha256=sha(path),split=split,rows=len(values),case=dict(seed=seed,map=task['episode']['map'],folder=str(folder.resolve()))))
        coverage.append(dict(seed=seed,map=task['episode']['map'],split=split,eligible_contexts=len(candidates)))
        if candidates:
            root=min(candidates,key=lambda c:c['frame']);slot=task['seeds'].index(seed)
            contexts.append(dict(**root,context_id=f"underfire-{seed}-{root['frame']}",seed=seed,job=job['job'],split=split if teacher else 'train',role='auxiliary_'+split if teacher else 'fresh_audit_not_fitting',
                episode=task['episode']['id'],instance=task['instances'][slot],offset=assignment['offset']+slot,directory=str(folder.resolve())))
        save(a.out/'progress.json',dict(stage='building',cases_done=len(chunks),cases=32,rows=totals['rows']))
    for path,h in pins.items():assert sha(path)==h,path
    save(a.out/'contexts.json',dict(contexts=contexts,coverage=coverage));save(a.out/'unknown-attacks.json',unknown_rows)
    report=dict(state='complete',version='combat_underfire_teacher_dataset_v1' if teacher else 'combat_enemy_threat_fresh_dataset_v1',protocol_sha256=sha(a.protocol),chunks=chunks,totals=dict(totals),
        worlds=sorted(j['seed'] for j in pool['jobs']),cases=32,source_weights_sha256=p['source_weights_sha256'],contexts=len(contexts),
        contexts_sha256=sha(a.out/'contexts.json'),unknown_attacks_sha256=sha(a.out/'unknown-attacks.json'),inputs=pins,
        training_performed=False,actor_training=False,ordinary_training_eligible=False,neural_execution=False,promotion=False,
        scope=('Independent auxiliary source corpus with fixed whole-world TRAIN/development roles; no fitting yet. ' if teacher else 'Fresh alternate-site forecast/coverage audit only. ')+
            'H1 targets verified; remaining horizon slots masked, not invented. Roots selected from past observed health decrease, never future outcomes. No predictor fitting.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',rows=totals['rows'],contexts=len(contexts)))
    print(dict(totals=dict(totals),worlds=32,contexts=len(contexts),context_maps=sorted({c['instance']['map'] for c in contexts})))


if __name__=='__main__':main()
