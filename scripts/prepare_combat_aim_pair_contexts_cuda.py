"""Choose balanced multi-geometry TRAIN contexts and seal intervention heads."""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import re
from evaluate_registered_combat_models import read, save, sha
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import outcome
from ppo_combat import torch, training_devices, layers
from combat_joint_aim_policy import JointAim, events, validate
from combat_aim_candidates import choose, frame_uniforms, OFFSETS


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('baseline','out'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--exclude-proof',type=Path)
    ap.add_argument('--additional-per-geometry',type=int,default=3)
    ap.add_argument('--protocol',type=Path)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    closed=read(pin(a.baseline/'complete.json'));assert closed['native_cases']==closed['cuda_cases']==16
    spec=read(pin(a.baseline/'spec.json',closed['spec_sha256']));assert spec['split']=='train' and not spec['training_performed']
    pin(spec['protocol'],spec['protocol_sha256']);source=Path(spec['source']);seal=read(pin(source/'complete.json'))
    model=read(pin(source/'weights.json',seal['weights_sha256']));pool=read(pin(a.baseline/'pool/report.json',closed['pool_sha256']))
    processing=read(pin(a.baseline/'processing/report.json',closed['processing_sha256']));assert processing['device']=='cuda'
    plan=read(pin(a.baseline/'baseline-plan.json'));cases={};candidates=[];totals={}
    for job in pool['jobs']:
        assert not job['error'];folder=a.baseline/'processing'/f"case-{job['job']:04d}"
        meta=read(pin(folder/'cuda/report.json'));rows=list(map(json.loads,pin(folder/'cuda/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
        byframe={r['frame']:r for r in rows};steps=list(map(json.loads,pin(folder/'native/replay-0/steps.jsonl').open(encoding='utf-8-sig')))
        windows={};byframe_step={}
        for s in steps:
            o=s['observation'];n=s.get('native_step');d=s.get('server_execution') or {}
            if (s['owner']=='provider' and s.get('sample') and n and o['identity']['life']==1 and o['health']>0
                    and d.get('matched') and d.get('window_exclusive') and d.get('recovery_commands')==0):
                key=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence']);assert key not in windows
                windows[key]=s;byframe_step[o['identity']['frame']]=s
        capture=read(pin(Path(job['root'])/'report.json'));assert capture['capture_complete'] and capture['provenance_valid']
        result=capture['results'][0];assert result['generated_fixture']['split']=='train' and result['seed']==job['seed']
        log=pin(Path(result['root'])/'server.log');shots={}
        with log.open(encoding='utf-8-sig') as stream:
            for shot in scan(stream,set(windows)):
                if shot['eligible']:shots.setdefault(windows[shot['window']]['observation']['identity']['frame'],[]).append(shot)
        task=plan['tasks'][job['task_index']];episode=task['episode']['id'];match=re.search(r'campaign-(.+)-site-(\d+)-',episode);assert match
        geometry=match.group(1)+'-site-'+match.group(2);maximum=0.;possible=0
        for row in rows:
            event=row['sample'].get('aim_candidate')
            if not event:continue
            assert event['candidate']==0,'Baseline made a rare nonzero correction; inspect before pairing'
            gate,_=frame_uniforms(row['sample']['sampling_seed'],row['frame'],1);previous=maximum;maximum=max(maximum,gate)
            if gate-previous<1e-5 or row['frame'] not in byframe_step or len(shots.get(row['frame'],[]))!=1:continue
            shot=shots[row['frame']][0];label=outcome(shot)
            if label not in ('live_monster_damage','solid_contact'):continue
            possible+=1;candidates.append(dict(job=job['job'],seed=job['seed'],episode=episode,geometry=geometry,frame=row['frame'],
                baseline_outcome=label,baseline_shot=shot,threshold=(previous+gate)/2,prior_gate_max=previous,current_gate=gate,
                instance=task['instances'][task['seeds'].index(job['seed'])]))
        totals[str(job['seed'])]=dict(geometry=geometry,candidate_states=possible,eligible_shots=sum(map(len,shots.values())))
        cases[job['job']]=dict(rows=rows,byframe=byframe)
    groups=sorted({c['geometry'] for c in candidates});assert len(groups)==4,('Missing geometry',groups,totals)
    options={g:{label:sorted((c for c in candidates if c['geometry']==g and c['baseline_outcome']==label),key=lambda c:(c['seed'],c['frame']))
                for label in ('live_monster_damage','solid_contact')} for g in groups}
    combinations=[labels for labels in itertools.product(('live_monster_damage','solid_contact'),repeat=4)
                  if all(options[g][label] for g,label in zip(groups,labels))]
    assert combinations,'No state for a geometry'
    labels=min(combinations,key=lambda v:(abs(v.count('live_monster_damage')-2),v))
    selected=[copy.deepcopy(options[g][label][0]) for g,label in zip(groups,labels)]
    exclusions=[]
    if a.exclude_proof:
        assert a.protocol and a.additional_per_geometry==3
        old_seal=read(pin(a.exclude_proof/'complete.json'));old=read(pin(a.exclude_proof/'report.json',old_seal['report_sha256']))
        pin(a.exclude_proof/'labels.jsonl',old_seal['labels_sha256'])
        assert old['accepted_variants']==20 and old['label_contexts']==4
        assert old['inputs'][str((source/'weights.json').resolve())]==seal['weights_sha256']
        exclusions=[dict(seed=c['seed'],frame=c['frame'],geometry=c['geometry'],baseline_outcome=c['baseline_selection_outcome']) for c in old['contexts']]
        excluded={(c['seed'],c['frame']) for c in exclusions};selected=[]
        for g in groups:
            prior=[c for c in exclusions if c['geometry']==g];assert len(prior)==1
            chosen=list(prior)
            for _ in range(a.additional_per_geometry):
                eligible=[c for c in candidates if c['geometry']==g and (c['seed'],c['frame']) not in excluded
                    and all((c['seed'],c['frame'])!=(s['seed'],s['frame']) for s in chosen)
                    and sum(s['seed']==c['seed'] for s in chosen)<2]
                assert eligible,('Insufficient distinct states',g,chosen)
                counts={label:sum(s['baseline_outcome']==label for s in chosen) for label in ('live_monster_damage','solid_contact')}
                c=min(eligible,key=lambda c:(counts[c['baseline_outcome']],sum(s['seed']==c['seed'] for s in chosen),c['seed'],c['frame']))
                chosen.append(c);selected.append(copy.deepcopy(c))
        spec=dict(spec,protocol=str(pin(a.protocol)),protocol_sha256=sha(a.protocol))
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    joint=JointAim(model).requires_grad_(False);a.out.mkdir(parents=True);variants=[]
    for ci,c in enumerate(selected):
        rows=[r for r in cases[c['job']]['rows'] if r['frame']<=c['frame']];assert rows[-1]['frame']==c['frame']
        x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
        cat=lambda n:torch.tensor([r['sample'].get(n,0) for r in rows],device='cuda',dtype=torch.long)
        z,_,timing,active=events(rows,x,cat('target'),cat('aim_mode'),cat('vertical'),cat('weapon'))
        captured=[];handle=joint.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
        with torch.inference_mode():joint(x,z,cat('target'),cat('aim_mode'),cat('vertical'),cat('weapon'),timing)
        handle.remove();hints=captured[0][-1,-27:-2].double();assert hints.shape==(25,)
        top=[k for k in torch.argsort(hints,descending=True).tolist() if k][:2]
        others=sorted((k for k in range(1,25) if k not in top),key=lambda k:hashlib.sha256(f"{c['seed']}:{c['frame']}:{k}".encode()).hexdigest())[:2]
        c.update(context_id=('extra-' if a.exclude_proof else 'context-')+f'{ci:02d}',categories=[0,*top,*others],predictor_top_two=top,other_two=others,
            predicted_hit_probabilities=hints.sigmoid().tolist(),baseline_reference=str(a.baseline.resolve()))
        uniforms=torch.tensor([frame_uniforms(r['sample']['sampling_seed'],r['frame'],1) for r in rows],device='cuda',dtype=torch.float64)
        for k in c['categories']:
            p0=.999999 if k==0 else c['threshold'];q=torch.full((25,),(1-p0)/24 if k==0 else (1-p0)*1e-8,device='cuda',dtype=torch.float64);q[0]=p0
            if k:q[k]=(1-p0)*(1-23e-8)
            local=JointAim(model).requires_grad_(False)
            with torch.no_grad():local.head.layers[-1].weight.zero_();local.head.layers[-1].bias.copy_(q.log().float())
            updated=copy.deepcopy(model);updated['aim_candidate_head']['layers']=layers(local.head.layers);validate(updated)
            exported=JointAim(updated).requires_grad_(False)
            with torch.inference_mode():
                logits=exported.head(torch.zeros((len(rows),1359),device='cuda')).double()
                picked=choose(logits,uniforms[:,0],uniforms[:,1],active)
                assert bool((picked[:-1]==0).all()) and int(picked[-1])==k,(ci,k,picked.tolist())
            directory=a.out/'models'/f'context-{ci:02d}-category-{k:02d}';directory.mkdir(parents=True);save(directory/'weights.json',updated)
            variants.append(dict(context_id=c['context_id'],category=k,model=str((directory/'weights.json').resolve()),model_sha256=sha(directory/'weights.json'),
                episode=c['episode'],seed=c['seed'],offset=spec['offset']+plan['tasks'][next(j['task_index'] for j in pool['jobs'] if j['job']==c['job'])]['seeds'].index(c['seed']),
                frame=c['frame'],reference_job=c['job'],instance=c['instance'],offset_yaw_degrees=OFFSETS[k//5],offset_pitch_degrees=OFFSETS[k%5]))
    for p,h in pins.items():assert sha(Path(p))==h,p
    save(a.out/'report.json',dict(state='complete',version='combat_aim_pair_context_preparation_v1',device='cuda',baseline=str(a.baseline.resolve()),
        registry=spec['registry'],source=str(source.resolve()),source_weights_sha256=seal['weights_sha256'],protocol=spec['protocol'],protocol_sha256=spec['protocol_sha256'],
        candidate_states=len(candidates),case_counts=totals,contexts=selected,variants=variants,inputs=pins,script_sha256=sha(__file__),
        excluded_existing_contexts=exclusions,existing_proof=str(a.exclude_proof.resolve()) if a.exclude_proof else None,
        requested_balance_met=sum(c['baseline_outcome']=='live_monster_damage' for c in selected)*2==len(selected),training_performed=False,validation_used=False,promotion=False,
        scope='Targeted TRAIN state selection, four geometries; CUDA prefix category simulation only. Actual native state matching required before causal labels.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps(dict(candidate_states=len(candidates),balance_met=sum(c['baseline_outcome']=='live_monster_damage' for c in selected)*2==len(selected),
        contexts=[{k:c[k] for k in ('context_id','geometry','seed','frame','baseline_outcome','categories')} for c in selected]),indent=2))


if __name__=='__main__':main()
