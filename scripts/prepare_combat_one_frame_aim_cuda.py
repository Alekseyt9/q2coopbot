"""Seal two one-frame diagnostic models; preserve the entire baseline head/body."""
import argparse
import copy
import json
import re
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from ppo_combat import torch, training_devices
from ppo_recurrent import durable_write
from combat_joint_aim_policy import JointAim, base_model, validate, events, PROBE_VERSION
from combat_aim_candidates import choose, frame_uniforms
from fork_conditional_weapon_exploration_cuda import equal_tree
from train_combat_joint_aim_cuda import cuda_tree
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import outcome


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('baseline','protocol','out'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--coverage',type=Path)
    ap.add_argument('--seed',type=int);ap.add_argument('--frame',type=int)
    ap.add_argument('--holdout-dataset',type=Path)
    ap.add_argument('--categories',default='0,4')
    a=ap.parse_args();assert not a.out.exists();pins={}
    categories=list(map(int,a.categories.split(',')));assert categories[0]==0 and len(set(categories))==len(categories) and all(0<=k<25 for k in categories)
    assert 2<=len(categories)<=25
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    closed=read(pin(a.baseline/'complete.json'));spec=read(pin(a.baseline/'spec.json',closed['spec_sha256']))
    source=Path(spec['source']);seal=read(pin(source/'complete.json'))
    for n,k in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(source/n,seal[k])
    model=read(source/'weights.json');assert not model.get('diagnostic_joint_aim_probe')
    pool=read(pin(a.baseline/'pool/report.json',closed['pool_sha256']))
    explicit=a.seed is not None or a.frame is not None
    if explicit:
        assert a.seed is not None and a.frame is not None and a.frame>0 and a.holdout_dataset
        holdseal=read(pin(a.holdout_dataset/'complete.json'));hold=read(pin(a.holdout_dataset/'report.json',holdseal['report_sha256']))
        assert hold['source_weights_sha256']==seal['weights_sha256'] and a.seed not in hold['heldout_seeds'],'Development seed cannot acquire TRAIN labels'
        job=next(j for j in pool['jobs'] if j['seed']==a.seed);c=dict(seed=a.seed,frame=a.frame,job=job['job'])
    else:
        assert a.coverage
        coverage_seal=read(pin(a.coverage/'complete.json'));coverage=read(pin(a.coverage/'report.json',coverage_seal['report_sha256']))
        c=coverage['smoke_candidate'];assert c['seed']==1172065112 and c['frame']==110 and c['constant_gate_cannot_first_intervene']
        job=next(j for j in pool['jobs'] if j['job']==c['job'])
    assert job['seed']==c['seed'];plan=read(pin(a.baseline/'baseline-plan.json'));task=plan['tasks'][job['task_index']]
    assert task['split']=='train';c['episode']=task['episode']['id'];frame=c['frame']
    site=re.search(r'campaign-(.+)-site-(\d+)-',c['episode']);assert site
    geometry=site.group(1)+'-site-'+site.group(2);context_id=f"one-frame-{c['seed']}-{frame}" if explicit else 'one-frame-smoke'
    instance=task['instances'][task['seeds'].index(c['seed'])]
    folder=a.baseline/'processing'/f"case-{job['job']:04d}"/'cuda';meta=read(pin(folder/'report.json'))
    rows=list(map(json.loads,pin(folder/'rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')));pin(a.protocol)
    steps=folder.parent/'native/replay-0/steps.jsonl';step=next(s for s in map(json.loads,pin(steps).open(encoding='utf-8-sig')) if s['observation']['identity']['frame']==frame and s['owner']=='provider')
    dispatch=step.get('server_execution') or {};assert step['observation']['identity']['life']==1 and dispatch.get('matched') and dispatch.get('window_exclusive') and dispatch.get('recovery_commands')==0
    capture=read(pin(Path(job['root'])/'report.json'));log=pin(Path(capture['results'][0]['root'])/'server.log');n=step['native_step']
    key=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence'])
    with log.open(encoding='utf-8-sig') as stream:shots=[q for q in scan(stream,{key}) if q['eligible']]
    assert len(shots)==1,'Selected frame needs exactly one actual eligible MG shot'
    baseline_outcome=outcome(shots[0])
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    cp=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert cp['weights_sha256']==seal['weights_sha256']
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):ordinary=JointAim(model).requires_grad_(False)
    x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
    cat=lambda n:torch.tensor([r['sample'].get(n,0) for r in rows],device='cuda',dtype=torch.long)
    z,old_choice,timing,active=events(rows,x,cat('target'),cat('aim_mode'),cat('vertical'),cat('weapon'))
    frames=torch.tensor([r['frame'] for r in rows],device='cuda',dtype=torch.long);at=frames==frame
    assert int(at.sum())==1 and bool(active[at].all()) and bool((old_choice[at]==0).all())
    uniforms=torch.tensor([frame_uniforms(r['sample']['sampling_seed'],r['frame'],1) for r in rows],device='cuda',dtype=torch.float64)
    args=(x,z,cat('target'),cat('aim_mode'),cat('vertical'),cat('weapon'),timing)
    captured=[];handle=ordinary.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
    with torch.inference_mode():original_logits,_=ordinary(*args)
    handle.remove();hints=captured[0][at][0,-27:-2].double().sigmoid().tolist()
    a.out.mkdir(parents=True);variants=[]
    for k in categories:
        logits=model['aim_candidate_head']['layers'][-1]['bias']
        if k:
            q=torch.full((25,),1e-8,device='cuda',dtype=torch.float64);q[k]=1-24e-8;logits=q.log().float().tolist()
        probe=dict(version=PROBE_VERSION,frame=frame,logits=logits);updated=copy.deepcopy(model);updated['diagnostic_joint_aim_probe']=probe
        rejected=[]
        for fn in (base_model,validate,JointAim):
            try:fn(updated)
            except AssertionError:rejected.append(fn.__name__)
            else:raise AssertionError('Ordinary entrypoint accepted probe: '+fn.__name__)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):module=JointAim(updated,allow_probe=True).requires_grad_(False)
        with torch.inference_mode():
            actual,mask=module(*args,frames=frames);assert torch.equal(mask,active)
            assert torch.equal(actual[~at],original_logits[~at]),'Baseline changed outside selected frame'
            picked=choose(actual.double(),uniforms[:,0],uniforms[:,1],active)
            expected=old_choice.clone();expected[at]=k;assert torch.equal(picked,expected)
        try:module(*args)
        except AssertionError:pass
        else:raise AssertionError('Probe accepted scoring without recorded frame')
        assert equal_tree(ordinary.head.state_dict(),module.head.state_dict()) and equal_tree(ordinary.hints.state_dict(),module.hints.state_dict())
        stripped=copy.deepcopy(updated);stripped.pop('diagnostic_joint_aim_probe');assert stripped==model
        directory=a.out/'models'/f'category-{k:02d}';directory.mkdir(parents=True);save(directory/'weights.json',updated)
        fork=copy.deepcopy(cp);fork.update(weights_sha256=sha(directory/'weights.json'),
            actor_training={**cp['actor_training'],'scope':PROBE_VERSION},
            diagnostic_joint_aim_probe=dict(probe,source_weights_sha256=seal['weights_sha256'],protocol_sha256=sha(a.protocol)))
        durable_write(directory/'checkpoint.pt',lambda f:torch.save(fork,f));loaded=torch.load(directory/'checkpoint.pt',map_location='cuda',weights_only=True)
        for key in cp:
            if key not in ('weights_sha256','actor_training'):assert equal_tree(cp[key],cuda_tree(loaded[key])),key
        save(directory/'report.json',dict(state='complete',version=PROBE_VERSION,device='cuda',diagnostic_only=True,training_performed=False,promotion=False,
            source=str(source.resolve()),source_weights_sha256=seal['weights_sha256'],only_metadata_changed=True,all_neural_and_optimizer_rng_history_exact=True,
            ordinary_entrypoints_rejected=rejected,missing_frame_rejected=True,reference_choices_verified=True))
        save(directory/'complete.json',dict(state='complete',weights_sha256=sha(directory/'weights.json'),checkpoint_sha256=sha(directory/'checkpoint.pt'),report_sha256=sha(directory/'report.json')))
        variants.append(dict(context_id=context_id,category=k,model=str((directory/'weights.json').resolve()),model_sha256=sha(directory/'weights.json'),
            episode=c['episode'],seed=c['seed'],frame=frame,reference_job=c['job'],offset=spec['offset']+task['seeds'].index(c['seed']),instance=instance,
            offset_yaw_degrees=(0,-5,-2.5,2.5,5)[k//5],offset_pitch_degrees=(0,-5,-2.5,2.5,5)[k%5]))
    for p,h in pins.items():assert sha(Path(p))==h,p
    save(a.out/'report.json',dict(state='complete',version='combat_one_frame_aim_preparation_v1',device='cuda',diagnostic_only=True,
        baseline=str(a.baseline.resolve()),source=str(source.resolve()),source_weights_sha256=seal['weights_sha256'],registry=spec['registry'],
        protocol=str(a.protocol.resolve()),protocol_sha256=sha(a.protocol),
        contexts=[dict(context_id=context_id,geometry=geometry,seed=c['seed'],frame=frame,job=c['job'],baseline_outcome=baseline_outcome,
            episode=c['episode'],predicted_hit_probabilities=hints)],
        variants=variants,inputs=pins,script_sha256=sha(__file__),training_performed=False,validation_used=False,promotion=False))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')));print('CUDA one-frame forks sealed; metadata-only change, reference categories verified')


if __name__=='__main__':main()
