"""Select observable TRAIN danger contexts and seal a one-context movement smoke."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
from evaluate_registered_combat_models import read, save, sha
from ppo_combat import torch, training_devices
from ppo_recurrent import prepare_context, sequence
from combat_joint_aim_policy import JointAim, base_model, probabilities, events
from combat_movement_probe import VERSION, shifts, validate as validate_movement
from combat_precision_head import selected_means
from migrate_combat_navigation_cuda import branch
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','protocol','out'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--context-id', help='Explicit context from the observable-only index')
    ap.add_argument('--list-only', action='store_true', help='Seal context quotas without executing neural networks')
    a=ap.parse_args(); assert not a.out.exists(); pins={}

    def pin(path,expected=None):
        path=Path(path).resolve(); h=sha(path); assert expected is None or expected==h,str(path)
        pins[str(path)]=h; return path

    closed=read(pin(a.corpus/'complete.json')); assert closed['native_cases']==closed['cuda_cases']==32
    collection=read(pin(a.corpus/'protocol.json',closed['protocol_sha256'])); assert collection['train_only'] and not collection.get('diagnostic_intervention')
    pin(a.protocol)
    collection_protocol=read(a.protocol) if a.protocol.suffix=='.json' else None
    batch_protocol=bool(collection_protocol and collection_protocol.get('version')=='combat_movement_pair_collection_v1')
    assert not (a.list_only or a.context_id) or batch_protocol, 'Batch contexts require the predeclared collection protocol'
    source=Path(collection['source']); ss=read(pin(source/'complete.json'))
    model=read(pin(source/'weights.json',ss['weights_sha256'])); pin(source/'checkpoint.pt',ss['checkpoint_sha256'])
    if batch_protocol:
        assert ss['weights_sha256']==collection_protocol['source_weights_sha256']
        assert collection_protocol['observation_timing_contract']=='frozen_world_v1'
    assert not model.get('diagnostic_joint_aim_probe') and model.get('diagnostic_movement_probe') is None
    assignment=collection['assignments'][0]; plan=read(pin(assignment['plan'],assignment['plan_sha256']))
    assert all(t['split']=='train' and t['modes']==['learned'] for t in plan['tasks'])
    pin(plan['registry_path'],plan['registry_sha256'])
    pool=read(pin(a.corpus/'pool/report.json',closed['pool_sha256'])); assert pool['source_unchanged']
    process=read(pin(a.corpus/'processing-train/report.json')); jobs={j['job']:j for j in pool['jobs']}
    contexts=[]; coverage=[]
    for ci in sorted(process['cases'],key=lambda c:c['seed']):
        job=jobs[ci['job']]; task=plan['tasks'][job['task_index']]
        slot=task['seeds'].index(job['seed']); split='development' if slot==3 else 'train'
        directory=a.corpus/'processing-train'/f"case-{job['job']:04d}"
        meta=read(pin(directory/'cuda/report.json')); assert meta['numerical_verification']=='cuda_verified_v1'
        if batch_protocol: assert meta['observation_timing_contract']=='frozen_world_v1'
        sp=next((directory/'native').glob('replay-*/steps.jsonl')); pin(sp,meta['source_sha256'][str(sp.resolve())])
        steps=[json.loads(line) for line in sp.open(encoding='utf-8-sig')]
        first=min(s['observation']['identity']['frame'] for s in steps if s['owner']=='provider')
        choices=[]
        for s in steps:
            o=s['observation']; sample=s.get('sample'); dispatch=s.get('server_execution') or {}
            if s['owner']!='provider' or not sample or not sample.get('aim_candidate') or not s.get('next_observation'): continue
            if o['identity']['life']!=1 or o['identity']['frame']<first+10 or o['health']<=0 or not o['on_ground']: continue
            if not dispatch.get('matched') or not dispatch.get('window_exclusive') or dispatch.get('recovery_commands')!=0: continue
            if not sample['attack'] or sample['vertical']!=0 or s['action'].get('weapon','') not in ('','Machinegun'): continue
            threats=[e for e in o['enemies'] if e['class']=='monster_parasite' and e.get('clear_shot') and e['distance']<=288]
            if not threats or not s['action'].get('target_entity'): continue
            selected=next(e for e in o['enemies'] if e['id']==s['action']['target_entity'])
            speed=math.hypot(*o['velocity'][:2]); distance=min(e['distance'] for e in threats)
            choices.append(dict(frame=o['identity']['frame'],speed=speed,parasite_distance=distance,
                selected_class=selected['class'],observed_enemies=len(o['enemies']),
                priority=(selected['class']=='monster_parasite',speed<50,o['identity']['frame'])))
        chosen=min(choices,key=lambda c:c['priority']) if choices else None
        coverage.append(dict(seed=job['seed'],episode=task['episode']['id'],split=split,eligible_contexts=len(choices),selected=bool(chosen)))
        if chosen:
            chosen.pop('priority')
            contexts.append(dict(**chosen,context_id=f"movement-{job['seed']}-{chosen['frame']}",seed=job['seed'],job=job['job'],
                split=split,episode=task['episode']['id'],instance=task['instances'][slot],offset=assignment['offset']+slot,
                directory=str(directory.resolve())))
    selected=[]
    if batch_protocol:
        assert {c['instance']['map'] for c in contexts}<=set(collection_protocol['maps'])
        for episode in sorted({c['episode'] for c in contexts}):
            for split,quota in [('train',2),('development',1)]:
                candidates=[c for c in contexts if c['episode']==episode and c['split']==split]
                selected.extend(sorted(candidates,key=lambda c:(c['selected_class']=='monster_parasite',c['speed']<50,c['seed']))[:quota])
        assert len(selected)<=collection_protocol['max_contexts'] and len({c['seed'] for c in selected})==len(selected)
    if a.list_only:
        assert not a.context_id
        for path,h in pins.items(): assert sha(path)==h,path
        a.out.mkdir(parents=True)
        report=dict(state='complete',version='combat_movement_context_index_v1',baseline=str(a.corpus.resolve()),
            source_weights_sha256=ss['weights_sha256'],protocol=str(a.protocol.resolve()),protocol_sha256=sha(a.protocol),
            contexts=selected,contexts_available=contexts,coverage=coverage,inputs=pins,neural_execution=False,
            training_performed=False,confirmation_used=False,promotion=False)
        save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
        print(dict(contexts=len(selected),available=len(contexts),train=sum(c['split']=='train' for c in selected),development=sum(c['split']=='development' for c in selected)))
        return
    train=[c for c in contexts if c['split']=='train']; assert train
    context=min(train,key=lambda c:(c['selected_class']=='monster_parasite',c['speed']<50,c['episode'],c['seed']))
    if a.context_id:
        context=next(c for c in selected if c['context_id']==a.context_id)
    directory=Path(context['directory']); meta=read(pin(directory/'cuda/report.json'))
    rows=[json.loads(l) for l in pin(directory/'cuda/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')]
    sequence_rows=[json.loads(l) for l in pin(directory/'cuda/sequence.jsonl',meta['sequence_sha256']).open(encoding='utf-8-sig')]
    assert {r['seed'] for r in rows}=={context['seed']}
    training_devices(); torch.set_default_device('cuda'); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    prepared=prepare_context(sequence_rows,rows,'cuda')
    core=base_model(model,allow_shared=True)
    actor,critic,joint=branch(core,'actor').requires_grad_(False),branch(core,'value').requires_grad_(False),JointAim(model,allow_shared=True).requires_grad_(False)
    actor_before=copy.deepcopy(actor.state_dict()); hints_before=copy.deepcopy(joint.state_dict())
    x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
    frames=torch.tensor([r['frame'] for r in rows],device='cuda',dtype=torch.long); at=frames==context['frame']; assert int(at.sum())==1
    cat=lambda key:torch.tensor([r['sample'].get(key,0) for r in rows],device='cuda',dtype=torch.long)
    target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
    z=torch.tensor([r['sample']['latent'] for r in rows],device='cuda',dtype=torch.float32)
    attack=torch.tensor([float(r['sample']['attack']) for r in rows],device='cuda'); std=torch.tensor(model['log_std'],device='cuda')
    source_z,choice,timing,active=events(rows,x,target,mode,vertical,weapon)
    with torch.no_grad():
        raw,memory_error=sequence(actor,prepared,32,True,'actor'); assert memory_error<2e-5
        mean=selected_means(raw,target,mode,x,source_z[:,:2],actor.sampled_motion_aim_branch,vertical,actor.motor_branch)
        base=probabilities(joint,raw,std,source_z,z,attack,vertical,weapon,x,target,mode,choice,timing,
            actor.sampled_motion_aim_branch,actor.motor_branch,actor.conditional_fire_branch)
        native_lp=torch.tensor([r['sample']['log_probability'] for r in rows],device='cuda')
        replay_error=float((base[0]-native_lp).abs().max()); assert replay_error<.003
    variants=[]; a.out.mkdir(parents=True); (a.out/'models').mkdir()
    proposals=[('control',[0.,0.]),('left',[0.,-1.25]),('right',[0.,1.25]),('forward',[1.25,0.]),('backward',[-1.25,0.])]
    stop_tensor=-raw[at,:2][0]
    if bool((stop_tensor.abs()<=2).all()): proposals.append(('near_stationary',stop_tensor.tolist()))
    for index,(label,delta) in enumerate(proposals):
        changed=copy.deepcopy(model); changed['diagnostic_movement_probe']=dict(version=VERSION,frame=context['frame'],mean_shift=delta)
        validate_movement(changed)
        try: base_model(changed,allow_shared=True)
        except AssertionError: pass
        else: raise AssertionError('Diagnostic model admitted to ordinary base loader')
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            decoded=branch(base_model(changed,allow_probe=True,allow_shared=True),'actor').requires_grad_(False)
            decoded_joint=JointAim(changed,allow_probe=True,allow_shared=True).requires_grad_(False)
        assert equal_tree(decoded.state_dict(),actor_before) and equal_tree(decoded_joint.state_dict(),hints_before)
        with torch.no_grad():
            decoded_raw,_=sequence(decoded,prepared,32,True,'actor'); assert torch.equal(decoded_raw,raw)
            shift=shifts(changed,frames)
            scored=probabilities(decoded_joint,raw,std,source_z,z,attack,vertical,weapon,x,target,mode,choice,timing,
                actor.sampled_motion_aim_branch,actor.motor_branch,actor.conditional_fire_branch,movement_shift=shift)
            analytic=(shift*(source_z[:,:2]-mean[:,:2])/std[:2].exp().square()-.5*shift.square()/std[:2].exp().square()).sum(-1)
            error=float(((scored[0]-base[0])-analytic).abs().max()); assert error<.001,error
            assert torch.equal(scored[0][~at],base[0][~at]) and torch.equal(scored[2],base[2])
            if label=='control': assert torch.equal(scored[0],base[0])
        folder=a.out/'models'/label; folder.mkdir(); save(folder/'weights.json',changed)
        variants.append(dict(label=label,index=index,model=str((folder/'weights.json').resolve()),model_sha256=sha(folder/'weights.json'),
            mean_shift=delta,frame=context['frame'],gaussian_joint_formula_max_error=error))
    assert equal_tree(actor.state_dict(),actor_before) and equal_tree(joint.state_dict(),hints_before)
    for path,h in pins.items(): assert sha(path)==h,path
    shutil.copy2(__file__,a.out/Path(__file__).name)
    report=dict(state='complete',version='combat_movement_probe_preparation_v1',device='cuda',diagnostic_only=True,
        baseline=str(a.corpus.resolve()),source=str(source.resolve()),source_weights_sha256=ss['weights_sha256'],
        registry=plan['registry_path'],protocol=str(a.protocol.resolve()),protocol_sha256=sha(a.protocol),
        context=context,contexts_available=contexts,coverage=coverage,variants=variants,inputs=pins,
        collection_protocol=batch_protocol,internal_development=context['split']=='development',
        ordinary_outside_frame_exact=True,zero_control_exact=True,neural_parameters_exact=True,source_lp_max_error=replay_error,
        training_performed=False,validation_used=False,confirmation_used=False,promotion=False,
        scope='One observable training-registry context; internal development stays separate. Alternatives not yet executed. Density proof is not evidence of better maneuvers.')
    save(a.out/'report.json',report); save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(context=context,variants=len(variants),available_contexts=len(contexts),source_lp_max_error=replay_error))


if __name__=='__main__': main()
