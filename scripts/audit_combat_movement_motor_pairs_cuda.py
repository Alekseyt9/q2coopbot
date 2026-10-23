"""Frozen motor forecasts versus admitted native alternative shots; no fitting."""
import argparse
import json
from pathlib import Path

from evaluate_registered_combat_models import read,save,sha
from report_combat_machinegun_hits import scan,vector
from ppo_combat import torch,training_devices
from combat_joint_aim_policy import JointAim
from combat_action_effect import inputs as motor_inputs
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','source','protocol','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();root=a.collection.resolve();assert not a.out.exists();pins={}

    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path)
        pins[str(path)]=h;return path

    p=read(pin(a.protocol));assert p['version']=='combat_movement_motor_pair_audit_v1'
    ds=read(pin(root/'dataset/report.json',p['dataset_report_sha256']));assert ds['gate_passed']
    records=[r for r in read(pin(root/'dataset/records.json',ds['records_sha256']))['records'] if r['split']=='train']
    closed=read(pin(root/'complete.json'));spec=read(pin(root/'spec.json',closed['spec_sha256']))
    pool=read(pin(root/'pool/report.json',closed['pool_sha256']));jobs={j['plan_index']:j for j in pool['jobs']}
    ss=read(pin(a.source/'complete.json'));model=read(pin(a.source/'weights.json',ss['weights_sha256']))
    assert ss['weights_sha256']==p['source_weights_sha256']==ds['source_weights_sha256']
    by_context={r['context']['context_id']:r for r in spec['contexts']};members=[];missing=[]
    for record in records:
        context=record['context'];f=context['frame'];item=by_context[context['context_id']]
        proof=read(pin(record['proof'],record['proof_sha256']));assert proof['excluded']==0 and proof['baseline_full_exact']
        for v in item['variants']:
            i=v['plan_index'];folder=root/f'case-{i:02d}'
            meta=read(pin(folder/'cuda/report.json',proof['inputs'][str((folder/'cuda/report.json').resolve())]))
            rows=list(map(json.loads,pin(folder/'cuda/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
            row=next(r for r in rows if r['frame']==f)
            sp=next((folder/'native').glob('replay-*/steps.jsonl'))
            step=next(s for s in map(json.loads,pin(sp,meta['source_sha256'][str(sp.resolve())]).open(encoding='utf-8-sig')) if s['observation']['identity']['frame']==f)
            assert step['owner']=='provider' and step['server_execution']['window_exclusive'] and step['server_execution']['recovery_commands']==0
            ns=step['native_step'];window=tuple(ns[k] for k in ('spawncount','begin_frame','actor','sequence'))
            capture=read(pin(Path(jobs[i]['root'])/'report.json'));assert capture['capture_complete'] and capture['provenance_valid']
            path=pin(Path(capture['results'][0]['root'])/'server.log')
            with path.open(encoding='utf-8-sig') as stream:
                shots=[s for s in scan(stream,{window}) if s['eligible']]
            if len(shots)!=1:
                missing.append(dict(context_id=context['context_id'],seed=record['seed'],map=record['map'],label=v['label'],native_shots=len(shots)))
                continue
            shot=shots[0]
            members.append(dict(context_id=context['context_id'],seed=record['seed'],map=record['map'],label=v['label'],
                features=row['features'],sample=row['sample'],position=step['observation']['position'],
                next_position=step['next_observation']['position'] if step.get('next_observation') else None,
                native_muzzle=vector(shot['fields']['start']),window=list(window)))
    assert members
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    joint=JointAim(model,allow_shared=True).requires_grad_(False);before=copy_state(joint)
    tensor=lambda values:torch.tensor(values,dtype=torch.float64,device='cuda')
    x=tensor([m['features'] for m in members]);z=tensor([m['sample']['latent'] for m in members])
    cat=lambda k:torch.tensor([m['sample'].get(k,0) for m in members],device='cuda',dtype=torch.long)
    timing=tensor([m['sample']['aim_candidate']['timing'] for m in members])
    with torch.no_grad():
        c=joint.action_context(x,z,cat('target'),cat('aim_mode'),cat('vertical'),cat('weapon'))
        ins=motor_inputs(c,timing)
        prediction=joint.hints.motor((ins-joint.hints.motor_mean)/joint.hints.motor_std)
        position=tensor([m['position'] for m in members]);muzzle=tensor([m['native_muzzle'] for m in members])
        base=position.clone();base[:,2]+=torch.where(c[:,3]>0,-10.,14.)
        yaw=torch.atan2(c[:,6],c[:,7]);cy,sy=yaw.cos(),yaw.sin()
        local=lambda v:torch.stack((cy*v[:,0]+sy*v[:,1],-sy*v[:,0]+cy*v[:,1],v[:,2]),-1)
        actual=local(muzzle-base)/32
        velocity=local(c[:,10:13]*400)*.1/32
        absolute=lambda q:dict(horizontal_rmse=float(((q[:,:2]-actual[:,:2])*32).square().sum(-1).mean().sqrt()),
            horizontal_mean_error=float(((q[:,:2]-actual[:,:2])*32).norm(dim=-1).mean()))
        table={(m['context_id'],m['label']):i for i,m in enumerate(members)}
        pairs=[(i,table[(m['context_id'],'control')]) for i,m in enumerate(members) if m['label']!='control' and (m['context_id'],'control') in table]
        assert pairs
        left=torch.tensor([i for i,j in pairs],device='cuda');right=torch.tensor([j for i,j in pairs],device='cuda')
        predicted_delta=(prediction[left,:2]-prediction[right,:2])*32
        actual_delta=(actual[left,:2]-actual[right,:2])*32
        baseline_rmse=float(actual_delta.square().sum(-1).mean().sqrt())
        model_rmse=float((predicted_delta-actual_delta).square().sum(-1).mean().sqrt())
        magnitude=actual_delta.norm(dim=-1);eligible=magnitude>=5
        cosine=(predicted_delta*actual_delta).sum(-1)/(predicted_delta.norm(dim=-1)*magnitude).clamp_min(1e-12)
        cosine_mean=float(cosine[eligible].mean()) if bool(eligible.any()) else None
        pair_members=[members[i] for i,j in pairs]
        seeds=len({m['seed'] for m in pair_members});maps=len({m['map'] for m in pair_members})
        g=p['feature_admission']
        gates=dict(pairs=len(pairs)>=g['minimum_pairs'],seeds=seeds>=g['minimum_seeds'],maps=maps>=g['minimum_maps'],
            relative_rmse=baseline_rmse>0 and model_rmse<=g['paired_horizontal_rmse_over_zero_max']*baseline_rmse,
            direction=cosine_mean is not None and cosine_mean>=g['minimum_cosine_on_effects_at_least5'])
        timing_ids=[i for i,m in enumerate(members) if m['next_position'] is not None]
        ni=torch.tensor(timing_ids,device='cuda');next_pose=tensor([members[i]['next_position'] for i in timing_ids])
        timing_error=(next_pose[:,:2]-muzzle[ni,:2]).norm(dim=-1)
        labels=[dict(context_id=m['context_id'],seed=m['seed'],map=m['map'],label=m['label'],predicted_motor=prediction[i].tolist(),actual_local_displacement=actual[i].tolist()) for i,m in enumerate(members)]
        paired=[dict(context_id=m['context_id'],seed=m['seed'],map=m['map'],label=m['label'],predicted_delta=predicted_delta[i].tolist(),actual_delta=actual_delta[i].tolist(),cosine=float(cosine[i])) for i,m in enumerate(pair_members)]
        metrics=dict(shots=len(members),pairs=len(pairs),seeds=seeds,maps=maps,
            absolute=dict(motor=absolute(prediction),zero=absolute(torch.zeros_like(prediction)),constant_velocity100ms=absolute(velocity)),
            paired=dict(model_horizontal_rmse=model_rmse,zero_horizontal_rmse=baseline_rmse,
                ratio=model_rmse/baseline_rmse if baseline_rmse else None,large_effect_pairs=int(eligible.sum()),cosine_on_large_effects=cosine_mean),
            native_shot_to_next_observation_xy=dict(count=len(timing_ids),mean=float(timing_error.mean()),max=float(timing_error.max())))
    assert equal_tree(joint.state_dict(),before)
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);save(a.out/'labels.json',labels);save(a.out/'pairs.json',paired)
    report=dict(state='complete',version='combat_movement_motor_pair_audit_v1',device='cuda',split='train_only',
        metrics=metrics,missing_shots=missing,gates=gates,feature_admission_passed=all(gates.values()),inputs=pins,
        labels_sha256=sha(a.out/'labels.json'),pairs_sha256=sha(a.out/'pairs.json'),script_sha256=sha(__file__),
        neural_parameters_exact=True,training_performed=False,new_games=False,promotion=False,
        scope='Frozen movement effect forecast audit, native targets only offline; no useful action selector or battle superiority established. Coverage excludes absent/multiple MG shots.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(metrics=metrics,gates=gates,feature_admission_passed=report['feature_admission_passed'],missing_shots=len(missing)))


def copy_state(module):
    return {k:v.detach().clone() for k,v in module.state_dict().items()}


if __name__=='__main__':main()
