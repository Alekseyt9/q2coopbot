"""Real-history finite difference: raw movement reference versus external mean shift."""
import argparse
import copy
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,training_devices
from ppo_recurrent import prepare_context,sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_precision_head import selected_means


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','corpus','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path)
        pins[str(path)]=h;return path
    ss=read(pin(a.source/'complete.json'));model=read(pin(a.source/'weights.json',ss['weights_sha256']))
    assert ss['weights_sha256']=='559ff0ad4297047cdc4a9ad531fa9159668b045b7f41bf8e32a6df975278bea6'
    seal=read(pin(a.corpus/'complete.json'));pool=read(pin(a.corpus/'pool/report.json',seal['pool_sha256']))
    plan=read(pin(a.corpus/'train-plan.json'))
    chosen=[]
    for mp in ('base3','city1','jail1','ware2'):
        selected=min([j for j in pool['jobs'] if plan['tasks'][j['task_index']]['episode']['map']==mp],key=lambda j:j['seed'])
        chosen.append((mp,selected))
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    actor=branch(base_model(model,allow_shared=True),'actor').requires_grad_(False)
    sigma=torch.tensor(model['log_std'],device='cuda').exp();results=[]
    a.out.mkdir(parents=True)
    save(a.out/'protocol.json',dict(version='combat_movement_reference_coupling_audit_v1',source_weights_sha256=ss['weights_sha256'],
        selected_jobs=[j['job'] for mp,j in chosen],perturbations=[[.02,0],[-.02,0],[0,.02],[0,-.02]],
        scope='Finite-difference output-bias perturbations on frozen TRAIN histories, not past-update attribution or quality. External mean shift keeps raw reference fixed by construction.',
        materiality='Aggregated eligible conditional-aim KL at least5 times movement-only KL on at least3 of4 perturbations; descriptive decision for testing an independent residual head.',training_performed=False))
    for mp,job in chosen:
        directory=a.corpus/'processing-train'/f"case-{job['job']:04d}/cuda"
        meta=read(pin(directory/'report.json'));assert meta['model_sha256']==ss['weights_sha256']
        rows=list(map(json.loads,pin(directory/'rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
        context=list(map(json.loads,pin(directory/'sequence.jsonl',meta['sequence_sha256']).open(encoding='utf-8-sig')))
        prepared=prepare_context(context,rows,'cuda');x=torch.tensor([r['features'] for r in rows],device='cuda')
        cat=lambda k:torch.tensor([r['sample'].get(k,0) for r in rows],device='cuda',dtype=torch.long)
        target,mode,vertical=(cat(k) for k in ('target','aim_mode','vertical'))
        source=torch.tensor([(r['sample'].get('aim_candidate') or {}).get('source_latent',r['sample']['latent']) for r in rows],device='cuda')
        eligible=torch.tensor([bool(r['sample'].get('aim_candidate')) for r in rows],device='cuda')
        with torch.no_grad():
            old,error=sequence(actor,prepared,32,True,'actor');assert error<2e-5
            original=selected_means(old,target,mode,x,source[:,:2],actor.sampled_motion_aim_branch,vertical,actor.motor_branch)
            for delta in ([.02,0],[-.02,0],[0,.02],[0,-.02]):
                other=copy.deepcopy(actor);other.base.head.bias[:2].add_(torch.tensor(delta,device='cuda'))
                raw,e=sequence(other,prepared,32,True,'actor');assert e<2e-5
                changed=selected_means(raw,target,mode,x,source[:,:2],other.sampled_motion_aim_branch,vertical,other.motor_branch)
                expected=torch.tensor(delta,device='cuda');assert float(((raw-old)[:,:2]-expected).abs().max())<1e-6
                move_kl=.5*((changed[:,:2]-original[:,:2])/sigma[:2]).square().sum(-1)
                aim_kl=.5*((changed[:,2:]-original[:,2:])/sigma[2:]).square().sum(-1)
                external=original.clone();external[:,:2]+=expected
                external_aim=.5*((external[:,2:]-original[:,2:])/sigma[2:]).square().sum(-1);assert not bool(torch.count_nonzero(external_aim))
                results.append(dict(map=mp,seed=job['seed'],delta=delta,rows=len(rows),eligible_rows=int(eligible.sum()),
                    movement_kl_mean=float(move_kl.mean()),conditional_aim_kl_mean=float(aim_kl.mean()),conditional_aim_kl_max=float(aim_kl.max()),
                    eligible_movement_kl_mean=float(move_kl[eligible].mean()) if bool(eligible.any()) else None,
                    eligible_conditional_aim_kl_mean=float(aim_kl[eligible].mean()) if bool(eligible.any()) else None,
                    external_conditional_aim_kl=0.))
    aggregate=[]
    for delta in ([.02,0],[-.02,0],[0,.02],[0,-.02]):
        group=[r for r in results if r['delta']==delta];n=sum(r['eligible_rows'] for r in group)
        mean=lambda key:sum(r[key]*r['eligible_rows'] for r in group if r['eligible_rows'])/n
        movement,aim=mean('eligible_movement_kl_mean'),mean('eligible_conditional_aim_kl_mean')
        aggregate.append(dict(delta=delta,eligible_rows=n,movement_kl=movement,conditional_aim_kl=aim,ratio=aim/movement,material=aim>=5*movement))
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_movement_reference_coupling_audit_v1',device='cuda',results=results,aggregate=aggregate,
        material_coupling=sum(r['material'] for r in aggregate)>=3,inputs=pins,training_performed=False,new_games=False,source_changed=False,
        scope='Conditional Gaussian KL only, at fixed recorded actions/targets/modes. Raw-bias perturbation also may affect categorical logits; not full joint KL or proof of prior PPO failure cause. External-shift aim independence is structural.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(material_coupling=report['material_coupling'],aggregate=aggregate))


if __name__=='__main__':main()
