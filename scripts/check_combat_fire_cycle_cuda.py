"""CUDA parity and exact off-cycle preservation for the observed-cycle head."""
import argparse
from pathlib import Path
from ppo_combat import torch,training_devices
from combat_full_joint_ppo import FullJointScope,equal_tree
from combat_weapon_fire_residual import CYCLE_VERSION,active
from combat_causal_fire_pairs import load
from train_combat_causal_fire_cuda import context,fire
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','migrated','dataset','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    data=load(a.dataset);cases=data['cases'];rows=[c['input'] for c in cases];history=[r for c in cases for r in c['input']['sequence_prefix']]
    model0,model1=read(a.source/'weights.json'),read(a.migrated/'weights.json')
    cp=torch.load(a.migrated/'checkpoint.pt',map_location='cuda',weights_only=True)
    old=FullJointScope(model0,cp['config']);new=FullJointScope(model1,cp['config'],allow_cycle_fire=True)
    new.restore(cp,sha(a.migrated/'weights.json'))
    checks={}
    try:FullJointScope(model1,cp['config'])
    except AssertionError:checks['ordinary_full_ppo_rejected']=True
    else:raise AssertionError('Cycle policy silently accepted ordinary PPO')
    with torch.no_grad():
        b0,b1=context(old,rows,history),context(new,rows,history)
        assert torch.equal(b0[1],b1[1]) and torch.equal(fire(old,b0),fire(new,b1))
        checks['nine_real_roots_zero_exact']=True
        n=12*30*2;x=b1[0][:1].repeat(n,1);raw=b1[1][:1].repeat(n,1)
        weapons=torch.arange(12,device='cuda').repeat_interleave(30*2)
        frames=torch.arange(30,device='cuda').repeat_interleave(2).repeat(12)
        alive=torch.arange(2,device='cuda').repeat(12*30)
        x[:,0]=alive;x[:,5]=frames/30;x[:,1121:1133]=torch.nn.functional.one_hot(weapons,12)
        values=[v[:1].repeat(n) for v in b1[2]];latent=b1[3][:1].repeat(n,1)
        batch=(x,raw,values,latent);baseline=fire(old,batch)
        mask=active(x,CYCLE_VERSION);assert int(mask.sum())==1
        checks['all_720_observed_states_gate']=True
        head=new.actor.weapon_fire_residual_branch
        for bias in (-.25,.25):
            head[-1].bias.fill_(bias)
            result=fire(new,batch)
            assert torch.equal(result[~mask],baseline[~mask]),'Off-cycle fire changed'
            assert bool(((result-baseline)[mask]-bias).abs().max()<1e-6)
        checks['nonzero_positive_negative_localized_exact']=True
        for key,value in old.actor.state_dict().items():
            assert equal_tree(value,new.actor.state_dict()[key])
        checks['all_original_actor_parameters_exact']=True
    save(a.out,dict(state='passed',device='cuda',checks=checks,
        inputs={str(p.resolve()):sha(p) for p in (a.source/'weights.json',a.migrated/'weights.json',a.dataset,Path(__file__))},
        scope='CUDA numerical/metadata validation only; synthetic nonzero head is never exported. Native parity/quality separate.'))
    print('Passed',len(checks),'cycle CUDA checks including 720 observed states')


if __name__=='__main__':main()
