"""Audit the one-factor behavior-KL pair before native quality evaluation."""
import argparse
import pathlib
from ppo_recurrent import torch,durable_json
from process_combat_architecture_pool import read,sha
from prepare_combat_target_heads import build
from combat_conditional_sac import aim_distribution


def exact(x,y):
    if torch.is_tensor(x):assert x.is_cuda and y.is_cuda and torch.equal(x,y)
    elif isinstance(x,dict):
        assert x.keys()==y.keys()
        for k in x:exact(x[k],y[k])
    elif isinstance(x,(list,tuple)):
        assert len(x)==len(y)
        for u,v in zip(x,y):exact(u,v)
    else:assert x==y


def main():
    ap=argparse.ArgumentParser()
    for n in ('source','control','candidate','replay','legacy-control','out'):ap.add_argument('--'+n,type=pathlib.Path,required=True)
    a=ap.parse_args();assert torch.cuda.is_available() and not a.out.exists()
    torch.set_default_device('cuda');torch.set_num_threads(2);pins={}
    def pin(p,expected=None):
        p=pathlib.Path(p);h=sha(p);assert expected is None or h==expected
        pins[str(p.resolve())]=h;return p
    def load(root):
        seal=read(pin(root/'complete.json'))
        for f,k in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(root/f,seal[k])
        model=read(root/'weights.json');cp=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
        for kind in ('actor','value'):
            for k,v in build(model,kind).state_dict().items():assert torch.equal(v,cp[kind][k])
        assert torch.equal(torch.tensor(model['log_std'],device='cuda',dtype=torch.float32),cp['log_std'])
        return model,cp
    source_model,source=load(a.source);results={};boundary=[];protocols=[]
    meta=read(pin(a.replay/'complete.json'));d=torch.load(pin(a.replay/'replay.pt',meta['replay_sha256']),map_location='cuda',weights_only=True)
    ix=d['current'];raw=d['raw'][ix];features=d['features'][ix];condition=d['condition'];movement=d['latent'][:,:2]
    teacher=build(source_model,'actor')
    with torch.no_grad():reference=aim_distribution(raw,features,condition,movement,teacher.sampled_motion_aim_branch,source['log_std'])
    for label,root,weight in (('control',a.control,0.),('candidate',a.candidate,.01)):
        model,cp=load(root);report=read(root/'report.json');protocol=read(pin(root/'protocol.json',report['protocol_sha256']))
        assert protocol['behavior_kl_weight']==weight and report['behavior_kl_weight']==weight
        assert cp['config']==source['config'] and torch.equal(cp['log_std'][:2],source['log_std'][:2])
        assert report['rows']==meta['rows'] and report['training_cases']==meta['training_cases']
        assert report['actor_steps']==256 and report['critic_steps']==512 and not report['resumed']
        for k,v in cp['actor'].items():
            if not k.startswith('sampled_motion_aim_branch.'):assert torch.equal(v,source['actor'][k])
        exact(cp['value'],source['value']);exact(cp['value_optimizer'],source['value_optimizer'])
        assert not cp['actor_optimizer']['state']
        sac=torch.load(pin(root/'sac-checkpoint.pt',report['sac_checkpoint_sha256']),map_location='cuda',weights_only=True)
        exact(sac['actor'],cp['actor']);exact(sac['log_std'],cp['log_std'])
        for name,steps in (('actor_optimizer',256),('critic_optimizer',512)):
            assert sac[name]['state']
            for state in sac[name]['state'].values():
                assert float(state['step'])==steps
                for v in state.values():
                    if torch.is_tensor(v):assert v.is_cuda and bool(torch.isfinite(v).all())
        if weight:
            exact(sac['behavior_anchor']['branch'],teacher.sampled_motion_aim_branch.state_dict())
            exact(sac['behavior_anchor']['log_std'],source['log_std'])
            assert sac['behavior_anchor']['weights_sha256']==meta['source_weights_sha256']
        else:
            assert sha(root/'weights.json')==sha(pin(a.legacy_control/'weights.json')), 'Zero-coefficient numerical regression'
        boundary.append(torch.load(pin(root/'warmup-boundary.pt',report['warmup_boundary_sha256']),map_location='cuda',weights_only=True))
        protocols.append(protocol)
        with torch.no_grad():
            current=aim_distribution(raw,features,condition,movement,build(model,'actor').sampled_motion_aim_branch,cp['log_std'])
            kl=torch.distributions.kl_divergence(current,reference).sum(-1)
            distance=((current.loc-reference.loc)/reference.scale).square().sum(-1).sqrt()
            scale=torch.where(condition[:,1]==1,15.,180.)[:,None]
            shift=(current.loc.tanh()-reference.loc.tanh())*scale
            results[label]=dict(mean_current_to_source_kl=float(kl.mean()),fraction_beyond_3_behavior_std=float((distance>3).float().mean()),
                mean_behavior_std_distance=float(distance.mean()),mean_axis_shift_degrees=float(shift.abs().mean()))
    exact(boundary[0],boundary[1])
    ignored={'behavior_kl_weight','behavior_anchor_weights_sha256'}
    assert {k:v for k,v in protocols[0].items() if k not in ignored}=={k:v for k,v in protocols[1].items() if k not in ignored}
    for p,h in pins.items():assert sha(p)==h
    durable_json(a.out,dict(state='complete',version='combat_sac_behavior_pair_cuda_audit_v1',device='cuda',
        equal_source_data_rng_and_warmup_critic_optimizer_boundary=True,zero_coefficient_matches_legacy_weights_exact=True,
        rows=meta['rows'],training_cases=meta['training_cases'],results=results,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),training_performed=False,promotion=False,
        scope='One-factor fixed-budget CUDA KL pair. Protected actor/value state, saved source anchor, exact256-step warmup/RNG boundary and legacy zero-weight regression. KL/mean shift is TRAIN diagnostic; native fresh quality required.'))
    print(results)


if __name__=='__main__':main()
