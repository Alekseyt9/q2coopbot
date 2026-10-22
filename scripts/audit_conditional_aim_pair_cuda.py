"""Independent CUDA ownership/optimizer/readback audit before native comparison."""
import argparse
import pathlib
from ppo_recurrent import torch, prepare_context, sequence, durable_json
from prepare_combat_target_heads import build
from process_combat_architecture_pool import read, sha
from combat_conditional_sac import TwinQ, condition_state


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','ppo','sac','replay','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    a=ap.parse_args();assert torch.cuda.is_available() and not a.out.exists()
    torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path);digest=sha(path);assert expected is None or digest==expected
        pins[str(path.resolve())]=digest;return path
    def load(root):
        seal=read(pin(root/'complete.json'))
        for file,key in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(root/file,seal[key])
        model=read(root/'weights.json');cp=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert cp['weights_sha256']==seal['weights_sha256']
        for kind in ('actor','value'):
            restored=build(model,kind).state_dict()
            assert restored.keys()==cp[kind].keys()
            for k,v in restored.items():assert torch.equal(v,cp[kind][k])
        assert torch.equal(cp['log_std'],torch.tensor(model['log_std'],device='cuda',dtype=torch.float32))
        return model,cp
    original,initial=load(a.source);results={}
    cache=read(pin(a.replay/'complete.json'));pin(a.replay/'replay.pt',cache['replay_sha256'])
    data=torch.load(a.replay/'replay.pt',map_location='cuda',weights_only=True)
    corpus=pathlib.Path(cache['corpus']);merged=corpus/'processing-train/merged'
    import json
    rows=[json.loads(l) for l in (merged/'rollout.jsonl').read_text().splitlines()]
    context=[json.loads(l) for l in (merged/'sequence.jsonl').read_text().splitlines()]
    prepared=prepare_context(context,rows,'cuda')
    with torch.no_grad():reference,error=sequence(build(original,'actor'),prepared,check_states=True)
    assert error<2e-5
    for name,root in (('ppo',a.ppo),('sac',a.sac)):
        model,cp=load(root);report=read(root/'report.json')
        assert cp['config']==initial['config']
        assert torch.equal(cp['log_std'][:2],initial['log_std'][:2])
        changed=[]
        for k,v in cp['actor'].items():
            assert v.is_cuda and bool(torch.isfinite(v).all())
            if not torch.equal(v,initial['actor'][k]):
                assert k.startswith('sampled_motion_aim_branch.'),k;changed.append(k)
        assert changed
        with torch.no_grad():raw,_=sequence(build(model,'actor'),prepared,check_states=True)
        assert torch.equal(raw,reference), 'Frozen category/movement/mode logits drifted'
        assert read(merged/'report.json')['rollout_sha256'] in cp['consumed_rollouts']
        if name=='ppo':
            assert cp['actor_training']['scope']=='conditional-aim' and report['actor_steps']>0
            actor=build(model,'actor');std=torch.nn.Parameter(cp['log_std'].clone())
            from combat_weapon_state_residual import optimizer
            opt=optimizer(actor,std,cp['config']['actor_lr'],.0003)
            names={id(p):k for k,p in actor.named_parameters()};names[id(std)]='$log_std'
            mapping={identifier:p for group,saved in zip(opt.param_groups,cp['actor_optimizer']['param_groups']) for p,identifier in zip(group['params'],saved['params'])}
            for identifier,state in cp['actor_optimizer']['state'].items():
                parameter=mapping[identifier];key=names[id(parameter)]
                assert key=='$log_std' or key.startswith('sampled_motion_aim_branch.')
                assert float(state['step'])==cp['total_actor_steps']-initial['total_actor_steps']
                assert state['exp_avg'].shape==state['exp_avg_sq'].shape==parameter.shape
                for value in state.values():
                    if torch.is_tensor(value):assert value.is_cuda and bool(torch.isfinite(value).all())
                if key=='$log_std':assert bool((state['exp_avg'][:2]==0).all()) and bool((state['exp_avg_sq'][:2]==0).all())
            assert all(float(s['step'])==(cp['updates_completed']-initial['updates_completed'])*cp['config']['value_steps'] for s in cp['value_optimizer']['state'].values())
            assert cp['value_optimizer']['state']
        else:
            for key,value in cp['value'].items():assert torch.equal(value,initial['value'][key])
            assert not cp['actor_optimizer']['state'] and not cp['value_optimizer']['state']
            checkpoint=torch.load(pin(root/'sac-checkpoint.pt',report['sac_checkpoint_sha256']),map_location='cuda',weights_only=True)
            for k,v in checkpoint['actor'].items():assert torch.equal(v,cp['actor'][k])
            assert torch.equal(checkpoint['log_std'],cp['log_std'])
            for opt_name,steps in (('actor_optimizer',report.get('total_actor_steps',report['actor_steps'])),('critic_optimizer',report.get('total_critic_steps',report['critic_steps']))):
                opt=checkpoint[opt_name];assert opt['state']
                for state in opt['state'].values():
                    assert float(state['step'])==steps
                    assert state['exp_avg'].shape==state['exp_avg_sq'].shape
                    for value in state.values():
                        if torch.is_tensor(value):assert value.is_cuda and bool(torch.isfinite(value).all())
            ci=data['current'][:128]
            state=condition_state(data['features'][ci],data['history'][ci],data['condition'][:128],data['latent'][:128,:2])
            q=TwinQ(state.shape[1]);q.load_state_dict(checkpoint['critic'])
            with torch.no_grad():first,second=q(state,data['latent'][:128,2:].tanh())
            assert bool(torch.isfinite(first).all() & torch.isfinite(second).all())
            assert any(not torch.equal(v,checkpoint['target_critic'][k]) for k,v in checkpoint['critic'].items())
            assert cp['updates_completed']==initial['updates_completed'] and cp['total_actor_steps']==initial['total_actor_steps']
        results[name]=dict(changed_actor_tensors=changed,actor_steps=report['actor_steps'],raw_logits_exact=True,
            movement_std_exact=True,optimizer_shapes_steps_and_finiteness=True)
    for path,digest in pins.items():assert sha(path)==digest
    durable_json(a.out,dict(state='complete',version='combat_conditional_aim_pair_cuda_audit_v1',device='cuda',
        gpu=torch.cuda.get_device_name(),rows=len(rows),training_cases=cache['training_cases'],results=results,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),training_performed=False,promotion=False,
        scope='Both models restored independently on CUDA. Only conditional aim branch/aim variance changed; all category/motion/mode raw logits exact on full causal source TRAIN histories. Native held-out quality still required. SAC warm replay is not proof of efficient online SAC.'))
    print('Conditional aim pair CUDA audit passed')


if __name__=='__main__':main()
