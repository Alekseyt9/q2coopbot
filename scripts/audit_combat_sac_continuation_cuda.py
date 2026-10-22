"""Read-back SAC continuity and terminal native target errors on new TRAIN data."""
import argparse
import pathlib
from process_combat_architecture_pool import read,sha
from ppo_recurrent import torch,durable_json
from prepare_combat_target_heads import build
from combat_replay_catalog import policy_family
from combat_conditional_sac import TwinQ,condition_state


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','update','fresh-replay','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    a=ap.parse_args();assert torch.cuda.is_available() and not a.out.exists()
    torch.set_default_device('cuda');torch.set_num_threads(2);pins={}
    def pin(p,expected=None):
        p=pathlib.Path(p);h=sha(p);assert expected is None or h==expected
        pins[str(p.resolve())]=h;return p
    def load(root):
        seal=read(pin(root/'complete.json'));report=read(pin(root/'report.json',seal['report_sha256']))
        model=read(pin(root/'weights.json',seal['weights_sha256']))
        cp=torch.load(pin(root/'checkpoint.pt',seal['checkpoint_sha256']),map_location='cuda',weights_only=True)
        sac=torch.load(pin(root/'sac-checkpoint.pt',report['sac_checkpoint_sha256']),map_location='cuda',weights_only=True)
        assert cp['weights_sha256']==sac['weights_sha256']==seal['weights_sha256']
        for k,v in build(model,'actor').state_dict().items():assert torch.equal(v,cp['actor'][k]) and torch.equal(v,sac['actor'][k])
        assert torch.equal(cp['log_std'],sac['log_std'])
        return model,cp,sac,report
    old_model,old_cp,old,_=load(a.source);model,cp,new,report=load(a.update)
    assert report['resumed'] and policy_family(old_model)==policy_family(model)
    assert cp['config']==old_cp['config'] and torch.equal(cp['log_std'][:2],old_cp['log_std'][:2])
    for k,v in cp['value'].items():assert torch.equal(v,old_cp['value'][k])
    assert cp['updates_completed']==old_cp['updates_completed'] and cp['total_actor_steps']==old_cp['total_actor_steps']
    deltas={}
    for kind,amount in (('actor_optimizer',report['actor_steps']),('critic_optimizer',report['critic_steps'])):
        before,after=old[kind],new[kind]
        assert before['param_groups']==after['param_groups'] and before['state'].keys()==after['state'].keys()
        for key,state in after['state'].items():
            assert float(state['step'])-float(before['state'][key]['step'])==amount
            for tag,value in state.items():
                if torch.is_tensor(value):assert value.is_cuda and bool(torch.isfinite(value).all())
                if tag in ('exp_avg','exp_avg_sq'):assert value.shape==before['state'][key][tag].shape
        deltas[kind]=amount
    for name in ('critic','target_critic'):
        assert old[name].keys()==new[name].keys()
        assert any(not torch.equal(v,old[name][k]) for k,v in new[name].items())
        assert all(v.is_cuda and bool(torch.isfinite(v).all()) for v in new[name].values())
    cache=read(pin(a.fresh_replay/'complete.json'))
    assert cache['source_weights_sha256']==old['weights_sha256']
    d=torch.load(pin(a.fresh_replay/'replay.pt',cache['replay_sha256']),map_location='cuda',weights_only=True)
    mask=~d['bootstrap'];indices=d['current'][mask]
    assert len(indices)>0
    state=condition_state(d['features'][indices],d['history'][indices],d['condition'][mask],d['latent'][mask,:2])
    errors={}
    with torch.no_grad():
        for label,checkpoint in (('before',old),('after',new)):
            q=TwinQ(state.shape[1]);q.load_state_dict(checkpoint['critic'])
            first,second=q(state,d['latent'][mask,2:].tanh());target=d['reward'][mask]
            errors[label]=dict(terminal_action_rows=len(indices),minimum_q_rmse=float((torch.minimum(first,second)-target).square().mean().sqrt()),
                q1_rmse=float((first-target).square().mean().sqrt()),q2_rmse=float((second-target).square().mean().sqrt()))
    for p,h in pins.items():assert sha(p)==h
    durable_json(a.out,dict(state='complete',device='cuda',version='combat_sac_continuation_cuda_audit_v1',optimizer_step_deltas=deltas,
        fresh_training_cases=cache['training_cases'],fresh_rows=cache['rows'],terminal_native_target_errors=errors,
        policy_family_exact=True,inputs=[dict(path=p,sha256=h) for p,h in pins.items()],training_performed=False,promotion=False,
        scope='Sealed update continuity, exact protected policy family and finite CUDA states. Error against actual zero-bootstrap native reward on fresh TRAIN transitions before/after fitting those same transitions; not held-out calibration or game quality.'))
    print(dict(state='complete',optimizer_deltas=deltas,terminal_errors=errors))


if __name__=='__main__':main()
