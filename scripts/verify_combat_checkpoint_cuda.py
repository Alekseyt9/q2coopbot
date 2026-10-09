"""Verify serialized model/optimizer continuation entirely on CUDA."""
import argparse,json,pathlib
from process_combat_architecture_pool import read,sha,save
from ppo_recurrent import torch,CausalAttention


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    assert torch.cuda.is_available();results=[]
    for binding in read(a.models):
        model=read(binding['model']);report=read(binding['resume_report'])
        assert sha(binding['resume_checkpoint'])==binding['resume_checkpoint_sha256']
        cp=torch.load(binding['resume_checkpoint'],map_location='cuda',weights_only=True)
        assert cp['version']=='combat_architecture_checkpoint_v1' and cp['weights_sha256']==sha(binding['model'])
        assert cp['updates_completed']==report['updates_completed']==binding['parent_updates_completed']
        assert cp['total_actor_steps']==report['total_actor_steps'] and cp['retention_weights']==[0.,0.]
        assert len(cp['consumed_rollouts'])==cp['updates_completed'] and len(set(cp['consumed_rollouts']))==len(cp['consumed_rollouts'])
        spec=model['attention'];assert cp['architecture']==spec['version']
        for name in ('actor','value'):
            module=CausalAttention(model[name],spec[name],spec['heads'],spec['window']).to('cuda')
            assert module.state_dict().keys()==cp[name].keys()
            assert all(v.is_cuda and cp[name][k].is_cuda and torch.equal(v,cp[name][k]) for k,v in module.state_dict().items())
            state=cp[name+'_optimizer'];assert state['state'],'Missing optimizer moments'
            # Actor optimizer also has the learned log_std parameter.
            for value in state['state'].values():
                assert value['exp_avg'].is_cuda and value['exp_avg_sq'].is_cuda
                assert torch.isfinite(value['exp_avg']).all() and torch.isfinite(value['exp_avg_sq']).all()
                assert float(value['step'])>0
            del module
        assert torch.equal(torch.tensor(model['log_std'],device='cuda'),cp['log_std'])
        results.append(dict(model=binding['id'],updates_completed=cp['updates_completed'],total_actor_steps=cp['total_actor_steps'],
                            consumed_rollouts=len(cp['consumed_rollouts']),checkpoint_sha256=sha(binding['resume_checkpoint'])))
        del cp
    save(a.out,dict(device='cuda',gpu=torch.cuda.get_device_name(0),checks=results,models_sha256=sha(a.models)))
    print(json.dumps(results))


if __name__=='__main__':main()
