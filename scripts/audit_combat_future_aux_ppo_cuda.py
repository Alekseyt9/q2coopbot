"""Initial-state and legacy/zero-weight parity for the auxiliary PPO pilot."""
import argparse
import pathlib

from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args();assert not a.out.exists()
    import torch
    assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
    from fork_conditional_weapon_exploration_cuda import equal_tree
    r=a.root.resolve();protocol=read(r/'protocol.json');assert read(r/'training-complete.json')['state']=='complete'
    source=pathlib.Path(protocol['source']);seal=read(source/'complete.json')
    original=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=True)
    boundaries={};checkpoints={};reports={};inputs=[]
    for name in ('legacy','control','candidate'):
        root=r/name;closed=read(root/'complete.json')
        for f,k in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):
            assert sha(root/f)==closed[k];inputs.append(dict(path=str(root/f),sha256=closed[k]))
        report=read(root/'report.json');assert report['device']=='cuda' and report['actor_steps']<=10
        assert report['behavior_sha256']==seal['weights_sha256'] and report['resume_sha256']==sha(source/'checkpoint.pt')
        boundary=torch.load(r/(name+'-initial.pt'),map_location='cuda',weights_only=True)
        assert sha(r/(name+'-initial.pt'))==report['initial_boundary_sha256']
        for key in ('actor','value','log_std','actor_optimizer','value_optimizer','rng','cuda_rng'):
            assert equal_tree(boundary[key],original[key]),('source boundary',name,key)
        assert boundary['rollout_sha256'] not in original['consumed_rollouts']
        boundaries[name]=boundary;reports[name]=report
        checkpoints[name]=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
    for key in ('decoder','optimizer','decoder_steps'):
        assert equal_tree(boundaries['control']['future_auxiliary'][key],boundaries['candidate']['future_auxiliary'][key])
    assert sha(r/'legacy/weights.json')==sha(r/'control/weights.json'),'Weight0 changed runtime policy'
    for key in ('actor','value','log_std','actor_optimizer','value_optimizer','rng','cuda_rng','consumed_rollouts','total_actor_steps'):
        assert equal_tree(checkpoints['legacy'][key],checkpoints['control'][key]),('zero effect',key)
    assert equal_tree(checkpoints['control']['value'],checkpoints['candidate']['value'])
    assert equal_tree(checkpoints['control']['value_optimizer'],checkpoints['candidate']['value_optimizer'])
    assert not equal_tree(checkpoints['control']['actor'],checkpoints['candidate']['actor'])
    summary={}
    for name in ('legacy','control','candidate'):
        report=reports[name];cp=checkpoints[name];f=report['future_auxiliary']
        if f is not None:
            assert f['spec']['weight']==(0 if name=='control' else protocol['future_coefficients'][1])
            assert f['after']['decoder_steps']==report['actor_steps']
            assert cp['future_auxiliary']['decoder_steps']==report['actor_steps']
        summary[name]=dict(actor_steps=report['actor_steps'],kl=report['final_approx_kl'],seconds=report['seconds'],
            future_before=None if f is None else f['before'],future_after=None if f is None else f['after'])
    for item in protocol['inputs']:
        path=pathlib.Path(item['path'])
        # Code used by a closed fit is preserved beside its receipts. Source
        # files may evolve after completion; verify the exact archived bytes.
        if path.name=='ppo_recurrent.py':path=r/'legacy/trainer.py'
        elif path.name=='combat_future_auxiliary.py':path=r/'future-helper.py'
        assert sha(path)==item['sha256']
    save(a.out,dict(state='complete',version=protocol['version'],device='cuda',initial_actor_critic_adam_rng_exact=True,
        initial_decoder_exact=True,legacy_zero_runtime_weights_exact=True,legacy_zero_actor_adam_rng_exact=True,
        candidate_runtime_weights_different=True,critic_and_optimizer_exact=True,training_data_unconsumed=True,
        variants=summary,inputs=inputs,script_sha256=sha(__file__),training_performed=False,promotion=False,
        scope='Equal initial policy/critic/optimizer/RNG and TRAIN data. Weight0 exactly reproduces legacy weights/state. Candidate differs only by explicit future objective fork, decoder training remains separate. No native-quality evidence yet.'))
    print(a.out)


if __name__=='__main__':main()
