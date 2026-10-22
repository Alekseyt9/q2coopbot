"""Independent CUDA checkpoint/Adam continuity audit for sealed pilot rounds."""
import argparse
import pathlib
from ppo_recurrent import torch
from migrate_combat_navigation_cuda import branch
from fork_conditional_weapon_exploration_cuda import equal_tree
from process_combat_architecture_pool import read,save,sha


def optimizer_steps(before,after,expected):
    assert before['param_groups']==after['param_groups']
    if not before['state']:
        identifiers={p for group in after['param_groups'] for p in group['params']}
        assert set(after['state'])<=identifiers
        assert bool(after['state']) or expected==0
        for state in after['state'].values():
            assert float(state['step'])==expected
            assert state['exp_avg'].shape==state['exp_avg_sq'].shape
            for value in state.values():
                if torch.is_tensor(value):assert value.is_cuda and bool(torch.isfinite(value).all())
        return dict(states=len(after['state']),step_delta=expected,initialized_from_empty=True)
    assert set(before['state'])==set(after['state']), 'Unexpected optimizer state census'
    deltas=[]
    for key,old in before['state'].items():
        new=after['state'][key];assert set(old)==set(new)
        for name,value in new.items():
            if torch.is_tensor(value):assert value.is_cuda and bool(torch.isfinite(value).all())
        delta=float(new['step'])-float(old['step']);assert delta==expected
        for name in ('exp_avg','exp_avg_sq'):assert old[name].shape==new[name].shape
        deltas.append(delta)
    return dict(states=len(deltas),step_delta=expected)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pilot',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--partial',action='store_true',help='Audit only already sealed rounds, without a training-complete claim')
    args=ap.parse_args();assert torch.cuda.is_available() and not args.out.exists()
    torch.set_default_device('cuda');torch.set_num_threads(2)
    root=args.pilot.resolve();pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path);digest=sha(path);assert expected is None or digest==expected
        pins[str(path.resolve())]=digest;return path
    protocol=read(pin(root/'protocol.json'));assert protocol['cuda_only'] and not protocol['promotion']
    if args.partial and not (root/'complete.json').exists():
        # This progress snapshot is not immutable input; every listed round's
        # actual completion seal is verified independently below.
        rounds=read(root/'progress.json').get('completed',[])
    else:
        seal=read(pin(root/'complete.json'));assert seal['state']=='complete' and seal['phase']=='training_only'
        assert seal['protocol_sha256']==sha(root/'protocol.json');rounds=seal['rounds']
        assert len(rounds)==len(protocol['rounds'])
    assert rounds and [r['round'] for r in rounds]==list(range(1,len(rounds)+1))
    source=pathlib.Path(protocol['source']);seen=set();results=[]
    def checkpoint(path):
        seal=read(pin(path/'complete.json'))
        for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(path/name,seal[key])
        cp=torch.load(path/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert cp['weights_sha256']==seal['weights_sha256']
        model=read(path/'weights.json')
        assert equal_tree(branch(model,'actor').state_dict(),cp['actor'])
        assert equal_tree(branch(model,'value').state_dict(),cp['value'])
        torch.testing.assert_close(torch.tensor(model['log_std'],device='cuda',dtype=torch.float32),cp['log_std'],rtol=0,atol=0)
        return cp
    old=checkpoint(source);config=old['config']
    for round in rounds:
        assert pathlib.Path(round['source']).resolve()==source.resolve()
        fit=read(pin(pathlib.Path(round['fit'])/'complete.json',round['fit_complete_sha256']))
        assert fit['state']=='complete' and fit['device']=='cuda' and pathlib.Path(fit['source']).resolve()==source.resolve()
        corpus=read(pin(pathlib.Path(round['corpus'])/'complete.json',round['corpus_complete_sha256']))
        assert corpus['state']=='complete' and corpus['native_cases']==corpus['cuda_cases']==protocol['rounds'][round['round']-1]['cases']
        assert not seen.intersection(round['seeds']);seen.update(round['seeds'])
        source=pathlib.Path(round['update']);new=checkpoint(source)
        assert new['config']==config and new['anchor_sha256']==old['anchor_sha256'] and new['bank_sha256']==old['bank_sha256']
        assert new['retention_weights']==old['retention_weights']
        changed={}
        for kind in ('actor','value'):
            assert set(new[kind])==set(old[kind])
            assert all(t.is_cuda and bool(torch.isfinite(t).all()) for t in new[kind].values())
            changed[kind]=sum(not torch.equal(t,old[kind][name]) for name,t in new[kind].items())
        accepted=fit['accepted_actor_steps'];assert accepted==round['accepted_actor_steps']
        assert changed['value']>0
        if accepted:assert changed['actor']>0
        else:assert changed['actor']==0 and torch.equal(new['log_std'],old['log_std'])
        actor_opt=optimizer_steps(old['actor_optimizer'],new['actor_optimizer'],accepted)
        value_opt=optimizer_steps(old['value_optimizer'],new['value_optimizer'],config['value_steps'])
        results.append(dict(round=round['round'],accepted_actor_steps=accepted,changed_tensors=changed,
            actor_optimizer=actor_opt,value_optimizer=value_opt,weights_sha256=new['weights_sha256']))
        old=new
    all_complete=len(rounds)==len(protocol['rounds'])
    if all_complete:assert len(seen)==protocol['native_training_budget']
    for p,h in pins.items():assert sha(p)==h
    save(args.out,dict(state='complete',version='registered_ppo_pilot_checkpoint_cuda_audit_v1',device='cuda',
        gpu=torch.cuda.get_device_name(0),rounds=results,all_training_rounds_audited=all_complete,
        training_cases_audited=len(seen),config_objective_and_retention_unchanged=True,
        model_json_checkpoint_exact=True,optimizer_moments_finite_shapes_and_step_continuity=True,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
        training_performed=False,promotion=False,
        scope='CUDA numerical equality of model JSON and saved actor/value/std, finite tensors, unchanged config/objective, exact Adam state census/groups/shapes and per-state step increments. Completed phases only; partial mode is not whole-pilot completion. Does not prove arbitrary mid-step resume, quality improvement or optimality.'))
    print(dict(state='complete',rounds=len(results),training_cases=len(seen),all_rounds=all_complete,optimizer_step_continuity=True))


if __name__=='__main__':main()
