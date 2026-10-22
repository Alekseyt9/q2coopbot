"""CUDA component ablation of a closed BC candidate; no further fitting."""
import argparse
import copy
import json
import pathlib

from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','trained','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--scope',choices=('target','fire','target-fire','weapon'),required=True)
    args=ap.parse_args();assert not args.out.exists()
    import torch
    assert torch.cuda.is_available();torch.set_default_device('cuda');torch.set_num_threads(2)
    from migrate_combat_navigation_cuda import branch
    from combat_spatial_aim import export as export_spatial
    from combat_weapon_state_residual import optimizer
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from ppo_recurrent import durable_json,durable_write
    pins={};models={};cps={}
    for label,root in [('source',args.source),('trained',args.trained)]:
        seal=read(root/'complete.json');pins[str((root/'complete.json').resolve())]=sha(root/'complete.json')
        for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
            path=root/name;assert sha(path)==seal[key];pins[str(path.resolve())]=seal[key]
        models[label]=read(root/'weights.json');cps[label]=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert equal_tree(branch(models[label],'actor').state_dict(),cps[label]['actor'])
    source,trained=cps['source'],cps['trained'];assert trained['joint_decision_bc']['actor_adam_reset']
    keys=('base.head.weight','base.head.bias','base.residual.weight','base.residual.bias')
    for k,value in source['actor'].items():
        if k not in keys:assert torch.equal(value,trained['actor'][k])
    chosen=torch.zeros(81,dtype=torch.bool,device='cuda')
    if args.scope in ('target','target-fire'):chosen[20:29]=True
    if args.scope in ('fire','target-fire'):chosen[4]=True
    if args.scope=='weapon':chosen[8:20]=True
    actor=branch(models['source'],'actor');state=actor.state_dict()
    for k in keys:state[k][chosen]=trained['actor'][k][chosen]
    actor.load_state_dict(state)
    model=copy.deepcopy(models['source']);base,cell=actor.export();model['actor']=base;model['attention']['actor']=cell;export_spatial(model,actor)
    assert equal_tree(branch(model,'actor').state_dict(),actor.state_dict())
    fork=copy.deepcopy(source);fork['actor']={k:v.detach().clone() for k,v in actor.state_dict().items()}
    std=torch.nn.Parameter(fork['log_std'].detach().clone())
    opt=optimizer(actor,std,fork['config']['actor_lr'],fork.get('actor_training',{}).get('weapon_state_residual',{}).get('learning_rate',.0003))
    fork['actor_optimizer']=opt.state_dict()
    fork['joint_decision_bc']=dict(trained['joint_decision_bc'],component_ablation=args.scope,trained_weights_sha256=sha(args.trained/'weights.json'))
    args.out.mkdir(parents=True)
    durable_json(args.out/'weights.json',model);fork['weights_sha256']=sha(args.out/'weights.json')
    durable_write(args.out/'checkpoint.pt',lambda output:torch.save(fork,output))
    assert equal_tree(torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True),fork)
    for path,digest in pins.items():assert sha(path)==digest
    durable_json(args.out/'report.json',dict(state='complete',device='cuda',training_performed=False,promotion=False,scope=args.scope,
        source=str(args.source.resolve()),trained=str(args.trained.resolve()),derived_from_bc=True,actor_adam_reset=True,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
        note='Exact selected BC output rows transplanted into source. No new fitting or quality evidence. Final native evaluation required.'))
    durable_json(args.out/'complete.json',dict(version='combat_update_complete_v1',weights_sha256=sha(args.out/'weights.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    print(json.dumps(dict(state='complete',scope=args.scope)))


if __name__=='__main__':main()
