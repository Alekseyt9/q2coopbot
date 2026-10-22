"""Exact BC movement-weight/aim-weight components, CUDA only, no refit."""
import argparse
import copy
import json
import pathlib
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','trained','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--scope',choices=('movement-weights','aim-weights'),required=True)
    args=ap.parse_args();assert not args.out.exists()
    import torch
    assert torch.cuda.is_available();torch.set_default_device('cuda');torch.set_num_threads(2)
    from migrate_combat_navigation_cuda import branch
    from combat_spatial_aim import export as export_spatial
    from combat_weapon_state_residual import optimizer
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from ppo_recurrent import durable_json,durable_write
    pins={};models={};cps={}
    for name,root in [('source',args.source),('trained',args.trained)]:
        seal=read(root/'complete.json');pins[str((root/'complete.json').resolve())]=sha(root/'complete.json')
        for file,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
            assert sha(root/file)==seal[key];pins[str((root/file).resolve())]=seal[key]
        models[name]=read(root/'weights.json');cps[name]=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert equal_tree(branch(models[name],'actor').state_dict(),cps[name]['actor'])
    before,after=cps['source'],cps['trained'];assert after['teacher_motion_bc']['actor_adam_reset']
    heads=('base.head.weight','base.head.bias','base.residual.weight','base.residual.bias')
    allowed=torch.zeros(81,dtype=torch.bool,device='cuda');allowed[:2]=True;allowed[63:81]=True
    for key,value in after['actor'].items():
        if key in heads:assert torch.equal(value[~allowed],before['actor'][key][~allowed])
        elif not key.startswith('sampled_motion_aim_branch.'):assert torch.equal(value,before['actor'][key])
    selected=torch.zeros_like(allowed)
    if args.scope=='movement-weights':selected[:2]=True
    else:selected[63:81]=True
    actor=branch(models['source'],'actor');state=actor.state_dict()
    for key in heads:state[key][selected]=after['actor'][key][selected]
    if args.scope=='aim-weights':
        for key in state:
            if key.startswith('sampled_motion_aim_branch.'):state[key].copy_(after['actor'][key])
    actor.load_state_dict(state)
    model=copy.deepcopy(models['source']);base,cell=actor.export();model['actor']=base;model['attention']['actor']=cell;export_spatial(model,actor)
    assert equal_tree(branch(model,'actor').state_dict(),actor.state_dict())
    args.out.mkdir(parents=True)
    save(args.out/'protocol.json',dict(version='teacher_motion_component_fork_v1',scope=args.scope,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],training_performed=False,
        note='Parameter intervention, not output isolation: frozen spatial aim also consumes current movement means. All components need matched fresh native evaluation.'))
    durable_json(args.out/'weights.json',model);fork=copy.deepcopy(before)
    fork.update(actor={k:v.detach().clone() for k,v in actor.state_dict().items()},weights_sha256=sha(args.out/'weights.json'))
    std=torch.nn.Parameter(fork['log_std'].detach().clone())
    fork['actor_optimizer']=optimizer(actor,std,fork['config']['actor_lr'],fork.get('actor_training',{}).get('weapon_state_residual',{}).get('learning_rate',.0003)).state_dict()
    fork['teacher_motion_bc']=dict(after['teacher_motion_bc'],protocol_sha256=sha(args.out/'protocol.json'),component_scope=args.scope,
        trained_weights_sha256=sha(args.trained/'weights.json'),additional_fitting=False)
    durable_write(args.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    assert equal_tree(torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True),fork)
    for path,digest in pins.items():assert sha(path)==digest
    durable_json(args.out/'report.json',dict(state='complete',device='cuda',training_performed=False,promotion=False,
        scope=args.scope,derived_from_bc=True,actor_adam_reset=True,source=str(args.source.resolve()),trained=str(args.trained.resolve()),
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__)))
    durable_json(args.out/'complete.json',dict(version='combat_update_complete_v1',weights_sha256=sha(args.out/'weights.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    print(json.dumps(dict(state='complete',scope=args.scope)))


if __name__=='__main__':main()
