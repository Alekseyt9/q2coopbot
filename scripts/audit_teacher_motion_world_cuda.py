"""Teacher-forced versus own-mean movement/aim coupling, on CUDA only."""
import argparse
import gzip
import json
import pathlib
from process_combat_architecture_pool import read,save,sha
from train_registered_joint_decisions_cuda import groups


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--labels',type=pathlib.Path,required=True)
    ap.add_argument('--variant',action='append',required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    import torch
    assert torch.cuda.is_available();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from migrate_combat_navigation_cuda import branch
    from combat_precision_head import selected_means,mode_distribution
    from combat_control_coupling import planar_world_input
    from fork_conditional_weapon_exploration_cuda import equal_tree
    pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();digest=sha(path);assert expected is None or expected==digest
        pins[str(path)]=digest;return path
    meta=read(pin(args.labels/'complete.json'));assert meta['state']=='complete' and meta['version']=='registered_teacher_ground_motion_labels_v1'
    for binding in meta['inputs']:pin(binding['path'],binding['sha256'])
    datasets={}
    for split,info in meta['splits'].items():
        with gzip.open(pin(info['path'],info['sha256']),'rt',encoding='utf-8') as f:rows=list(map(json.loads,f))
        segments=groups(rows);length=max(map(len,segments));x=torch.zeros((len(segments),length,1147),device='cuda')
        positions=[];labels=[]
        for b,segment in enumerate(segments):
            x[b,:len(segment)]=torch.tensor([r['features'] for r in segment],dtype=torch.float32,device='cuda')
            for t,r in enumerate(segment):
                if r['label'] is not None:positions.append((b,t));labels.append(r['label'])
        ix=torch.tensor(positions,dtype=torch.long,device='cuda');features=x[ix[:,0],ix[:,1]]
        movement=torch.tensor([l['movement'] for l in labels],dtype=torch.float32,device='cuda')
        angles=torch.tensor([l['aim_degrees'] for l in labels],dtype=torch.float32,device='cuda')
        target=torch.tensor([l['target'] for l in labels],dtype=torch.long,device='cuda')
        mode=torch.tensor([l['mode'] for l in labels],dtype=torch.long,device='cuda')
        moving=torch.tensor([l['kind']=='moving' for l in labels],dtype=torch.bool,device='cuda')
        truth=torch.cat((movement,angles/180.),-1)
        datasets[split]=(x,ix,features,movement,angles,target,mode,moving,truth)
    results={}
    with torch.no_grad():
        for variant in args.variant:
            name,path=variant.split('=',1);root=pathlib.Path(path);assert name not in results
            seal=read(pin(root/'complete.json'))
            for file,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256')]:pin(root/file,seal[key])
            model=read(root/'weights.json');actor=branch(model,'actor');cp=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
            assert equal_tree(actor.state_dict(),cp['actor']);results[name]={}
            for split,d in datasets.items():
                x,ix,features,movement,angles,target,mode,moving,truth=d
                raw=actor(x)[0][ix[:,0],ix[:,1]];own_mode=mode_distribution(raw,target).probs.argmax(-1)
                teacher_move=movement.clamp(-1+1e-6,1-1e-6).atanh()
                forced=selected_means(raw,target,mode,features,teacher_move,actor.sampled_motion_aim_branch)
                own=selected_means(raw,target,own_mode,features,raw[:,:2],actor.sampled_motion_aim_branch)
                forced_angle=forced[:,2:4].tanh()*torch.where(mode==1,15.,180.)[:,None]
                own_angle=own[:,2:4].tanh()*torch.where(own_mode==1,15.,180.)[:,None]
                predicted=torch.cat((raw[:,:2].tanh(),own_angle/180.),-1)
                world_error=(planar_world_input(features,predicted)-planar_world_input(features,truth)).square().mean(-1)
                body_error=(raw[:,:2].tanh()-movement).square().mean(-1)
                balanced=lambda v:float((.5*(v[moving].mean()+v[~moving].mean())).sqrt())
                results[name][split]=dict(labels=len(movement),movement_body_balanced_rmse=balanced(body_error),
                    mean_action_world_input_balanced_rmse=balanced(world_error),
                    teacher_forced_aim_rmse_degrees=float((forced_angle-angles).square().mean().sqrt()),
                    own_mean_and_argmax_mode_aim_rmse_degrees=float((own_angle-angles).square().mean().sqrt()),
                    own_fine_mode_fraction=float(own_mode.float().mean()),teacher_mode_agreement=float((own_mode==mode).float().mean()))
    for path,digest in pins.items():assert sha(path)==digest
    save(args.out,dict(state='complete',version='teacher_motion_world_input_audit_v1',device='cuda',variants=results,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),training_performed=False,promotion=False,
        scope='Matched teacher histories and teacher-selected available targets; no fitting. Contrast teacher-forced movement/mode versus own mean movement/argmax mode. Planar world INPUT basis is a continuous approximation, not PMove velocity/collision/displacement or full stochastic gameplay. No evaluation logs used as training data.'))
    print(json.dumps(results))


if __name__=='__main__':main()
