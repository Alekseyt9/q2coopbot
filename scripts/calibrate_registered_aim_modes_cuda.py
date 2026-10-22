"""Fit only coarse/fine logit intercepts on TRAIN, CUDA, no runtime rule."""
import argparse
import copy
import gzip
import json
import pathlib
from process_combat_architecture_pool import read,save,sha
from train_registered_joint_decisions_cuda import groups


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','labels','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    import torch
    assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
    torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from migrate_combat_navigation_cuda import branch
    from combat_precision_head import mode_distribution
    from combat_spatial_aim import export as export_spatial
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from ppo_recurrent import durable_json,durable_write
    pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();digest=sha(path);assert expected is None or expected==digest
        pins[str(path)]=digest;return path
    seal=read(pin(args.source/'complete.json'))
    for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(args.source/name,seal[key])
    model=read(args.source/'weights.json');cp=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    actor=branch(model,'actor');assert equal_tree(actor.state_dict(),cp['actor']) and not cp['actor_optimizer']['state']
    assert cp['weights_sha256']==seal['weights_sha256'] and model['aim_mode_head']=='combat_target_coarse_fine_aim_v1'
    meta=read(pin(args.labels/'complete.json'));assert meta['state']=='complete' and meta['version']=='registered_teacher_ground_motion_labels_v1' and meta['test_deferred']
    for item in meta['inputs']:pin(item['path'],item['sha256'])
    def load(split):
        info=meta['splits'][split]
        with gzip.open(pin(info['path'],info['sha256']),'rt',encoding='utf-8') as f:rows=list(map(json.loads,f))
        segments=groups(rows);length=max(map(len,segments));x=torch.zeros((len(segments),length,1147),device='cuda');positions=[];labels=[]
        for b,segment in enumerate(segments):
            x[b,:len(segment)]=torch.tensor([r['features'] for r in segment],dtype=torch.float32,device='cuda')
            for t,r in enumerate(segment):
                if r['label'] is not None:positions.append((b,t));labels.append(r['label'])
        ix=torch.tensor(positions,dtype=torch.long,device='cuda')
        target=torch.tensor([l['target'] for l in labels],dtype=torch.long,device='cuda')
        mode=torch.tensor([l['mode'] for l in labels],dtype=torch.long,device='cuda')
        return x,ix,target,mode,{r['seed'] for r in rows}
    def predict(data):
        x,ix,target,mode,_=data;raw=actor(x)[0][ix[:,0],ix[:,1]]
        return raw,mode_distribution(raw,target)
    def metrics(distribution,mode):
        return dict(labels=len(mode),natural_cross_entropy=float(-distribution.log_prob(mode).mean()),
            coarse_label_fraction=float((mode==0).float().mean()),coarse_probability_mean=float(distribution.probs[:,0].mean()),
            coarse_argmax_fraction=float((distribution.probs.argmax(-1)==0).float().mean()),
            mode_accuracy=float((distribution.probs.argmax(-1)==mode).float().mean()))
    with torch.no_grad():
        train=load('train');raw,distribution=predict(train);before_train=metrics(distribution,train[3])
    args.out.mkdir(parents=True)
    save(args.out/'protocol.json',dict(version='registered_aim_mode_intercept_calibration_v1',source=str(args.source.resolve()),labels=str(args.labels.resolve()),
        fit_split='train',selection_on_development=False,algorithm='64-step CUDA float64 bisection of natural Bernoulli NLL derivative per observed target;skip single-class targets',
        calibrated_parameters='Only corresponding base.head.bias mode pairs63+2*target; symmetric half shift',
        component='aim-weights',capacity_changed=False,training_performed=True,promotion=False,inputs=[dict(path=p,sha256=h) for p,h in pins.items()]))
    changes=[];mask=torch.zeros(81,dtype=torch.bool,device='cuda')
    initial={k:v.detach().clone() for k,v in actor.state_dict().items()}
    with torch.no_grad():
        for target in train[2].unique():
            selected=train[2]==target;mode=train[3][selected];counts=torch.bincount(mode,minlength=2)
            if not bool((counts>0).all()):continue
            t=int(target);logits=raw[selected,63+2*t:65+2*t].double();difference=logits[:,0]-logits[:,1]
            truth=(mode==0).double().mean();span=difference.abs().max()+20.
            low=-span;high=span
            for _ in range(64):
                middle=(low+high)/2;probability=(difference+middle).sigmoid().mean()
                low=torch.where(probability<truth,middle,low);high=torch.where(probability>=truth,middle,high)
            shift=((low+high)/2).float();pair=63+2*t
            actor.base.head.bias[pair]+=shift/2;actor.base.head.bias[pair+1]-=shift/2;mask[pair:pair+2]=True
            changes.append(dict(target=t,coarse=int(counts[0]),fine=int(counts[1]),logit_difference_shift=float(shift)))
        assert changes
        _,after_distribution=predict(train);after_train=metrics(after_distribution,train[3])
        assert after_train['natural_cross_entropy']<=before_train['natural_cross_entropy']+1e-6
        assert abs(after_train['coarse_probability_mean']-after_train['coarse_label_fraction'])<1e-6
        # Development is read only AFTER the complete train-only fit.
        dev=load('development');assert not train[4].intersection(dev[4]);_,after_dev_distribution=predict(dev)
        after_dev=metrics(after_dev_distribution,dev[3])
        calibrated_bias=actor.base.head.bias.detach().clone();actor.base.head.bias.copy_(initial['base.head.bias'])
        _,before_dev_distribution=predict(dev);before_dev=metrics(before_dev_distribution,dev[3]);actor.base.head.bias.copy_(calibrated_bias)
        state=actor.state_dict()
        for k,v in state.items():
            if k=='base.head.bias':assert torch.equal(v[~mask],initial[k][~mask])
            else:assert torch.equal(v,initial[k]),k
    updated=copy.deepcopy(model);base,cell=actor.export();updated['actor']=base;updated['attention']['actor']=cell;export_spatial(updated,actor)
    assert equal_tree(branch(updated,'actor').state_dict(),state)
    durable_json(args.out/'weights.json',updated);fork=copy.deepcopy(cp)
    fork.update(actor={k:v.detach().clone() for k,v in state.items()},weights_sha256=sha(args.out/'weights.json'))
    fork['teacher_motion_bc']=dict(cp['teacher_motion_bc'],protocol_sha256=sha(args.out/'protocol.json'))
    fork['aim_mode_calibration']=dict(source_weights_sha256=seal['weights_sha256'],fit_split='train',changes=changes,actor_adam_reset=True,requires_fresh_own_policy_rollouts=True)
    durable_write(args.out/'checkpoint.pt',lambda f:torch.save(fork,f));assert equal_tree(torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True),fork)
    for p,h in pins.items():assert sha(p)==h
    durable_write(args.out/'calibrator.py',lambda f:f.write(pathlib.Path(__file__).read_bytes()))
    durable_json(args.out/'report.json',dict(state='complete',device='cuda',training_performed=True,promotion=False,
        before=dict(train=before_train,development=before_dev),after=dict(train=after_train,development=after_dev),changes=changes,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
        scope='Train-only one-dimensional natural-NLL mode probability calibration; no validation fitting or new layers. All other actor/value/std/optimizer states unchanged. Teacher target fixed for calibration; live behavior still needs matched native evaluation.'))
    durable_json(args.out/'complete.json',dict(version='combat_update_complete_v1',weights_sha256=sha(args.out/'weights.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    print(json.dumps(dict(state='complete',changes=changes,before=dict(train=before_train,development=before_dev),after=dict(train=after_train,development=after_dev))))


if __name__=='__main__':main()
