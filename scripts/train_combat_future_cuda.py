"""Fixed paired action-conditioned/blind observed-future experiment; CUDA only."""
import argparse
import copy
import math
import os
import pathlib
import statistics
import time

from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--seed',type=int,default=20261022)
    args = ap.parse_args(); assert not args.out.exists()
    assert 0 <= args.seed < 2147483646
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
    torch.set_default_device('cuda');torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    pins = {}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();digest=sha(path);assert expected is None or expected==digest,str(path)
        pins[str(path)]=digest;return path
    meta=read(pin(args.data/'complete.json'));assert meta['state']=='complete' and meta['version']=='combat_observed_future_dataset_v1'
    assert meta['inputs_width']==1153 and meta['dt_game_seconds']==.1 and meta['displacement_scale']==64
    for item in meta['inputs']:pin(item['path'],item['sha256'])
    data={};seen=set()
    for split,info in meta['splits'].items():
        seeds=set(info['seeds']);assert not seen.intersection(seeds);seen.update(seeds)
        with np.load(pin(info['path'],info['sha256']),allow_pickle=False) as arrays:
            d={k:torch.as_tensor(arrays[k],device='cuda') for k in arrays.files}
        assert d['x'].shape==(info['rows'],1153) and bool(torch.isfinite(d['x']).all())
        assert bool(d['motion_mask'][:,0].all());d['blind']=d['x'].clone();d['blind'][:,-6:]=0
        data[split]=d
    assert set(data)=={'train','development'}
    def network():return torch.nn.Sequential(torch.nn.Linear(1153,128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,43)).to('cuda')
    initial=network()
    with torch.no_grad():initial[-1].weight.zero_();initial[-1].bias.zero_()
    models={name:copy.deepcopy(initial) for name in ('action','blind')}
    assert all(torch.equal(a,b) for a,b in zip(models['action'].parameters(),models['blind'].parameters()))
    opts={name:torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001) for name,net in models.items()}
    generator=torch.Generator(device='cuda').manual_seed(args.seed+1)
    draws=torch.randint(len(data['train']['x']),(2000,256),generator=generator,device='cuda')
    def outputs(net,d,ix=None,blind=False):
        x=d['blind' if blind else 'x'] if ix is None else d['blind' if blind else 'x'][ix]
        raw=net(x);cv=d['cv'] if ix is None else d['cv'][ix]
        return cv+raw[:,:27].reshape(-1,9,3),raw[:,27:35],raw[:,35:43]
    def loss(net,d,ix,blind):
        motion,present,clear=outputs(net,d,ix,blind)
        m=d['motion_mask'][ix];p=d['present_mask'][ix];c=d['clear_mask'][ix]
        motion_loss=torch.nn.functional.smooth_l1_loss(motion[m],d['target'][ix][m],beta=.1)
        present_loss=torch.nn.functional.binary_cross_entropy_with_logits(present[p],d['present'][ix][p]) if bool(p.any()) else present.sum()*0
        clear_loss=torch.nn.functional.binary_cross_entropy_with_logits(clear[c],d['clear'][ix][c]) if bool(c.any()) else clear.sum()*0
        return motion_loss+.1*present_loss+.1*clear_loss
    def rmse(error,mask):
        return float(error[mask].square().mean().sqrt()) if bool(mask.any()) else None
    def binary(prob,truth,mask):
        if not bool(mask.any()):return dict(count=0)
        p,t=prob[mask],truth[mask];positive=p>=.5;actual=t==1
        tp=int((positive&actual).sum());fp=int((positive&~actual).sum());fn=int((~positive&actual).sum())
        return dict(count=len(p),brier=float((p-t).square().mean()),true_positive=tp,false_positive=fp,false_negative=fn,
                    precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)
    def metrics(d,motion,present,clear,select=None):
        if select is None:select=torch.ones(len(d['x']),dtype=torch.bool,device='cuda')
        error=(motion-d['target'])*64;m=d['motion_mask']&select[:,None]
        enemy=m.clone();enemy[:,0]=False
        known=enemy&d['velocity_known'];unknown=enemy&~d['velocity_known']
        pmove=select&d['pmove_mask']
        return dict(rows=int(select.sum()),self_rmse=rmse(error[:,0],select),enemy_rmse=rmse(error,enemy),
            self_axis_rmse=[rmse(error[:,0,i],select) for i in range(3)],
            enemy_axis_rmse=[rmse(error[...,i],enemy) for i in range(3)],
            known_velocity_enemy_rmse=rmse(error,known),unknown_velocity_enemy_rmse=rmse(error,unknown),
            action_sensitive_self_rmse=rmse(error[:,0],select&d['action_sensitive']),
            pmove_common_self_rmse=rmse(error[:,0],pmove),pmove_rows=int(pmove.sum()),
            track_present=binary(present,d['present'],d['present_mask']&select[:,None]),
            clear_shot=binary(clear,d['clear'],d['clear_mask']&select[:,None]),
            clear_common=binary(clear,d['clear'],d['clear_mask']&d['clear_cv_mask']&select[:,None]))
    @torch.no_grad()
    def evaluate():
        result={}
        for split,d in data.items():
            predictions={name:(m,p.sigmoid(),c.sigmoid()) for name,net in models.items()
                         for m,p,c in [outputs(net,d,blind=name=='blind')]}
            predictions.update(cv=(d['cv'],torch.ones_like(d['present']),d['clear_cv']),
                               zero=(torch.zeros_like(d['cv']),torch.ones_like(d['present']),d['clear_cv']))
            methods={}
            for name,(motion,present,clear) in predictions.items():
                pooled=metrics(d,motion,present,clear);cases=[]
                for i,case in enumerate(meta['splits'][split]['cases']):
                    cases.append(dict(seed=case['seed'],map=case['map'],episode=case['episode'],
                                      **metrics(d,motion,present,clear,d['case_index']==i)))
                case_weighted={}
                for key in ('self_rmse','enemy_rmse','action_sensitive_self_rmse'):
                    values=[c[key] for c in cases if c[key] is not None]
                    case_weighted[key]=dict(rmse=math.sqrt(sum(v*v for v in values)/len(values)),
                        mean_case_rmse=statistics.mean(values),case_rmse_std=statistics.stdev(values) if len(values)>1 else 0,
                        min_case_rmse=min(values),max_case_rmse=max(values))
                maps={name:metrics(d,motion,present,clear,torch.tensor([meta['splits'][split]['cases'][int(i)]['map']==name for i in d['case_index'].tolist()],device='cuda'))
                      for name in sorted({c['map'] for c in cases})}
                types={name:rmse((motion[:,1:]-d['target'][:,1:])*64,d['motion_mask'][:,1:]&(d['class_id']==i)) for name,i in meta['classes'].items()}
                methods[name]=dict(pooled=pooled,case_weighted=case_weighted,cases=cases,maps=maps,enemy_type_rmse=types)
            pm=d['pmove_mask'];methods['pmove']=dict(rows=int(pm.sum()),self_rmse=rmse((d['pmove']-d['target'][:,0])*64,pm))
            result[split]=methods
        return result
    args.out.mkdir(parents=True)
    save(args.out/'protocol.json',dict(version='combat_observed_future_pair_v1',seed=args.seed,batch_seed=args.seed+1,
        steps=2000,batch_size=256,learning_rate=.001,weight_decay=.0001,hidden=[128,128],inputs=1153,outputs=43,
        residual_baseline='constant_velocity',displacement_scale=64,motion_loss='masked SmoothL1 beta0.1',
        motion_weight=1,present_bce_weight=.1,clear_bce_weight=.1,identical_initial_weights=True,identical_train_minibatches=True,
        zero_output_initialization=True,selection_on_development=False,checkpoint='fixed final',actor_changed=False,
        dataset=str(args.data.resolve()),inputs_sha256=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__)))
    before=evaluate();history=[];start=time.perf_counter()
    for step,ix in enumerate(draws):
        values={}
        for name,net in models.items():
            opts[name].zero_grad();value=loss(net,data['train'],ix,name=='blind');assert bool(torch.isfinite(value))
            value.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opts[name].step()
            values[name]=float(value.detach())
        if (step+1)%250==0:
            history.append(dict(step=step+1,train_losses=values));save(args.out/'progress.json',dict(stage='cuda-training',step=step+1,steps=2000))
    after=evaluate();torch.cuda.synchronize()
    checkpoint=dict(version='combat_observed_future_pair_v1',models={n:m.state_dict() for n,m in models.items()},
        optimizers={n:o.state_dict() for n,o in opts.items()},steps=2000,seed=args.seed,train_draws=draws,
        dataset_sha256=sha(args.data/'complete.json'),protocol_sha256=sha(args.out/'protocol.json'))
    torch.save(checkpoint,args.out/'checkpoint.pt')
    restored=torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert all(torch.equal(v,restored['models'][n][k]) for n,m in models.items() for k,v in m.state_dict().items())
    for p,h in pins.items():assert sha(p)==h
    report=dict(state='complete',device='cuda',steps=2000,seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in initial.parameters()),
        training_performed=True,actor_changed=False,promotion=False,before=before,after=after,history=history,
        protocol_sha256=sha(args.out/'protocol.json'),script_sha256=sha(__file__),
        scope='One-step observed motion/track prediction on held-out cases; units are world units per100ms, axes RMSE not vector norm. No future actions or native hidden state input, no combat-policy update or native-quality claim.')
    save(args.out/'report.json',report)
    (args.out/'trainer.py').write_bytes(pathlib.Path(__file__).read_bytes())
    save(args.out/'complete.json',dict(state='complete',version='combat_observed_future_pair_v1',device='cuda',
        checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json'),protocol_sha256=sha(args.out/'protocol.json'),
        training_performed=True,actor_changed=False,promotion=False))
    save(args.out/'progress.json',dict(stage='complete',steps=2000))
    print(args.out/'complete.json')


if __name__=='__main__':main()
