"""Refit only frozen-encoder world predictors: proposed vs applied action inputs."""
import argparse
import collections
import copy
import json
import math
import pathlib
import torch
from torch import nn
from train_combat_hjepa import sha,save,export,serialized_forward
from train_combat_hjepa_branch_preferences import network
from audit_combat_hjepa_world_actions import primitive


def load(paths,split,action_space):
    groups=collections.defaultdict(list);inputs={};seeds=set();seen=set()
    for path in paths:
        meta_path=pathlib.Path(str(path)+'.meta.json');meta=json.loads(meta_path.read_text())
        if meta['version']!='hjepa_data_v1' or meta['feature_version']!='combat_features_v6':raise ValueError('world data contract')
        inputs[str(path.resolve())]=sha(path);inputs[str(meta_path.resolve())]=sha(meta_path)
        sources={}
        for name,digest in meta['sources_sha256'].items():
            p=pathlib.Path(name)
            if sha(p)!=digest:raise ValueError('native source changed')
            if p.name!='steps.jsonl':continue
            report_path=p.parent.parent.parent/'report.json';report=json.loads(report_path.read_text())
            worker=p.parent.parent.resolve()
            result=[r for r in report['results'] if pathlib.Path(r['root']).resolve()==worker]
            if (not report['capture_complete'] or not report['provenance_valid'] or len(result)!=1 or
                    not result[0]['capture_valid'] or result[0]['generated_fixture']['split']!=split):
                raise ValueError('native world source split/receipt mismatch')
            inputs[str(p.resolve())]=digest;inputs[str(report_path.resolve())]=sha(report_path)
            indexed={}
            for line in p.read_text().splitlines():
                s=json.loads(line);frame=s['observation']['identity']['frame']
                if frame in indexed:raise ValueError('ambiguous native frame')
                indexed[frame]=s
            sources[digest]=(indexed,result[0]['seed'])
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        if len(rows)!=meta['rows']:raise ValueError('world row count changed')
        for r in rows:
            source=r['source_sha256'];step,seed=sources[source][0][r['frame']],sources[source][1]
            if step['owner']!='provider':continue
            unique=(source,r['frame'])
            if unique in seen:raise ValueError('duplicate world transition')
            seen.add(unique);seeds.add(seed)
            if (not step.get('native_step') or not step['server_execution']['matched'] or
                    not step['server_execution']['window_exclusive']):raise ValueError('unverified native world transition')
            applied=primitive(step['applied_action'])
            if any(abs(x-y)>1e-12 for x,y in zip(applied,r['action'])):raise ValueError('source AppliedAction mismatch')
            # Source digest namespaces parallel actors; no episode/frame renumbering.
            item=dict(r,action=primitive(step['action']) if action_space=='proposed' else applied)
            groups[(source,r['episode'])].append(item)
    clips=[]
    for group in groups.values():
        group.sort(key=lambda r:r['frame'])
        for start in range(len(group)-15):
            chunk=group[start:start+16]
            if any(b['frame']!=a['frame']+1 or a['next_features']!=b['features'] for a,b in zip(chunk,chunk[1:])):continue
            clips.append(chunk)
    if not clips:raise ValueError('no contiguous provider clips')
    x=torch.tensor([[r['features'] for r in c]+[c[-1]['next_features']] for c in clips],device='cuda')
    actions=torch.tensor([[r['action'] for r in c] for c in clips],device='cuda')
    if x.shape[-1]!=845 or actions.shape[-1]!=8 or not torch.isfinite(x).all() or not torch.isfinite(actions).all():
        raise ValueError('invalid predictor inputs')
    return x,actions,seeds,inputs,len(seen)


def frozen_data(model,x,a):
    result=[];z=x
    with torch.no_grad():
        for level in model['levels']:
            # Float32 fitting with exactly frozen serialized encoder weights.
            for index,layer in enumerate(level['encoder']):
                z=torch.nn.functional.linear(z,torch.tensor(layer['weight'],device=z.device,dtype=z.dtype),
                                              torch.tensor(layer['bias'],device=z.device,dtype=z.dtype))
                if index+1<len(level['encoder']):z=z.relu()
            stride=level['stride']
            result.append((torch.cat((z[:,0],a[:,:stride].flatten(1)),-1),z[:,stride]))
    return result


def loss(nets,data,index=None,utilities=None,utility_weight=0.):
    values=[];details=[]
    for level,(net,(x,y)) in enumerate(zip(nets,data)):
        if index is not None:x,y=x[index],y[index]
        predicted=net(x);latent=(predicted-y).square().mean()
        utility=(utilities[level](predicted)-utilities[level](y)).square().mean() if utilities is not None else latent.new_zeros(())
        values.append(latent+utility_weight*utility)
        details.append(dict(latent_mse=float(latent.detach()),utility_mse=float(utility.detach())))
    return sum(values),details


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',type=pathlib.Path,required=True)
    ap.add_argument('--validation',type=pathlib.Path,action='append',required=True)
    ap.add_argument('--parent',type=pathlib.Path,required=True)
    ap.add_argument('--action-space',choices=['applied','proposed'],required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--utility-weight',type=float,default=0.,help='fixed frozen-utility forecast loss coefficient')
    a=ap.parse_args()
    if not math.isfinite(a.utility_weight) or not 0<=a.utility_weight<=10:raise ValueError('invalid utility loss coefficient')
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    torch.set_num_threads(2);torch.manual_seed(261617);torch.cuda.manual_seed_all(261617)
    parent=json.loads(a.parent.read_text())
    tx,ta,train_seeds,train_inputs,train_rows=load([a.train],'train',a.action_space)
    vx,va,val_seeds,val_inputs,val_rows=load(a.validation,'validation',a.action_space)
    if train_seeds&val_seeds:raise ValueError('TRAIN/validation overlap')
    train=frozen_data(parent,tx,ta);val=frozen_data(parent,vx,va)
    nets=nn.ModuleList([network(l['predictor']) for l in parent['levels']]).cuda()
    utilities=nn.ModuleList([network(l['utility']) for l in parent['levels']]).cuda()
    utilities.requires_grad_(False)
    objective=lambda data,index=None:loss(nets,data,index,utilities,a.utility_weight)
    torch.manual_seed(261617);torch.cuda.manual_seed_all(261617)
    opt=torch.optim.Adam(nets.parameters(),lr=1e-3)
    with torch.no_grad():before=dict(train=objective(train)[1],validation=objective(val)[1])
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'trainer-frozen.py').write_bytes(pathlib.Path(__file__).read_bytes())
    history=[]
    for epoch in range(200):
        for index in torch.randperm(len(tx),device='cuda').split(128):
            opt.zero_grad();value,_=objective(train,index)
            if not torch.isfinite(value):raise ValueError('nonfinite predictor loss')
            value.backward();nn.utils.clip_grad_norm_(nets.parameters(),1.);opt.step()
        if epoch%50==0 or epoch==199:
            with torch.no_grad():row=dict(epoch=epoch+1,train=objective(train)[1],validation=objective(val)[1])
            history.append(row);print(json.dumps(row),flush=True)
    model=copy.deepcopy(parent)
    for i,net in enumerate(nets):model['levels'][i]['predictor']=export(net)
    if a.action_space=='proposed':model['world_action_space']='proposed_primitive_v1'
    else:model.pop('world_action_space',None)
    for old,new in zip(parent['levels'],model['levels']):
        if any(old[k]!=new[k] for k in ['stride','encoder','utility']):raise ValueError('frozen world component changed')
    ignored={'levels','world_action_space'}
    if {k:v for k,v in parent.items() if k not in ignored}!={k:v for k,v in model.items() if k not in ignored}:
        raise ValueError('motor/actor/weapon fields changed')
    save(a.out/'model.json',model)
    torch.save(dict(version='hjepa_world_predictor_fit_v2',action_space=a.action_space,utility_weight=a.utility_weight,predictors=nets.state_dict(),
        optimizer=opt.state_dict(),epochs=200,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),a.out/'checkpoint.pt')
    # Native replay fixture uses the runtime's constant-action hypothetical rollouts.
    x=vx[:8,0].cpu().double();actions=va[:8,0].cpu().double();z=x;latents=[];predictions=[]
    with torch.no_grad():
        for level in model['levels']:
            z=serialized_forward(z,level['encoder']);latents.append(z.tolist())
            predictions.append(serialized_forward(torch.cat((z,actions.repeat(1,level['stride'])),-1),level['predictor']).tolist())
        actor=serialized_forward(x,model['motor'])
    save(a.out/'parity.json',dict(features=x.tolist(),actions=actions.tolist(),latents=latents,predictions=predictions,actor=actor.tolist()))
    save(a.out/'report.json',dict(version='hjepa_world_predictor_fit_v2',promotion=False,action_space=a.action_space,utility_weight=a.utility_weight,
        epochs=200,seed=261617,train_rows=train_rows,validation_rows=val_rows,train_clips=len(tx),validation_clips=len(vx),
        train_worlds=len(train_seeds),validation_worlds=len(val_seeds),parent_sha256=sha(a.parent),model_sha256=sha(a.out/'model.json'),
        train_inputs_sha256=train_inputs,validation_inputs_sha256=val_inputs,before=before,history=history,
        scope='Predictors only, frozen encoders/utilities/motor/weapon head. Same fixed budget, no validation checkpoint selection. '
              'Provider-only source-namespaced actual causal clips, offline supervised dynamics, not PPO data.'))


if __name__=='__main__':main()
