"""Fixed-budget CUDA native-effect forecasts, with a matched action ablation."""
import argparse
import copy
import json
import pathlib
import torch
from torch import nn
from train_combat_hjepa import sha, save, export, network
from train_combat_hjepa_branch_preferences import network as imported_network


def load_data(root):
    meta=json.loads((root/'meta.json').read_text())
    if meta['version']!='hjepa_native_effects_v1' or meta['feature_width']!=871 or meta['action_width']!=20:
        raise ValueError('native effects contract')
    for name,digest in meta['sources_sha256'].items():
        if sha(pathlib.Path(name))!=digest:raise ValueError('changed native evidence')
    data={};worlds={}
    for role in ['fit','internal-holdout']:
        path=root/f'{role}.jsonl'
        if sha(path)!=meta['data_sha256'][role]:raise ValueError('effect data changed')
        groups={h:[] for h in [1,4,16]};worlds[role]=set();seen=set()
        for line in path.open():
            row=json.loads(line);h=row['horizon'];key=(row['seed'],row['frame'],h)
            if key in seen or row['partition']!=role or h not in groups:raise ValueError('duplicate/wrong effect row')
            seen.add(key);groups[h].append(row);worlds[role].add(row['seed'])
        for h,rows in groups.items():
            x=torch.tensor([r['features'] for r in rows],device='cuda')
            actions=torch.tensor([r['actions'] for r in rows],device='cuda').flatten(1)
            y=torch.tensor([r['effects'] for r in rows],device='cuda')
            if x.shape!=(len(rows),871) or actions.shape!=(len(rows),20*h) or y.shape!=(len(rows),6):
                raise ValueError('invalid effect dimensions')
            if not all(torch.isfinite(t).all() for t in [x,actions,y]):raise ValueError('nonfinite effects')
            data[role,h]=(x,actions,y,torch.tensor([r['seed'] for r in rows],device='cuda'))
    if worlds['fit']&worlds['internal-holdout'] or len(worlds['fit'])!=48 or len(worlds['internal-holdout'])!=16:
        raise ValueError('source-world partition changed')
    return data,meta


def scores(predicted,target,worlds,weights):
    effect_error=(predicted-target).square().mean(-1)
    reward_error=((predicted-target)@weights).square()
    return dict(rows=len(target),effect_mse=float(effect_error.mean()),reward_mse=float(reward_error.mean()),
                per_effect_mse=(predicted-target).square().mean(0).tolist(),
                world_mean_effect_mse=float(torch.stack([effect_error[worlds==s].mean() for s in worlds.unique()]).mean()),
                world_mean_reward_mse=float(torch.stack([reward_error[worlds==s].mean() for s in worlds.unique()]).mean()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=pathlib.Path,required=True)
    ap.add_argument('--parent',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args()
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    torch.set_num_threads(2);torch.manual_seed(261920);torch.cuda.manual_seed_all(261920)
    data,meta=load_data(a.data);parent=json.loads(a.parent.read_text())
    encoders=nn.ModuleList([imported_network(l['encoder']) for l in parent['levels']]).cuda()
    encoders.requires_grad_(False);normalization={};prepared={}
    with torch.no_grad():
        for level,h in enumerate([1,4,16]):
            for role in ['fit','internal-holdout']:
                features,actions,y,worlds=data[role,h];z=features[:,:845]
                for encoder in encoders[:level+1]:z=encoder(z)
                if role=='fit':normalization[h]=(z.mean(0),z.std(0,unbiased=False).clamp_min(.05))
                mean,scale=normalization[h]
                prepared[role,h]=(torch.cat(((z-mean)/scale,features[:,845:],actions),-1),y,worlds)
    a.out.mkdir(exist_ok=False,parents=True)
    (a.out/'trainer-frozen.py').write_bytes(pathlib.Path(__file__).read_bytes())
    weights=torch.tensor(meta['reward_weights'],device='cuda')
    results={};models={};history={};checkpoints={}
    for h in [1,4,16]:
        torch.manual_seed(261920+h);torch.cuda.manual_seed_all(261920+h)
        initial=network(58+20*h,6,hidden=64).cuda()
        fit_x,fit_y,_=prepared['fit',h];val_x,val_y,val_worlds=prepared['internal-holdout',h]
        constant=fit_y.mean(0).expand_as(val_y)
        results[h]=dict(zero=scores(torch.zeros_like(val_y),val_y,val_worlds,weights),
                        train_mean=scores(constant,val_y,val_worlds,weights))
        models[h]={};history[h]={};checkpoints[h]={}
        for variant in ['full','action-ablated']:
            net=copy.deepcopy(initial)
            tx=fit_x.clone();vx=val_x.clone()
            if variant=='action-ablated':tx[:,58:]=0.;vx[:,58:]=0.
            torch.manual_seed(261920+h);torch.cuda.manual_seed_all(261920+h)
            optimizer=torch.optim.Adam(net.parameters(),lr=1e-3);curve=[]
            for epoch in range(100):
                for index in torch.randperm(len(tx),device='cuda').split(128):
                    optimizer.zero_grad();value=(net(tx[index])-fit_y[index]).square().mean()
                    if not torch.isfinite(value):raise ValueError('nonfinite effect loss')
                    value.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
                if epoch in [0,49,99]:
                    with torch.no_grad():curve.append(dict(epoch=epoch+1,fit_mse=float((net(tx)-fit_y).square().mean()),holdout=scores(net(vx),val_y,val_worlds,weights)))
            with torch.no_grad():results[h][variant]=scores(net(vx),val_y,val_worlds,weights)
            models[h][variant]=export(net);history[h][variant]=curve
            checkpoints[h][variant]=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            print(json.dumps(dict(horizon=h,variant=variant,metrics=results[h][variant])),flush=True)
    gate=dict(h16_reward_gain10pct=results[16]['full']['reward_mse']<=.9*min(results[16]['action-ablated']['reward_mse'],results[16]['train_mean']['reward_mse']),
              all_effect_mse_beats_train_mean=all(results[h]['full']['effect_mse']<=results[h]['train_mean']['effect_mse'] for h in results))
    model=dict(version='hjepa_native_effect_predictor_v1',feature_version=meta['feature_version'],action_version=meta['action_version'],
               parent_sha256=sha(a.parent),encoders=[l['encoder'] for l in parent['levels']],effects=meta['effects'],reward_weights=meta['reward_weights'],
               levels=[dict(stride=h,latent_mean=normalization[h][0].tolist(),latent_scale=normalization[h][1].tolist(),heads=models[h]) for h in [1,4,16]])
    save(a.out/'model.json',model)
    torch.save(dict(version='hjepa_native_effect_fit_v1',heads=checkpoints,epochs=100),a.out/'checkpoint.pt')
    save(a.out/'report.json',dict(version='hjepa_native_effect_fit_v1',promotion=False,epochs=100,seed=261920,
         data_meta_sha256=sha(a.data/'meta.json'),data_sha256=meta['data_sha256'],parent_sha256=sha(a.parent),
         model_sha256=sha(a.out/'model.json'),results=results,history=history,gate=dict(criteria=gate,passed=all(gate.values())),
         scope='Frozen hierarchy encoders, causal equipped/own-request context and proposed full command sequences. Fit-only latent normalization and mean baseline. Same initialization, parameter count, minibatch order and fixed epoch budget for full and action-ablated heads. Holdout worlds separated, no checkpoint selection. Overlapping windows not independent; observed future commands are policy-correlated, not physical interventions or proof of action ranking. No runtime integration.'))
    print(json.dumps(dict(gate=gate),indent=2))


if __name__=='__main__':main()
