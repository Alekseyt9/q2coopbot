"""Joint world representation/effect learning; exports an offline-only bundle."""
import argparse
import copy
import json
import pathlib
import torch
from torch import nn
from torch.nn import functional as F
from train_combat_hjepa import sha,save,export,network
from train_combat_hjepa_branch_preferences import network as imported_network

STRIDES=[1,4,16]


def window_targets(effects,indexes,lengths):
    mask=torch.arange(16,device=effects.device)[None,:]<lengths[:,None]
    cumulative=(effects[indexes]*mask[...,None]).cumsum(1)
    return torch.stack([cumulative[:,h-1] for h in STRIDES],1)


def effect_predictions(raw):
    return torch.cat((raw[...,:1].sigmoid(),F.softplus(raw[...,1:4]),raw[...,4:].sigmoid()),-1)


class World(nn.Module):
    def __init__(self,parent,width,means):
        super().__init__();self.width=width
        layers=copy.deepcopy(parent['levels'])
        if width==845:layers[0]['encoder'][0]['weight']=[row[:845] for row in layers[0]['encoder'][0]['weight']]
        self.encoders=nn.ModuleList([imported_network(l['encoder']) for l in layers])
        self.predictors=nn.ModuleList([imported_network(l['predictor']) for l in layers])
        self.effects=nn.ModuleList()
        for i,h in enumerate(STRIDES):
            torch.manual_seed(261960+h)
            net=network(32+8*h,6)
            with torch.no_grad():
                value=means[i].cpu().clamp(1e-4,1-1e-4)
                bias=torch.log(value/(1-value));bias[1:4]=torch.log(torch.expm1(value[1:4]))
                net[-1].bias.copy_(bias)
            self.effects.append(net)

    def encode(self,x):
        result=[];z=x[...,:self.width]
        for encoder in self.encoders:z=encoder(z);result.append(z)
        return result


def get_batch(data,clip_ids):
    indexes=data['clip_indexes'][clip_ids]
    x=torch.cat((data['features'][indexes[:,0]][:,None],data['next_features'][indexes]),1)
    return x,data['actions'][indexes],data['targets'][clip_ids]


def objective(model,target_encoders,x,actions,targets):
    zs=model.encode(x);target=x[...,:model.width];target_z=[]
    with torch.no_grad():
        for encoder in target_encoders:target=encoder(target);target_z.append(target)
    total=x.new_zeros(())
    for i,h in enumerate(STRIDES):
        z=zs[i];inputs=torch.cat((z[:,0],actions[:,:h].flatten(1)),-1)
        prediction=model.predictors[i](inputs);raw=model.effects[i](inputs)
        flat=z.flatten(0,1);centered=flat-flat.mean(0)
        covariance=centered.T@centered/max(1,len(flat)-1)
        regularizer=F.relu(1-(flat.var(0,unbiased=False)+1e-4).sqrt()).mean()
        regularizer=regularizer+(covariance.square().sum()-covariance.diag().square().sum())/32
        total=total+F.mse_loss(prediction,target_z[i][:,h])+.2*regularizer
        total=total+F.mse_loss(effect_predictions(raw),targets[:,i])+.1*F.binary_cross_entropy_with_logits(raw[:,4:],targets[:,i,4:])
    return total/3


def evaluate(model,ema,data,clip_ids,mean_effects):
    totals=[dict(rows=0,latent=0.,persistence=0.,effect=0.,reward=0.,mean_reward=0.) for _ in STRIDES]
    with torch.no_grad():
        for ids in clip_ids.split(256):
            x,actions,targets=get_batch(data,ids);zs=model.encode(x);z=x[...,:model.width];tz=[]
            for encoder in ema:z=encoder(z);tz.append(z)
            for i,h in enumerate(STRIDES):
                inputs=torch.cat((zs[i][:,0],actions[:,:h].flatten(1)),-1)
                pred=model.predictors[i](inputs);eff=effect_predictions(model.effects[i](inputs));count=len(ids);row=totals[i]
                row['rows']+=count;row['latent']+=float((pred-tz[i][:,h]).square().mean())*count
                row['persistence']+=float((zs[i][:,0]-tz[i][:,h]).square().mean())*count
                row['effect']+=float((eff-targets[:,i]).square().mean())*count
                row['reward']+=float(((eff-targets[:,i])@data['reward_weights']).square().mean())*count
                row['mean_reward']+=float(((mean_effects[i]-targets[:,i])@data['reward_weights']).square().mean())*count
    return [dict(stride=h,clips=r['rows'],latent_mse=r['latent']/r['rows'],persistence_mse=r['persistence']/r['rows'],
                 normalized_prediction_mse=r['latent']/max(1e-12,r['persistence']),effect_mse=r['effect']/r['rows'],
                 reward_mse=r['reward']/r['rows'],fit_mean_reward_mse=r['mean_reward']/r['rows']) for h,r in zip(STRIDES,totals)]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--parent',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    torch.set_num_threads(2);meta=json.loads((a.data/'meta.json').read_text())
    if meta['version']!='hjepa_joint_world_data_v1' or sha(a.data/'data.pt')!=meta['data_sha256']:raise ValueError('joint world dataset changed')
    for name,digest in meta['sources_sha256'].items():
        if sha(pathlib.Path(name))!=digest:raise ValueError('joint world source changed')
    data=torch.load(a.data/'data.pt',map_location='cpu',weights_only=True)
    if data['version']!='hjepa_joint_world_data_v1' or data['features'].shape[1]!=871:raise ValueError('world data contract')
    for key,value in list(data.items()):
        if isinstance(value,torch.Tensor):data[key]=value.cuda()
    data['targets']=window_targets(data['effects'],data['clip_indexes'],data['clip_lengths'])
    fit=data['clip_fit'].nonzero().flatten();held=(~data['clip_fit']).nonzero().flatten()
    fit_worlds=set(data['seeds'][data['clip_indexes'][fit,0]].tolist());held_worlds=set(data['seeds'][data['clip_indexes'][held,0]].tolist())
    if fit_worlds&held_worlds or len(fit_worlds)!=48 or len(held_worlds)!=16:raise ValueError('world partition leakage')
    mean_effects=data['targets'][fit].mean(0);parent=json.loads(a.parent.read_text())
    if parent['feature_version']!='hjepa_context_features_v1':raise ValueError('zero-expanded871 parent required')
    a.out.mkdir(exist_ok=False,parents=True);(a.out/'trainer-frozen.py').write_bytes(pathlib.Path(__file__).read_bytes())
    reports={}
    for variant,width in [('legacy845',845),('context871',871)]:
        model=World(parent,width,mean_effects).cuda();ema=copy.deepcopy(model.encoders);ema.requires_grad_(False)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);generator=torch.Generator(device='cuda').manual_seed(261960);history=[]
        for step in range(1000):
            ids=fit[torch.randint(len(fit),(128,),generator=generator,device='cuda')]
            optimizer.zero_grad();loss=objective(model,ema,*get_batch(data,ids))
            if not torch.isfinite(loss):raise ValueError('nonfinite joint world loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            with torch.no_grad():
                for target,online in zip(ema.parameters(),model.encoders.parameters()):target.mul_(.99).add_(online,alpha=.01)
            if step in [0,499,999]:
                row=dict(step=step+1,holdout=evaluate(model,ema,data,held,mean_effects));history.append(row)
                print(json.dumps(dict(variant=variant,**row)),flush=True)
        target=a.out/variant;target.mkdir()
        save(target/'world.json',dict(kind='hjepa_joint_world_offline_v1',feature_version='combat_features_v6' if width==845 else 'hjepa_context_features_v1',
             action_space='proposed_primitive_v1',effects_activation='sigmoid_progress_terminals_softplus_costs_v1',reward_weights=data['reward_weights'].tolist(),
             levels=[dict(stride=h,encoder=export(model.encoders[i]),predictor=export(model.predictors[i]),effects=export(model.effects[i])) for i,h in enumerate(STRIDES)],
             parent_sha256=sha(a.parent),runtime_eligible=False))
        torch.save(dict(version='hjepa_joint_world_fit_v1',model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),steps=1000,
                        generator=generator.get_state(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),target/'checkpoint.pt')
        reports[variant]=dict(history=history,final=history[-1]['holdout'],model_sha256=sha(target/'world.json'),
                              context_columns_l2=float(model.encoders[0][0].weight[:,845:].detach().square().sum().sqrt()) if width==871 else 0.)
    new=reports['context871']['final'];old=reports['legacy845']['final']
    criteria=dict(h16_reward_beats_mean10pct=new[2]['reward_mse']<=.9*new[2]['fit_mean_reward_mse'],
                  h16_reward_beats_legacy10pct=new[2]['reward_mse']<=.9*old[2]['reward_mse'],
                  normalized_prediction_beats_persistence=all(r['normalized_prediction_mse']<1 for r in new))
    save(a.out/'report.json',dict(version='hjepa_joint_world_fit_v1',promotion=False,steps=1000,seed=261960,reports=reports,
         gate=dict(criteria=criteria,passed=all(criteria.values())),data_meta_sha256=sha(a.data/'meta.json'),parent_sha256=sha(a.parent),
         trainer_sha256=sha(pathlib.Path(__file__)),fit_worlds=48,holdout_worlds=16,
         scope='Both845and871encoders/predictors/effect heads train, same seed/sample order and common initial weights. EMA targets, fixed1000steps, no actor/motor/weapon or accumulated-progress utility fitting. Native effect labels and terminal lengths never policy inputs. Offline-only bundles intentionally cannot load as runtime controllers: old utility semantics are not assumed after representation changes. Internal holdout is diagnostic, not fresh external quality evidence.'))
    print(json.dumps(dict(gate=criteria),indent=2))


if __name__=='__main__':main()
