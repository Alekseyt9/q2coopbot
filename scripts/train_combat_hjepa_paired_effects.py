"""Fit frozen-encoder effect heads to controlled within-world action contrasts."""
import argparse
import copy
import json
import pathlib
import torch
from torch import nn
from train_combat_hjepa import sha, save, export, network
from train_combat_hjepa_branch_preferences import network as import_network
from prepare_combat_hjepa_effects import WEIGHTS


def pair_indices(rows):
    indexed={}
    for i,row in enumerate(rows):
        key=(row['seed'],row['variant'])
        if key in indexed:raise ValueError('duplicate controlled arm')
        indexed[key]=i
    left=[];right=[]
    for i,row in enumerate(rows):
        if row['variant']=='idle':continue
        reference='idle' if row['variant']=='fire' else 'fire'
        key=(row['seed'],reference)
        if key not in indexed:continue
        j=indexed[key]
        if rows[i]['features']!=rows[j]['features'] or rows[i]['horizon']!=rows[j]['horizon']:
            raise ValueError('paired initial state/horizon differs')
        left.append(i);right.append(j)
    if not left:raise ValueError('no controlled pairs')
    return left,right


def metrics(prediction,target,pairs,weights):
    left,right=pairs;difference=prediction[left]-prediction[right]
    actual=target[left]-target[right];pr=difference@weights;tr=actual@weights;nonzero=tr.abs()>1e-9
    return dict(rows=len(target),pairs=len(left),absolute_effect_mse=float((prediction-target).square().mean()),
                contrast_effect_mse=float((difference-actual).square().mean()),
                contrast_reward_mse=float((pr-tr).square().mean()),zero_contrast_reward_mse=float(tr.square().mean()),
                nonzero_reward_contrasts=int(nonzero.sum()),
                nonzero_reward_sign_accuracy=float(((pr[nonzero]>0)==(tr[nonzero]>0)).float().mean()) if nonzero.any() else None)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--contrasts',type=pathlib.Path,required=True)
    ap.add_argument('--partition',type=pathlib.Path,required=True)
    ap.add_argument('--parent',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args()
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    report=json.loads((a.contrasts/'report.json').read_text())
    if sha(a.contrasts/'rows.jsonl')!=report['rows_sha256']:raise ValueError('controlled rows changed')
    for name,digest in report['sources_sha256'].items():
        if sha(pathlib.Path(name))!=digest:raise ValueError('controlled source changed')
    partition=json.loads(a.partition.read_text());fit=set(partition['fit_engine_seeds']);held=set(partition['internal_holdout_engine_seeds'])
    if fit&held or len(fit)!=48 or len(held)!=16:raise ValueError('world partition changed')
    rows=[json.loads(l) for l in (a.contrasts/'rows.jsonl').open()]
    if {r['seed'] for r in rows}!=fit|held:raise ValueError('missing source worlds')
    parent=json.loads(a.parent.read_text());torch.set_num_threads(2)
    encoders=nn.ModuleList([import_network(l['encoder']) for l in parent['levels']]).cuda();encoders.requires_grad_(False)
    weights=torch.tensor(WEIGHTS,device='cuda');prepared={};normalization={};pair_worlds={}
    with torch.no_grad():
        for i,h in enumerate([1,4,16]):
            for role,seeds in [('fit',fit),('internal-holdout',held)]:
                subset=[r for r in rows if r['seed'] in seeds and r['horizon']==h]
                left,right=pair_indices(subset);features=torch.tensor([r['features'] for r in subset],device='cuda')
                actions=torch.tensor([r['planned_actions'] for r in subset],device='cuda').flatten(1)
                target=torch.tensor([r['effects'] for r in subset],device='cuda')
                if features.shape!=(len(subset),871) or actions.shape!=(len(subset),20*h) or target.shape!=(len(subset),6):raise ValueError('effect shapes')
                z=features[:,:845]
                for encoder in encoders[:i+1]:z=encoder(z)
                if role=='fit':normalization[h]=(z.mean(0),z.std(0,unbiased=False).clamp_min(.05))
                mean,scale=normalization[h];x=torch.cat(((z-mean)/scale,features[:,845:],actions),-1)
                if not torch.isfinite(x).all() or not torch.isfinite(target).all():raise ValueError('nonfinite controlled data')
                pairs=(torch.tensor(left,device='cuda'),torch.tensor(right,device='cuda'))
                prepared[role,h]=(x,target,pairs)
                pair_worlds[role,h]=len({subset[j]['seed'] for j in left})
    a.out.mkdir(exist_ok=False,parents=True);(a.out/'trainer-frozen.py').write_bytes(pathlib.Path(__file__).read_bytes())
    results={};models={};history={};checkpoints={}
    for h in [1,4,16]:
        torch.manual_seed(261940+h);torch.cuda.manual_seed_all(261940+h)
        initial=network(58+20*h,6,hidden=64).cuda()
        tx,ty,tp=prepared['fit',h];vx,vy,vp=prepared['internal-holdout',h]
        results[h]=dict(train_mean_absolute_effect_mse=float((ty.mean(0)-vy).square().mean()),
                        fit_pair_worlds=pair_worlds['fit',h],holdout_pair_worlds=pair_worlds['internal-holdout',h])
        models[h]={};history[h]={};checkpoints[h]={}
        for variant in ['paired','absolute']:
            net=copy.deepcopy(initial);optimizer=torch.optim.Adam(net.parameters(),lr=1e-3);curve=[]
            for step in range(300):
                optimizer.zero_grad();prediction=net(tx);absolute=(prediction-ty).square().mean()
                contrast=((prediction[tp[0]]-prediction[tp[1]])-(ty[tp[0]]-ty[tp[1]])).square().mean()
                loss=contrast+.2*absolute if variant=='paired' else absolute
                if not torch.isfinite(loss):raise ValueError('nonfinite paired objective')
                loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
                if step in [0,149,299]:
                    with torch.no_grad():curve.append(dict(step=step+1,fit=metrics(net(tx),ty,tp,weights),holdout=metrics(net(vx),vy,vp,weights)))
            with torch.no_grad():results[h][variant]=metrics(net(vx),vy,vp,weights)
            models[h][variant]=export(net);history[h][variant]=curve
            checkpoints[h][variant]=dict(model=net.state_dict(),optimizer=optimizer.state_dict())
            print(json.dumps(dict(horizon=h,variant=variant,metrics=results[h][variant])),flush=True)
    final=results[16];paired=final['paired'];absolute=final['absolute']
    criteria=dict(h16_contrast_reward_gain10pct=paired['contrast_reward_mse']<=.9*min(absolute['contrast_reward_mse'],paired['zero_contrast_reward_mse']),
                  h16_absolute_effect_beats_train_mean=paired['absolute_effect_mse']<=final['train_mean_absolute_effect_mse'],
                  h16_nonzero_sign_accuracy60pct=paired['nonzero_reward_sign_accuracy'] is not None and paired['nonzero_reward_sign_accuracy']>=.6)
    save(a.out/'model.json',dict(version='hjepa_paired_effect_predictor_v1',parent_sha256=sha(a.parent),encoders=[l['encoder'] for l in parent['levels']],
         reward_weights=WEIGHTS,levels=[dict(stride=h,latent_mean=normalization[h][0].tolist(),latent_scale=normalization[h][1].tolist(),heads=models[h]) for h in [1,4,16]]))
    torch.save(dict(version='hjepa_paired_effect_fit_v1',steps=300,heads=checkpoints),a.out/'checkpoint.pt')
    save(a.out/'report.json',dict(version='hjepa_paired_effect_fit_v1',promotion=False,seed=261940,steps=300,
         trainer_sha256=sha(pathlib.Path(__file__)),helper_sha256={name:sha(pathlib.Path(__file__).with_name(name)) for name in ['train_combat_hjepa.py','train_combat_hjepa_branch_preferences.py','prepare_combat_hjepa_effects.py']},
         contrast_report_sha256=sha(a.contrasts/'report.json'),partition_sha256=sha(a.partition),parent_sha256=sha(a.parent),model_sha256=sha(a.out/'model.json'),
         results=results,history=history,gate=dict(criteria=criteria,passed=all(criteria.values())),
         scope='Controlled constant-command pairs, source-world-separated48/16 TRAIN internal partition. Frozen encoders, fit-only normalization, identical initialization and300fullbatch steps. Paired objective vs absolute objective, no sweep or checkpoint selection. Internal holdout previously used for observational diagnostics, not external validation. No Go runtime integration or controller promotion.'))
    print(json.dumps(dict(gate=criteria),indent=2))


if __name__=='__main__':main()
