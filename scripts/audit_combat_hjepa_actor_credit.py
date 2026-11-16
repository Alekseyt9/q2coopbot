"""Read-only TRAIN primitive score-gradient decomposition; never changes weights."""
import argparse
import json
import math
import pathlib
import torch
from torch.nn import functional as F
from train_combat_hjepa import save
from train_combat_hjepa_actor_ppo import Actor,load_rollouts
from train_combat_hjepa_actor_natural import flatten


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=pathlib.Path,action='append',required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    torch.set_num_threads(2)
    rows,parent,binding,admission = load_rollouts(a.data)
    model = Actor(parent).cuda();parameters = list(model.parameters())
    tensor = lambda values,dtype=torch.float64:torch.tensor(values,dtype=dtype,device='cuda')
    x = tensor([r['features'] for r in rows]);latent = tensor([r['sample']['latent'] for r in rows])
    attack = tensor([r['sample']['attack'] for r in rows],torch.bool)
    pose = tensor([r['sample']['vertical'] for r in rows],torch.long)
    advantage = tensor([r['advantage'] for r in rows])
    masks,cache = [],{}
    for row in rows:
        if row['source'] not in cache:
            cache[row['source']] = {s['index']:s for s in map(json.loads,pathlib.Path(row['source']).open())}
        o = cache[row['source']][row['step_index']]['observation'];weapon = o['weapon'].lower()
        masks.append(('v_handgr' in weapon or weapon=='grenades') and o['gun_frame']==11)
    hold = tensor(masks,torch.bool)
    raw = model(x);temperature = model.temperature
    gaussian = (-.5*((latent-raw[:,:4])/model.log_std.exp()).square()-model.log_std-.5*math.log(2*math.pi)).sum(-1)
    fire = F.logsigmoid(torch.where(attack,raw[:,4]/temperature,-raw[:,4]/temperature))
    vertical = F.log_softmax(raw[:,5:]/temperature,-1).gather(1,pose[:,None])[:,0]

    def gradient(value):
        values = torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
        return flatten([torch.zeros_like(p) if v is None else v for p,v in zip(parameters,values)]).detach()

    gradients = {name:gradient((advantage*value).mean()) for name,value in [('gaussian',gaussian),('fire',fire),('pose',vertical)]}
    gradients['fire_hold11'] = gradient((advantage*fire*hold).mean())
    gradients['fire_other'] = gradient((advantage*fire*~hold).mean())
    full = gradient((advantage*(gaussian+fire+vertical)).mean())
    maximum = float((full-sum(gradients[k] for k in ['gaussian','fire','pose'])).abs().max())
    if maximum > 1e-9:raise ValueError('gradient factor decomposition mismatch')
    cosine = lambda u,v:float(u.dot(v)/(u.norm()*v.norm()).clamp_min(1e-30))
    groups = {}
    for name,mask in [('hold11',hold),('other',~hold)]:
        groups[name] = dict(rows=int(mask.sum()),releases=int((~attack&mask).sum()),
            mean_advantage=float(advantage[mask].mean()) if mask.any() else None,
            mean_release_advantage=float(advantage[mask&~attack].mean()) if (mask&~attack).any() else None,
            mean_attack_advantage=float(advantage[mask&attack].mean()) if (mask&attack).any() else None)
    report = dict(version='hjepa_actor_credit_gradient_audit_v1',admission=admission,binding=binding,groups=groups,
        gradient_norms={k:float(v.norm()) for k,v in gradients.items()},
        cosine_fire_hold_vs_other=cosine(gradients['fire_hold11'],gradients['fire_other']),
        cosine_fire_vs_other_factors=cosine(gradients['fire'],gradients['gaussian']+gradients['pose']),
        factor_sum_max_error=maximum,
        scope='Read-only initial behavior score-gradient of canonical native first-life RTG objective on admitted TRAIN samples. Global N denominator retained for phase contributions, no reweighting. No optimization, changed weights, alternative reward, state gate in policy, or combat quality claim. Correlated phases and other-stream baselines prevent interpreting mean conditional advantages as causal treatment effects.')
    save(a.out,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['admission','binding']},indent=2))


if __name__ == '__main__':main()
