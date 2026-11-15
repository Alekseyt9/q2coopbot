"""Distil U1 action targets into a separate H-JEPA motor.

Uses every eligible single frame; no future world-model window is required.
Inputs are the same client-only v6 features as the world model.
Accepts unmodified executed actions or explicitly marked offline ensemble
recommendations. Recommendations never replace the executed world-model action.
"""
import argparse
import hashlib
import json
import math
import pathlib
import time
import torch
from torch import nn
from torch.nn import functional as F
from train_combat_hjepa import export, save, sha


def load_rows(path):
    meta = json.loads(pathlib.Path(str(path)+'.meta.json').read_text())
    hypothetical = meta.get('label_kind') == 'hypothetical_ppo_ensemble_v1'
    for name,digest in meta.get('native_receipts_sha256',{}).items():
        if sha(pathlib.Path(name))!=digest:raise ValueError('changed native receipt')
    for name,digest in meta.get('teacher_variants_sha256',{}).items():
        if sha(pathlib.Path(name))!=digest:raise ValueError('changed teacher weights')
    lookup = {}
    for name, digest in meta['sources_sha256'].items():
        source = pathlib.Path(name)
        if sha(source) != digest:
            raise ValueError('changed native source')
        for line in source.read_text().splitlines():
            s = json.loads(line)
            ident = s['observation']['identity']
            key = (s['episode'], ident['frame'], ident['spawncount'], ident['connection'], ident['actor'], ident['life'], ident['map'])
            if key in lookup:
                raise ValueError('duplicate teacher identity')
            lookup[key] = s
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        episode, identity = r['episode'].split(':',1)
        ident = json.loads(identity)
        key = (episode,r['frame'],ident['spawncount'],ident['connection'],ident['actor'],ident['life'],ident['map'])
        s = lookup[key]
        if s['owner'] != 'provider' or not s['provider'].startswith('ppo:') or (s.get('interventions') and not hypothetical):
            continue
        execution = s.get('server_execution')
        if not execution or not execution['matched'] or not execution['window_exclusive']:
            continue
        if len(r['features']) != 845 or len(r['action']) != 8:
            raise ValueError('teacher feature/action dimensions')
        if hypothetical:
            if r.get('label_kind') != meta['label_kind'] or len(r.get('teacher_action',[]))!=8:
                raise ValueError('invalid hypothetical teacher label')
            r=dict(r,action=r['teacher_action'])
        rows.append(r)
    if not rows:
        raise ValueError('no unmodified verified teacher actions')
    return rows


def tensors(rows, device='cuda'):
    x = torch.tensor([r['features'] for r in rows],device=device,dtype=torch.float32)
    y = torch.tensor([r['action'] for r in rows],device=device,dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('nonfinite teacher rows')
    if (y[:,:4].abs()>1).any() or (y[:,4:]<0).any() or (y[:,4:]>1).any() or not torch.allclose(y[:,5:].sum(-1),torch.ones_like(y[:,4])):
        raise ValueError('invalid teacher primitive/distribution')
    return x,y


def loss_and_metrics(net, data):
    x,y=data
    raw=net(x)
    pred=raw[:,:4].tanh()
    error=pred-y[:,:4]
    yaw=(error[:,2]+1).remainder(2)-1
    pos=y[:,4].sum()
    pos_weight=((len(y)-pos)/pos.clamp_min(1)).clamp(.1,10)
    if ((y[:,4]>0)&(y[:,4]<1)).any():
        pos_weight=torch.ones_like(pos_weight)  # preserve ensemble probabilities
    fire=F.binary_cross_entropy_with_logits(raw[:,4],y[:,4],pos_weight=pos_weight)
    loss=(error[:,:2].square().mean()+200*(yaw.square().mean()+error[:,3].square().mean())
          +fire-(F.log_softmax(raw[:,5:],dim=-1)*y[:,5:]).sum(-1).mean())
    return loss,dict(yaw_rmse_deg=float(yaw.square().mean().sqrt().detach()*180),
                     pitch_rmse_deg=float(error[:,3].square().mean().sqrt().detach()*180),
                     movement_rmse=float(error[:,:2].square().mean().sqrt().detach()),
                     fire_accuracy=float(((raw[:,4]>=0)==(y[:,4]>=.5)).float().mean().detach()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',type=pathlib.Path,required=True)
    ap.add_argument('--validation',type=pathlib.Path,required=True)
    ap.add_argument('--world',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--epochs',type=int,default=600)
    ap.add_argument('--seed',type=int,default=251116)
    a=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if not 1<=a.epochs<=2000:raise ValueError('epoch budget')
    train_rows=load_rows(a.train);val_rows=load_rows(a.validation)
    seeds=lambda rows:{r['episode'].split(':',1)[0] for r in rows}
    if seeds(train_rows)&seeds(val_rows):raise ValueError('train/validation seed overlap')
    a.out.mkdir(parents=True,exist_ok=False)
    trainer_snapshot=pathlib.Path(__file__).read_bytes()
    (a.out/'trainer-frozen.py').write_bytes(trainer_snapshot)
    torch.set_num_threads(2);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    train,val=tensors(train_rows),tensors(val_rows)
    net=nn.Sequential(nn.Linear(845,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,8)).cuda()
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    history=[];start=time.perf_counter()
    with torch.no_grad():before=dict(train=loss_and_metrics(net,train)[1],validation=loss_and_metrics(net,val)[1])
    for epoch in range(a.epochs):
        for batch in torch.randperm(len(train_rows),device='cuda').split(128):
            opt.zero_grad();loss,_=loss_and_metrics(net,tuple(t[batch] for t in train))
            if not torch.isfinite(loss):raise ValueError('nonfinite motor loss')
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        if epoch%100==0 or epoch+1==a.epochs:
            with torch.no_grad():entry=dict(epoch=epoch+1,train=loss_and_metrics(net,train)[1],validation=loss_and_metrics(net,val)[1])
            history.append(entry);print(json.dumps(entry),flush=True)
    cp=dict(version='hjepa_motor_checkpoint_v1',motor=net.state_dict(),optimizer=opt.state_dict(),epochs=a.epochs,seed=a.seed,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    torch.save(cp,a.out/'checkpoint.pt')
    net=net.cpu().double()
    world=json.loads(a.world.read_text());world['motor']=export(net)
    for planner in ('bc','hierarchy','sequence'):
        save(a.out/(planner+'.json'),dict(world,planner=planner))
    with torch.no_grad():
        x=torch.tensor([r['features'] for r in val_rows[:8]],dtype=torch.float64)
        outputs=net(x).tolist()
    save(a.out/'motor-parity.json',dict(features=x.tolist(),outputs=outputs))
    save(a.out/'report.json',dict(version='hjepa_motor_training_v1',train_frames=len(train_rows),validation_frames=len(val_rows),
         before=before,history=history,epochs=a.epochs,seed=a.seed,seconds=time.perf_counter()-start,
         train_sha256=sha(a.train),validation_sha256=sha(a.validation),world_sha256=sha(a.world),
         trainer_sha256=hashlib.sha256(trainer_snapshot).hexdigest(),weights_sha256={p:sha(a.out/(p+'.json')) for p in ('bc','hierarchy','sequence')},
         label_kind=json.loads(pathlib.Path(str(a.train)+'.meta.json').read_text()).get('label_kind','native_executed_v1'),
         promotion=False,scope='Single-frame teacher BC direct v6 motor; see label_kind for executed vs hypothetical targets. Inherited world networks unchanged. Fixed final epoch, no live quality claim.'))


if __name__=='__main__':main()
