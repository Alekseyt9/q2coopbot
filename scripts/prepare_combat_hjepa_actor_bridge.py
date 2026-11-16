"""Exact learned-motor to64latent encoder/actor bridge, plus sampling fixtures."""
import argparse
import copy
import json
import math
import pathlib
import torch
from train_combat_hjepa import sha,save,export,network,serialized_forward


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent',type=pathlib.Path,required=True);ap.add_argument('--states',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    parent=json.loads(a.parent.read_text());motor=parent.get('motor',[])
    if parent['kind']!='combat_hjepa_v1' or parent['feature_version']!='hjepa_context_features_v1' or len(motor)!=3 or len(motor[0]['weight'][0])!=871 or len(motor[1]['bias'])!=64 or len(motor[-1]['bias'])!=8:
        raise ValueError('871->128->64->8 motor parent required')
    baseline=copy.deepcopy(parent);baseline['planner']='bc'
    for field in ['primitive_sampling','consistency_margin']:baseline.pop(field,None)
    bridge=copy.deepcopy(baseline);bridge.pop('motor');bridge['actor']=[copy.deepcopy(motor[-1])]
    identity=dict(weight=[[float(i==j) for j in range(64)] for i in range(64)],bias=[0.]*64)
    bridge['levels'][0]['encoder']=copy.deepcopy(motor[:-1])+[identity]
    torch.manual_seed(261980)
    for i,h in enumerate([1,4,16]):
        size=64 if i==0 else 32
        if i>0:bridge['levels'][i]['encoder']=export(network(64 if i==1 else 32,32))
        bridge['levels'][i]['predictor']=export(network(size+8*h,size))
        bridge['levels'][i]['utility']=[dict(weight=[[0.]*size],bias=[0.])]
    bridge['world_action_space']='proposed_primitive_v1'
    rows=[]
    for line in a.states.open():
        row=json.loads(line);features=row['weapon_features']
        if len(features)!=871 or features[:845]!=row['features']:raise ValueError('causal source features')
        rows.append(features)
        if len(rows)==8:break
    if len(rows)!=8:raise ValueError('eight parity states required')
    x=torch.tensor(rows,dtype=torch.float64);actions=torch.tensor([[.2,-.1,.01,0.,1.,1.,0.,0.]]*8,dtype=torch.float64)
    z=x;latents=[];predictions=[]
    with torch.no_grad():
        for level in bridge['levels']:
            z=serialized_forward(z,level['encoder']);latents.append(z.tolist());predictions.append(serialized_forward(torch.cat((z,actions.repeat(1,level['stride'])),-1),level['predictor']).tolist())
        source_raw=serialized_forward(x,motor);actor_raw=serialized_forward(torch.tensor(latents[0],dtype=torch.float64),bridge['actor'])
        error=float((source_raw-actor_raw).abs().max())
        if error>1e-12:raise ValueError('motor bridge changed logits')
    a.out.mkdir(exist_ok=False,parents=True)
    save(a.out/'motor.json',baseline);save(a.out/'actor.json',bridge)
    variants=[]
    for name in ['motor','actor']:
        p=a.out/f'{name}.json';variants.append(dict(name=name,mode='learned',model=str(p.resolve()),sha256=sha(p),contract='stock_weapons_fire_v1'))
    for stream in range(2):
        model=copy.deepcopy(bridge);model['primitive_sampling']=dict(version='hjepa_primitive_distribution_v1',seed=261980+stream,
            log_std=[math.log(.15),math.log(.15),math.log(.003),math.log(.003)],categorical_temperature=.25)
        name=f'sampled-s{stream}';path=a.out/f'{name}.json';save(path,model)
        variants.append(dict(name=name,mode='learned',model=str(path.resolve()),sha256=sha(path),contract='stock_weapons_fire_v1'))
    save(a.out/'models.json',variants)
    save(a.out/'parity.json',dict(features=rows,actions=actions.tolist(),latents=latents,predictions=predictions,actor=source_raw.tolist()))
    save(a.out/'report.json',dict(version='hjepa_actor_bridge_v1',promotion=False,parent_sha256=sha(a.parent),states_sha256=sha(a.states),
         converter_sha256=sha(pathlib.Path(__file__)),independent_raw_max_error=error,
         scope='Original motor hidden layers plus exact identity yield64latent; final motor layer is now actor. Direct motor removed. BC planning only: auxiliary world encoders/predictors newly initialized and utility zero, not used for action selection. Shared encoder/action computation initially preserves source policy. Sampling adds fresh full-primitive exploration; no quality or learning improvement yet. Weapon weights and seed preserved.'))


if __name__=='__main__':main()
