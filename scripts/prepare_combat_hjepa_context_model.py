"""Extend frozen845 H-JEPA world/motor inputs by26zero columns, preserving heads."""
import argparse
import copy
import json
import pathlib
import torch
from train_combat_hjepa import sha,save,serialized_forward


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent',type=pathlib.Path,required=True)
    ap.add_argument('--states',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args();parent=json.loads(a.parent.read_text())
    if parent['kind']!='combat_hjepa_v1' or parent['feature_version']!='combat_features_v6':raise ValueError('frozen845parent required')
    model=copy.deepcopy(parent);model['feature_version']='hjepa_context_features_v1'
    first_layers=[model['levels'][0]['encoder'][0]]
    if model.get('motor'):first_layers.append(model['motor'][0])
    for layer in first_layers:
        if any(len(row)!=845 for row in layer['weight']):raise ValueError('parent first-layer width')
        layer['weight']=[row+[0.]*26 for row in layer['weight']]
    check=copy.deepcopy(model);check['feature_version']=parent['feature_version']
    first=[check['levels'][0]['encoder'][0]]
    if check.get('motor'):first.append(check['motor'][0])
    for layer in first:layer['weight']=[row[:845] for row in layer['weight']]
    if check!=parent:raise ValueError('non-extension component changed')
    rows=[]
    for line in a.states.open():
        row=json.loads(line);features=row['weapon_features']
        if len(features)!=871 or features[:845]!=row['features']:raise ValueError('causal context prefix mismatch')
        rows.append(features)
        if len(rows)==8:break
    if len(rows)!=8:raise ValueError('eight causal parity states required')
    torch.set_num_threads(2)
    full=torch.tensor(rows,dtype=torch.float64);x=full[:,:845];action=torch.tensor([[.2,-.1,.01,0,1,1,0,0]]*8,dtype=torch.float64)
    z=x;latents=[];predictions=[]
    with torch.no_grad():
        for level in parent['levels']:
            z=serialized_forward(z,level['encoder']);latents.append(z.tolist())
            predictions.append(serialized_forward(torch.cat((z,action.repeat(1,level['stride'])),-1),level['predictor']).tolist())
        actor=serialized_forward(x,parent['motor']) if parent.get('motor') else serialized_forward(torch.tensor(latents[0],dtype=torch.float64),parent['actor'])
    a.out.mkdir(exist_ok=False,parents=True)
    (a.out/'legacy.json').write_bytes(a.parent.read_bytes());save(a.out/'context.json',model)
    save(a.out/'parity.json',dict(features=rows,actions=action.tolist(),latents=latents,predictions=predictions,actor=actor.tolist()))
    save(a.out/'report.json',dict(version='hjepa_context_zero_extension_v1',promotion=False,parent_sha256=sha(a.parent),model_sha256=sha(a.out/'context.json'),
         parity_sha256=sha(a.out/'parity.json'),states_sha256=sha(a.states),converter_sha256=sha(pathlib.Path(__file__)),
         scope='Only world/motor first-layer26zero columns and explicit feature header change. Old845weights, independent weapon head format, all other arrays/metadata preserved. Parity target computed from original845networks and causal source871features; no fitting.'))


if __name__=='__main__':main()
