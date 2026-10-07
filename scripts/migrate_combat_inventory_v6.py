"""Prefix-preserving v5 -> v6 observation migration; no fitting/optimizer."""
import argparse,copy,hashlib,json,pathlib
from combat_weapon_features import VERSION,PREFIX,WIDTH

def migrate(model):
    if model.get('kind')!='combat_ppo_v1' or model.get('feature_version')!='combat_features_v5':raise ValueError('Requires v5 PPO')
    if model.get('entity_attention'):raise ValueError('Entity attention inventory migration needs separate token contract')
    result=copy.deepcopy(model)
    for role,outputs in [('actor',8),('value',1)]:
        layers=result[role]
        if len(layers)!=3 or len(layers[-1]['bias'])!=outputs:raise ValueError('Unexpected architecture')
        for row in layers[0]['weight']:
            if len(row)!=PREFIX:raise ValueError('Unexpected v5 width')
            row.extend([0.]*(WIDTH-PREFIX))
    result['feature_version']=VERSION
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    source=a.model.read_bytes();model=migrate(json.loads(source.decode('utf-8-sig')))
    a.out.mkdir(exist_ok=False,parents=True)
    encoded=json.dumps(model,indent=2,allow_nan=False).encode('utf-8');(a.out/'weights.json').write_bytes(encoded)
    receipt={'version':'combat_inventory_migration_v1','source_sha256':hashlib.sha256(source).hexdigest(),'weights_sha256':hashlib.sha256(encoded).hexdigest(),'feature_version':VERSION,'width':WIDTH,'actor_outputs':8,'new_columns':WIDTH-PREFIX,'scope':'Zero-input-column migration, old movement/aim/fire/vertical preserved; weapon categorical head and optimizer migration not implemented; no training/live promotion'}
    (a.out/'migration.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
