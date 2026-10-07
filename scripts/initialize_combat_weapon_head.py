"""Expand V6 actor with a masked weapon head; no optimizer/learning step."""
import argparse,copy,hashlib,json,pathlib

def initialize(model):
    if model.get('kind')!='combat_ppo_v1' or model.get('feature_version')!='combat_features_v6' or model.get('weapon_head') or model.get('entity_attention'):
        raise ValueError('Requires V6 PPO without a weapon head or entity attention')
    result=copy.deepcopy(model)
    layer=result['actor'][-1]
    if len(layer['bias'])!=8:raise ValueError('Unexpected actor width')
    def extend(layer):
        width=len(layer['weight'][0])
        layer['weight'].extend([[0.]*width for _ in range(12)])
        layer['bias'].extend([0.]*12)
    extend(layer)
    layer['bias'][8]=5.  # initially prefer keep, retain stochastic exploration
    if result.get('attention'):extend(result['attention']['actor']['residual'])
    if result.get('memory'):extend(result['memory']['actor']['output'])
    result['weapon_head']='combat_masked_weapon_v1'
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,type=pathlib.Path);ap.add_argument('--out',required=True,type=pathlib.Path);a=ap.parse_args()
    source=a.model.read_bytes();result=initialize(json.loads(source.decode('utf-8-sig')))
    a.out.mkdir(exist_ok=False,parents=True)
    encoded=json.dumps(result,indent=2,allow_nan=False).encode('utf-8');(a.out/'weights.json').write_bytes(encoded)
    receipt=dict(version='combat_weapon_head_initialization_v1',source_sha256=hashlib.sha256(source).hexdigest(),weights_sha256=hashlib.sha256(encoded).hexdigest(),actor_outputs=20,keep_bias=5.,scope='Untrained weapon head. Existing eight logits preserved. Adam checkpoint migration not included; no optimizer steps or live promotion.')
    (a.out/'initialization.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
