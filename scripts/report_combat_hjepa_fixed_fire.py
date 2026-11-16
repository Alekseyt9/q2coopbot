"""Fixed TRAIN observation fire probes at one common temperature, read-only."""
import argparse
import json
import pathlib
import torch
from train_combat_hjepa import save,sha,serialized_forward


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=pathlib.Path,action='append',required=True)
    ap.add_argument('--model',action='append',required=True,help='label=path')
    ap.add_argument('--temperature',type=float,default=1.)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a = ap.parse_args()
    if not .05 <= a.temperature <= 5:raise ValueError('finite supported temperature required')
    torch.set_num_threads(2)
    features,labels,sources,cache = [],[],{},{}
    for path in a.data:
        mp = pathlib.Path(str(path)+'.meta.json');meta = json.loads(mp.read_text())
        if meta['version'] != 'hjepa_onpolicy_actor_rollout_v1' or meta['split'] != 'train' or meta['data_sha256'] != sha(path):
            raise ValueError('unchanged typed TRAIN source required')
        sources[str(path.resolve())],sources[str(mp.resolve())] = sha(path),sha(mp)
        with path.open(encoding='utf-8') as stream:
            for row in map(json.loads,stream):
                source = row['source']
                if source not in cache:
                    if sha(pathlib.Path(source)) != row['source_sha256']:
                        raise ValueError('probe observations changed')
                    sources[source] = row['source_sha256']
                    with pathlib.Path(source).open(encoding='utf-8') as native:
                        cache[source] = {s['index']:s for s in map(json.loads,native)}
                o = cache[source][row['step_index']]['observation']
                if o['identity'] != row['identity'] or len(row['features']) != 871:
                    raise ValueError('probe identity/feature mismatch')
                weapon = o['weapon'].lower()
                labels.append(('v_handgr' in weapon or weapon=='grenades') and o['gun_frame']==11)
                features.append(row['features'])
    x = torch.tensor(features,dtype=torch.float64);hold = torch.tensor(labels,dtype=torch.bool)
    results = {}
    with torch.no_grad():
        for named in a.model:
            name,path = named.split('=',1);path = pathlib.Path(path)
            if name in results:raise ValueError('duplicate model label')
            model = json.loads(path.read_text())
            if model.get('motor') or model['feature_version'] != 'hjepa_context_features_v1':
                raise ValueError('shared871 actor required')
            raw = serialized_forward(serialized_forward(x,model['levels'][0]['encoder']),model['actor'])
            release = (-raw[:,4]/a.temperature).sigmoid();groups = {}
            for label,mask in [('hold11',hold),('other',~hold)]:
                groups[label] = dict(rows=int(mask.sum()),mean_release_probability=float(release[mask].mean()) if mask.any() else None,
                    mean_logit=float(raw[mask,4].mean()) if mask.any() else None,
                    primitive_mean_releases=int((raw[mask,4]<0).sum()))
            results[name] = dict(path=str(path.resolve()),model_sha256=sha(path),groups=groups)
    report = dict(version='hjepa_fixed_train_fire_probe_v2',temperature=a.temperature,models=results,source_sha256=sources,
        scope='Fixed TRAIN current observations only, including explicitly exported tail rows. Shared temperature separates weight changes from sampling-temperature changes. Group labels use observed weapon/gunframe, never future effects or policy gates. No fitting, native rollout, validation inputs or combat-quality claim; primitive-mean releases use rawlogit<0.')
    save(a.out,report);print(json.dumps(dict(temperature=a.temperature,models=results),indent=2))


if __name__=='__main__':main()
