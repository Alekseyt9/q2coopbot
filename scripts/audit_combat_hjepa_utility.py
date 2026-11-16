"""Audit frozen utility values and forecast changes against native progress labels."""
import argparse
import collections
import json
import pathlib
import torch
from train_combat_hjepa import sha, save, serialized_forward
from audit_combat_hjepa_world_actions import primitive


def metrics(prediction, target):
    nonzero = target.abs() > 1e-9
    centered_p = prediction - prediction.mean()
    centered_t = target - target.mean()
    denominator = centered_p.square().sum().sqrt() * centered_t.square().sum().sqrt()
    return dict(rows=len(target), nonzero_targets=int(nonzero.sum()),
                mse=float((prediction-target).square().mean()),
                zero_prediction_mse=float(target.square().mean()),
                predicted_mean=float(prediction.mean()), target_mean=float(target.mean()),
                predicted_std=float(prediction.std(unbiased=False)),
                target_std=float(target.std(unbiased=False)),
                pearson=float((centered_p*centered_t).sum()/denominator) if denominator > 0 else None,
                nonzero_sign_accuracy=float(((prediction[nonzero]>0)==(target[nonzero]>0)).double().mean()) if nonzero.any() else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=pathlib.Path, action='append', required=True)
    ap.add_argument('--model', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    groups = collections.defaultdict(list)
    inputs = {}; raw = {}; seen = set()
    for path in args.data:
        meta_path = pathlib.Path(str(path)+'.meta.json')
        meta = json.loads(meta_path.read_text())
        if meta.get('objective_version') != 'native_progress_v3' or meta['data_sha256'] != sha(path):
            raise ValueError('native terminal labels required')
        inputs[str(path)] = sha(path); inputs[str(meta_path)] = sha(meta_path)
        for source, digest in meta['sources_sha256'].items():
            p = pathlib.Path(source)
            if sha(p) != digest:
                raise ValueError('source changed')
            inputs[source] = digest
            if p.name == 'steps.jsonl' and digest not in raw:
                indexed = {}
                for line in p.open():
                    step = json.loads(line); frame = step['observation']['identity']['frame']
                    if frame in indexed:
                        raise ValueError('duplicate source frame')
                    indexed[frame] = step
                raw[digest] = indexed
        rows = [json.loads(line) for line in path.open()]
        if len(rows) != meta['rows']:
            raise ValueError('row count changed')
        for row in rows:
            step = raw[row['source_sha256']][row['frame']]
            if step['owner'] != 'provider':
                continue
            if not step['server_execution']['matched'] or not step['server_execution']['window_exclusive']:
                raise ValueError('unverified provider action')
            key = (row['source_sha256'], row['frame'])
            if key in seen:
                raise ValueError('duplicate transition')
            seen.add(key)
            groups[(row['source_sha256'], row['episode'])].append(row)
    clips = []
    for rows in groups.values():
        rows.sort(key=lambda r: r['frame'])
        for i in range(len(rows)-15):
            clip = rows[i:i+16]
            if any(b['frame'] != a['frame']+1 or a['next_features'] != b['features'] for a,b in zip(clip,clip[1:])):
                continue
            clips.append(clip)
    if not clips:
        raise ValueError('no contiguous clips')
    x = torch.tensor([[r['features'] for r in c]+[c[-1]['next_features']] for c in clips],dtype=torch.float64)
    actions = torch.tensor([[primitive(raw[r['source_sha256']][r['frame']]['action']) for r in c] for c in clips],dtype=torch.float64)
    applied = torch.tensor([[r['action'] for r in c] for c in clips],dtype=torch.float64)
    u = torch.tensor([[r['utility'] for r in c]+[c[-1]['next_utility']] for c in clips],dtype=torch.float64)
    model = json.loads(args.model.read_text()); z = x; results = []
    with torch.no_grad():
        for level in model['levels']:
            z = serialized_forward(z, level['encoder']); stride = level['stride']
            forecast = serialized_forward(torch.cat((z[:,0],actions[:,:stride].flatten(1)),-1),level['predictor'])
            oracle = serialized_forward(torch.cat((z[:,0],applied[:,:stride].flatten(1)),-1),level['predictor'])
            actual_value = serialized_forward(z,level['utility']).squeeze(-1)
            predicted_value = serialized_forward(forecast,level['utility']).squeeze(-1)
            oracle_value = serialized_forward(oracle,level['utility']).squeeze(-1)
            delta = u[:,stride]-u[:,0]
            result = dict(stride=stride,
                          observed_value=metrics(actual_value[:,0],u[:,0]),
                          observed_value_change=metrics(actual_value[:,stride]-actual_value[:,0],delta),
                          raw_forecast_value_change=metrics(predicted_value-actual_value[:,0],delta),
                          applied_oracle_value_change=metrics(oracle_value-actual_value[:,0],delta))
            results.append(result)
    save(args.out,dict(version='hjepa_utility_diagnostic_v1', model_sha256=sha(args.model),
                      clips=len(clips),provider_transitions=len(seen),sources_sha256=inputs,levels=results,
                      scope='Offline diagnostic on overlapping contiguous provider clips. Native accumulated progress is a historical training label, not future return. Raw proposed actions predict realized environment outcomes including guards. Observed-state utility change isolates the value readout from forecast error; no counterfactual action ranking or independent sample significance is established. No fitting or policy changes.'))
    print(json.dumps(dict(clips=len(clips),levels=results),indent=2))


if __name__ == '__main__':
    main()
