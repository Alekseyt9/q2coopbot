"""Compact source-bound871 transition tables and terminal-aware world clips."""
import argparse
import collections
import json
import pathlib
import torch
from train_combat_hjepa import sha,save
from audit_combat_hjepa_world_actions import primitive
from prepare_combat_hjepa_effects import effect_vectors,WEIGHTS


def clip_windows(records):
    groups=collections.defaultdict(list)
    for row in records:groups[row['episode_key']].append(row)
    clips=[];lengths=[]
    for rows in groups.values():
        rows.sort(key=lambda r:r['frame'])
        if len({r['frame'] for r in rows})!=len(rows):raise ValueError('duplicate source frame')
        for start in range(len(rows)):
            chunk=[]
            for row in rows[start:start+16]:
                if chunk and (row['frame']!=chunk[-1]['frame']+1 or row['features']!=chunk[-1]['next_features']):break
                chunk.append(row)
                if row['terminal']:break
            if len(chunk)!=16 and not chunk[-1]['terminal']:continue
            ids=[r['tensor_index'] for r in chunk]
            clips.append(ids+[ids[-1]]*(16-len(ids)));lengths.append(len(ids))
    return clips,lengths


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=pathlib.Path,required=True)
    ap.add_argument('--partition',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args();mp=pathlib.Path(str(a.data)+'.meta.json');meta=json.loads(mp.read_text())
    if meta['version']!='hjepa_data_v1' or meta['feature_version']!='hjepa_context_features_v1' or not meta['guarded_world'] or not meta['terminal_world']:
        raise ValueError('typed871guarded/terminal world export required')
    partition=json.loads(a.partition.read_text());fit=set(partition['fit_engine_seeds']);held=set(partition['internal_holdout_engine_seeds'])
    if fit&held or len(fit)!=48 or len(held)!=16:raise ValueError('world partition changed')
    grouped=collections.defaultdict(list);count=0
    for line in a.data.open():
        row=json.loads(line);grouped[row['source_sha256']].append(row);count+=1
    if count!=meta['rows']:raise ValueError('export row count changed')
    inputs={str(a.data.resolve()):sha(a.data),str(mp.resolve()):sha(mp),str(a.partition.resolve()):sha(a.partition)}
    features=[];next_features=[];actions=[];effects=[];seeds=[];frames=[];step_ids=[];source_ids=[];source_paths=[];records=[];proof=[]
    for name,digest in meta['sources_sha256'].items():
        p=pathlib.Path(name)
        if sha(p)!=digest:raise ValueError('native source changed')
        inputs[str(p.resolve())]=digest
        if p.name!='steps.jsonl':continue
        report_path=p.parent.parent.parent/'report.json';report=json.loads(report_path.read_text());inputs[str(report_path.resolve())]=sha(report_path)
        candidates=[r for r in report['results'] if pathlib.Path(r['root']).resolve()==p.parent.parent.resolve()]
        if not report['capture_complete'] or not report['provenance_valid'] or len(candidates)!=1:raise ValueError('invalid native receipt')
        result=candidates[0];seed=result['seed']
        if not result['capture_valid'] or result['generated_fixture']['split']!='train' or seed not in fit|held:raise ValueError('wrong native split/world')
        steps=[json.loads(l) for l in p.open()];indexed={s['observation']['identity']['frame']:s for s in steps}
        if len(indexed)!=len(steps):raise ValueError('ambiguous native frame')
        op=p.with_name('server_outcomes.jsonl');inputs[str(op.resolve())]=sha(op)
        vectors,ledger=effect_vectors(result,steps,[json.loads(l) for l in op.open()])
        source_index=len(source_paths);source_paths.append(str(p.resolve()));constant=None;admitted=0
        for row in grouped.pop(digest,[]):
            s=indexed[row['frame']]
            if s['owner']!='provider' or row['frame']>=ledger['end_frame']:continue
            if not s.get('native_step') or not s['server_execution']['matched'] or not s['server_execution']['window_exclusive']:raise ValueError('unverified native world transition')
            actual=primitive(s['applied_action']);proposed=primitive(s['action'])
            if any(abs(x-y)>1e-12 for x,y in zip(actual,row['action'])) or s['action'].get('weapon',''):raise ValueError('action contract differs from proposed primitive keep')
            if constant is None:constant=proposed
            if proposed!=constant:raise ValueError('terminal padding requires declared constant proposals')
            if len(row['features'])!=871 or len(row['next_features'])!=871:raise ValueError('wrong world width')
            index=len(features);features.append(row['features']);next_features.append(row['next_features']);actions.append(proposed);effects.append(vectors[s['index']])
            seeds.append(seed);frames.append(row['frame']);step_ids.append(s['index']);source_ids.append(source_index);admitted+=1
            records.append(dict(episode_key=(digest,row['episode']),frame=row['frame'],features=row['features'],next_features=row['next_features'],tensor_index=index,terminal=s['index']==ledger['terminal_step']))
        proof.append(dict(seed=seed,source_sha256=digest,provider_rows=admitted,scalar_return=ledger['scalar_return'],terminal_step=ledger['terminal_step'],events_outside_first_life=ledger['events_outside_first_life']))
    if grouped:raise ValueError('unbound source rows')
    clips,lengths=clip_windows(records)
    if not clips:raise ValueError('no continuous world clips')
    clip_seeds=[seeds[c[0]] for c in clips]
    if set(seeds)!=fit|held:raise ValueError('source-world coverage incomplete')
    data=dict(version='hjepa_joint_world_data_v1',features=torch.tensor(features),next_features=torch.tensor(next_features),
              actions=torch.tensor(actions),effects=torch.tensor(effects),seeds=torch.tensor(seeds),frames=torch.tensor(frames),
              step_indexes=torch.tensor(step_ids),source_indexes=torch.tensor(source_ids),source_paths=source_paths,
              clip_indexes=torch.tensor(clips),clip_lengths=torch.tensor(lengths),clip_fit=torch.tensor([s in fit for s in clip_seeds]),
              reward_weights=torch.tensor(WEIGHTS))
    if not all(torch.isfinite(data[k]).all() for k in ['features','next_features','actions','effects']):raise ValueError('nonfinite world tensors')
    a.out.mkdir(exist_ok=False,parents=True);torch.save(data,a.out/'data.pt')
    stats=dict(transitions=len(features),clips=len(clips),fit_clips=sum(s in fit for s in clip_seeds),holdout_clips=sum(s in held for s in clip_seeds),
               terminal_absorbed_clips=sum(n<16 for n in lengths),fit_clip_worlds=len({s for s in clip_seeds if s in fit}),holdout_clip_worlds=len({s for s in clip_seeds if s in held}))
    save(a.out/'meta.json',dict(version='hjepa_joint_world_data_v1',feature_version=meta['feature_version'],stats=stats,data_sha256=sha(a.out/'data.pt'),
         sources_sha256=inputs,episode_ledgers=proof,preparer_sha256=sha(pathlib.Path(__file__)),
         scope='Actual871client transitions and proposed primitive8commands under constant diagnostic policies. No BC labels. Native effects are targets only. Compact indexes do not renumber source frames/steps/worlds.16frame clips break at source/life/control gaps; only verified terminal endpoints may repeat their next-state target and planned constant action. clip_lengths masks all repeated effect labels, so terminal reward is credited once. Fit/holdout remain disjoint TRAIN worlds; no seed/source/length/outcome metadata is a neural input.'))
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
