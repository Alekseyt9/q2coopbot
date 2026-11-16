"""World-separated, native-verified effect targets; outcomes never become inputs."""
import argparse
import collections
import json
import math
import pathlib
from train_combat_hjepa import sha, save
from hjepa_weapon_returns import episode_ledger
from audit_combat_hjepa_world_actions import primitive
from audit_combat_hjepa_weapons import NAMES

EFFECTS = ['monster_progress','received_hp_fraction','self_hp_fraction',
           'teammate_hp_fraction','win','death']
WEIGHTS = [.1,-.1,-1.,-.5,1.,-1.]


def effect_action(action):
    weapon=action.get('weapon','')
    if weapon not in NAMES:raise ValueError('unknown proposed weapon')
    return primitive(action)+[float(weapon==name) for name in NAMES]


def effect_vectors(result, steps, outcomes):
    ledger=episode_ledger(result,steps,outcomes)
    life=result['first_life'];actor=life['bot_entity']
    initial=sum(e['health'] for e in result['generated_start']['entities'] if e['class'].startswith('monster_'))
    vectors={s['index']:[0.]*6 for s in steps};dealt=0.
    for outcome in outcomes:
        v=vectors[outcome['step']]
        for e in outcome['events']:
            if not life['start_frame']<=e['frame']<=life['end_frame']:continue
            loss=max(0,min(max(0,e['health_before']),e['take']))
            if e['target']==actor:
                v[1]+=loss/100
                if e['attacker']==actor:v[2]+=loss/100
            elif e['attacker']==actor:
                if e['target_class'].startswith('monster_'):
                    before=min(1.,dealt/max(1,initial));dealt+=loss
                    v[0]+=min(1.,dealt/max(1,initial))-before
                elif e['target_class']=='player':v[3]+=loss/100
    terminal=ledger['terminal_step']
    if terminal is not None:vectors[terminal][5 if result['death_stop'] else 4]=1.
    for index,v in vectors.items():
        if not math.isclose(sum(a*b for a,b in zip(v,WEIGHTS)),ledger['reward'].get(index,0.)/100,abs_tol=1e-10):
            raise ValueError('effect vector changes verified scalar reward')
    return vectors,ledger


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--choices',type=pathlib.Path,required=True)
    ap.add_argument('--partition',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args();meta_path=pathlib.Path(str(a.choices)+'.meta.json');meta=json.loads(meta_path.read_text())
    if meta['version']!='hjepa_onpolicy_weapon_rollout_v1' or meta['split']!='train' or sha(a.choices)!=meta['data_sha256']:
        raise ValueError('verified TRAIN context export required')
    inputs={str(a.choices.resolve()):sha(a.choices),str(meta_path.resolve()):sha(meta_path),str(a.partition.resolve()):sha(a.partition)}
    for name,digest in meta['sources_sha256'].items():
        if sha(pathlib.Path(name))!=digest:raise ValueError('changed native source')
        inputs[name]=digest
    partition=json.loads(a.partition.read_text());fit=set(partition['fit_engine_seeds']);held=set(partition['internal_holdout_engine_seeds'])
    if fit&held:raise ValueError('source-world overlap')
    episodes={e['seed']:e['native_result'] for e in meta['episodes']}
    if set(episodes)!=fit|held:raise ValueError('partition does not cover exact source worlds')
    choices=collections.defaultdict(list)
    for line in a.choices.open():
        row=json.loads(line);choices[row['seed']].append(row)
    if sum(len(rows) for rows in choices.values())!=meta['rows']:raise ValueError('choice count changed')
    a.out.mkdir(exist_ok=False,parents=True)
    counts=collections.Counter();world_counts=collections.defaultdict(set);proof=[]
    files={role:(a.out/f'{role}.jsonl').open('x') for role in ['fit','internal-holdout']}
    try:
        for seed,rows in choices.items():
            result=episodes[seed];paths={pathlib.Path(r['source']).resolve() for r in rows}
            if len(paths)!=1:raise ValueError('ambiguous episode source')
            source=paths.pop();steps=[json.loads(l) for l in source.open()]
            source_digest=sha(source)
            outcomes_path=source.with_name('server_outcomes.jsonl');inputs[str(outcomes_path)]=sha(outcomes_path)
            outcomes=[json.loads(l) for l in outcomes_path.open()]
            vectors,ledger=effect_vectors(result,steps,outcomes);indexed={s['index']:s for s in steps}
            context={r['step_index']:r for r in rows};role='fit' if seed in fit else 'internal-holdout'
            if len(context)!=len(rows):raise ValueError('duplicate causal choice')
            ordered=sorted(context)
            for pos,index in enumerate(ordered):
                first=indexed[index];frame=first['observation']['identity']['frame']
                features=context[index]['features']
                if len(features)!=871 or not all(math.isfinite(v) for v in features):raise ValueError('context width/value')
                for horizon in [1,4,16]:
                    ids=ordered[pos:pos+horizon]
                    if len(ids)!=horizon:continue
                    chunk=[indexed[i] for i in ids]
                    if any(s['observation']['identity']['frame']!=frame+j or not s.get('next_observation') or
                           s['next_observation']['identity']['frame']!=frame+j+1 or frame+j+1>ledger['end_frame'] or
                           s['observation']['health']<=0 or not s['server_execution']['matched'] or
                           (not s['server_execution']['window_exclusive'] and s['index']!=ledger['terminal_step'])
                           for j,s in enumerate(chunk)):continue
                    effects=[sum(vectors[i][k] for i in ids) for k in range(6)]
                    reward=sum(v*w for v,w in zip(effects,WEIGHTS))
                    output=dict(version='hjepa_native_effect_row_v1',seed=seed,partition=role,frame=frame,horizon=horizon,
                                features=features,actions=[effect_action(s['action']) for s in chunk],effects=effects,
                                reward=reward,source_sha256=source_digest,step_indexes=ids,
                                terminal_nonexclusive=any(not s['server_execution']['window_exclusive'] for s in chunk))
                    files[role].write(json.dumps(output,allow_nan=False)+'\n')
                    counts[f'{role}/h{horizon}']+=1;world_counts[role].add(seed)
            proof.append(dict(seed=seed,partition=role,scalar_return=ledger['scalar_return'],terminal_step=ledger['terminal_step'],events_outside_first_life=ledger['events_outside_first_life']))
    finally:
        for f in files.values():f.close()
    save(a.out/'meta.json',dict(version='hjepa_native_effects_v1',feature_version='hjepa_weapon_features_v1',feature_width=871,
         action_version='primitive8_weapon_request12_v1',action_width=20,effects=EFFECTS,reward_weights=WEIGHTS,counts=dict(counts),
         worlds={k:len(v) for k,v in world_counts.items()},sources_sha256=inputs,episode_ledgers=proof,
         data_sha256={role:sha(a.out/f'{role}.jsonl') for role in files},
         scope='Actual proposed primitive sequences and initial causal871features predict realized native effects including guards. Full contiguous windows only, no padding or post-endpoint credit. Terminal nonexclusive window is marked; no exclusive-dispatch claim for that window. Fit/holdout retain TRAIN identity with disjoint worlds. Outcomes, rewards, seed and source proof fields are labels/metadata, never neural inputs.'))
    print(json.dumps(dict(counts=dict(counts),worlds={k:len(v) for k,v in world_counts.items()}),indent=2))


if __name__=='__main__':main()
