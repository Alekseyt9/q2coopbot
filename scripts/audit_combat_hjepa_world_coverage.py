"""Source-bound coverage of the actual provider transitions used for world fitting."""
import argparse
import collections
import json
import pathlib
from train_combat_hjepa import sha, save
from hjepa_weapon_returns import episode_ledger


def audit(inputs, split):
    steps_paths=[];sources={};retained=set()
    for name,digest in inputs.items():
        p=pathlib.Path(name)
        if sha(p)!=digest:raise ValueError('changed training evidence: '+name)
        sources[str(p.resolve())]=digest
        if p.name=='steps.jsonl':steps_paths.append(p)
        if str(p).endswith('.jsonl.meta.json'):
            meta=json.loads(p.read_text())
            if meta.get('version')=='hjepa_data_v1':
                data=pathlib.Path(str(p)[:-10])
                rows=[json.loads(line) for line in data.open()]
                if len(rows)!=meta['rows']:raise ValueError('export row count changed')
                retained.update((row['source_sha256'],row['frame']) for row in rows)
    if not retained:raise ValueError('no retained world export rows')
    counts=collections.Counter();worlds={};initial=collections.defaultdict(set)
    observed=collections.Counter();classes=collections.defaultdict(set)
    maps=collections.defaultdict(set);events=collections.Counter();episode_rows=[]
    for p in steps_paths:
        source_digest=sha(p)
        report_path=p.parent.parent.parent/'report.json'
        report=json.loads(report_path.read_text())
        if not report['capture_complete'] or not report['provenance_valid']:raise ValueError('incomplete capture')
        candidates=[r for r in report['results'] if pathlib.Path(r['root']).resolve()==p.parent.parent.resolve()]
        if len(candidates)!=1:raise ValueError('ambiguous native result')
        result=candidates[0];fixture=result['generated_fixture'];seed=fixture['engine_seed']
        if fixture['split']!=split or not result['capture_valid']:raise ValueError('wrong split or invalid native capture')
        if seed in worlds and worlds[seed]!=fixture:raise ValueError('same seed different world')
        worlds[seed]=fixture;initial[fixture['inventory']['initial_weapon']].add(seed)
        maps[fixture['map']].add(seed)
        for monster in fixture['monsters']:classes[monster['class']].add(seed)
        steps=[json.loads(l) for l in p.open()]
        outcomes_path=p.with_name('server_outcomes.jsonl');sources[str(outcomes_path.resolve())]=sha(outcomes_path)
        outcomes=[json.loads(l) for l in outcomes_path.open()]
        ledger=episode_ledger(result,steps,outcomes)
        admitted=[]
        for step in steps:
            if step['owner']!='provider':continue
            if (source_digest,step['observation']['identity']['frame']) not in retained:continue
            execution=step.get('server_execution')
            if not step.get('native_step') or not execution or not execution['matched']:raise ValueError('provider dispatch unverified')
            if not execution['window_exclusive']:continue
            if step['observation']['health']<=0:continue
            admitted.append(step)
            observed[step['observation']['weapon']]+=1
            counts['guarded_provider_rows']+=bool(step.get('interventions'))
        counts['native_episodes']+=1;counts['provider_rows']+=len(admitted)
        counts['provider_nonzero_reward_steps']+=sum(abs(ledger['reward'].get(step['index'],0.))>1e-9 for step in admitted)
        damage=result['first_life']['damage']
        for key in ['received_health_damage','self_health_damage','teammate_health_damage']:
            events[key]+=damage[key]
        episode_rows.append(dict(seed=seed,map=fixture['map'],initial_weapon=fixture['inventory']['initial_weapon'],provider_rows=len(admitted),source=str(p),scalar_return=ledger['scalar_return']))
    return dict(counts=dict(counts),distinct_worlds=len(worlds),initial_weapon_worlds={k:len(v) for k,v in initial.items()},
                monster_class_worlds={k:len(v) for k,v in classes.items()},map_worlds={k:len(v) for k,v in maps.items()},
                observed_weapon_provider_rows=dict(observed),native_health_totals=dict(events),episodes=episode_rows,sources_sha256=sources)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fit-report',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args();fit=json.loads(a.fit_report.read_text())
    train=audit(fit['train_inputs_sha256'],'train');val=audit(fit['validation_inputs_sha256'],'validation')
    save(a.out,dict(version='hjepa_world_coverage_v1',train=train,validation=val,fit_report_sha256=sha(a.fit_report),
                    scope='World counts deduplicate engine seeds; repeated weapon sampling streams are episodes, not new worlds. Provider counts restricted to exported rows, before contiguous16clip filtering. Health totals cover full first lives. All first-life native reward ledgers independently reconciled. No fit or policy change.'))
    print(json.dumps({k:{field:value for field,value in v.items() if field not in ['episodes','sources_sha256']} for k,v in [('train',train),('validation',val)]},indent=2))


if __name__=='__main__':main()
