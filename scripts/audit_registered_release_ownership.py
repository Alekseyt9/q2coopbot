"""Controller ownership from canonical native network release, not game frame."""
import argparse
import collections
import json
import pathlib
from process_combat_architecture_pool import read,save,sha
from process_registered_combat_pool_cuda import select_learned_jobs
from native_combat_release import release_boundary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pool',type=pathlib.Path,required=True)
    ap.add_argument('--plan-index',type=int,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists();pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path);digest=sha(path);assert expected is None or expected==digest
        pins[str(path.resolve())]=digest;return path
    pool=read(pin(args.pool/'report.json'));assert pool['state']=='complete' and pool['source_unchanged']
    jobs=select_learned_jobs(pool,args.plan_index);cases=[]
    for job in jobs:
        root=pathlib.Path(job['root']);capture=read(pin(root/'report.json'));manifest=read(pin(root/'manifest.json'))
        assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results'])==1
        result=capture['results'][0];assert result['capture_valid'] and result['dispatch_valid'] and result['seed_confirmed']
        assert result['seed']==job['seed']
        worker=pathlib.Path(result['root']);log=pin(worker/'server.log');trace=pin(worker/'bot.jsonl')
        with log.open(encoding='utf-8-sig') as stream:generation,release,game_release=release_boundary(stream,manifest,job['seed'])
        owners=collections.Counter();setup=collections.Counter();fallbacks=collections.Counter();seen=set();events=[]
        early=visible=visible_attack=unknown_action=unknown_observation=0
        with trace.open(encoding='utf-8-sig') as stream:
            for row in map(json.loads,stream):
                policy=row.get('combat_policy') or {};selection=policy.get('selection') or {};owner=selection.get('owner')
                if not owner:continue
                obs=policy.get('observation')
                if not obs:unknown_observation+=1;continue
                identity=obs['identity']
                if identity['life']!=1 or obs['health']<=0:continue
                key=tuple(identity.get(k) for k in ('map','spawncount','connection','actor','life','frame'))
                assert key not in seen;seen.add(key)
                assert owner in ('provider','rules')
                if identity['spawncount']!=generation or identity['frame']<release:
                    setup[owner]+=1;continue
                owners[owner]+=1;early+=identity['frame']<game_release
                if owner=='provider':continue
                fallbacks[selection.get('fallback') or 'unspecified']+=1
                if not any(e.get('clear_shot') for e in obs.get('enemies') or []):continue
                visible+=1;action=policy.get('applied')
                if action is None:unknown_action+=1
                else:visible_attack+=bool(action.get('attack'))
                events.append(dict(frame=identity['frame'],fallback=selection.get('fallback'),attack=action.get('attack') if action else None))
        cases.append(dict(job=job['job'],seed=job['seed'],release_generation=generation,network_release_frame=release,
            native_release_game_frame=game_release,owners=dict(owners),setup_owners=dict(setup),fallbacks=dict(fallbacks),
            formerly_dropped_released_frames=early,visible_enemy_rules_frames=visible,visible_enemy_rules_attack_frames=visible_attack,
            visible_rules_missing_applied_frames=unknown_action,unknown_observation_rows=unknown_observation,
            visible_rules_events=events,goal=bool(result['goal_stop']),capture_sha256=pins[str((root/'report.json').resolve())],
            manifest_sha256=pins[str((root/'manifest.json').resolve())],trace_sha256=pins[str(trace.resolve())],server_log_sha256=pins[str(log.resolve())]))
    fields=('formerly_dropped_released_frames','visible_enemy_rules_frames','visible_enemy_rules_attack_frames','visible_rules_missing_applied_frames','unknown_observation_rows')
    totals={key:sum(c[key] for c in cases) for key in fields}
    totals['owners']=dict(sum((collections.Counter(c['owners']) for c in cases),collections.Counter()))
    for p,h in pins.items():assert sha(p)==h
    save(args.out,dict(state='complete',version='registered_combat_native_release_ownership_v3',cases=cases,totals=totals,
        pool_sha256=sha(args.pool/'report.json'),selected_plan_index=args.plan_index,
        visible_control_clean=all(totals[k]==0 for k in fields if k!='formerly_dropped_released_frames'),
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
        release_helper_sha256=sha(pathlib.Path(__file__).with_name('native_combat_release.py')),
        training_performed=False,neural_execution=False,promotion=False,
        scope='Logged first-life alive ownership from actual native network release/generation. Unlike legacy game-frame cutoff, includes released network frames98-99 when native release game frame100. Counts visible-enemy rules decisions and missing evidence. Zero logged rules frames does not prove ownership of unlogged dispatches or cover campaign/coop. Historical audits and captures are unchanged.'))
    print(json.dumps(dict(cases=len(cases),totals=totals)))


if __name__=='__main__':main()
