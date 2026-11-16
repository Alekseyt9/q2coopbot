"""Report actual observed weapon diversity in a TRAIN-only exploration probe."""
import argparse
import collections
import json
import pathlib
from audit_combat_hjepa_weapons import canonical, available
from train_combat_hjepa import sha, save


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrix',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args()
    complete=json.loads((a.matrix/'complete.json').read_text(encoding='utf-8-sig'))
    plan_path=a.matrix/'plan.json'
    if complete['state']!='complete' or complete['source_plan_sha256']!=sha(plan_path):
        raise ValueError('complete immutable matrix required')
    plan=json.loads(plan_path.read_text(encoding='utf-8-sig'))
    fixtures={v['engine_seed']:v for task in plan['tasks'] for v in task['instances']}
    if len(fixtures)!=8 or any(f['split']!='train' for f in fixtures.values()):
        raise ValueError('eight TRAIN worlds required')
    protocol=json.loads((a.matrix.parent/'protocol.json').read_text())
    parent_path=pathlib.Path(protocol['parent'])
    if sha(parent_path)!=protocol['parent_sha256']:
        raise ValueError('parent changed')
    parent=json.loads(parent_path.read_text())
    expected={f'{init}-{cadence}-s{s}' for init in ['warm','cold'] for cadence in [0,8] for s in range(4)}
    if {v['name'] for v in complete['models']}!=expected:
        raise ValueError('incomplete factorial probe')
    factors=collections.defaultdict(list)
    initial_states=collections.defaultdict(list)
    for variant in complete['models']:
        init,cadence,stream=variant['name'].split('-');cadence=int(cadence);stream=int(stream[1:])
        path=pathlib.Path(variant['model'])
        if sha(path)!=variant['sha256']:
            raise ValueError('frozen model changed')
        model=json.loads(path.read_text())
        ignored={'weapon_actor','weapon_option_frames','weapon_sampling_seed'}
        if {k:v for k,v in model.items() if k not in ignored}!={k:v for k,v in parent.items() if k not in ignored}:
            raise ValueError('non-weapon model differs')
        if model.get('weapon_option_frames',0)!=cadence or model['weapon_sampling_seed']!=261400+stream:
            raise ValueError('wrong cadence/random stream')
        if init=='warm':
            if model['weapon_actor']!=parent['weapon_actor']:raise ValueError('warm actor differs')
        else:
            if model['weapon_actor'][:-1]!=parent['weapon_actor'][:-1] or any(
                    v!=0 for row in model['weapon_actor'][-1]['weight'] for v in row) or any(model['weapon_actor'][-1]['bias']):
                raise ValueError('cold actor not zero-last-layer initialization')
        seeds=set()
        for report_path in sorted((a.matrix/variant['name']).glob('case-*/report.json')):
            report=json.loads(report_path.read_text())
            if not report['capture_complete'] or not report['provenance_valid']:
                raise ValueError('invalid native case')
            for result in report['results']:
                seed=result['seed']
                if seed in seeds or result['generated_fixture']!=fixtures[seed] or not result['generated_start']['confirmed']:
                    raise ValueError('wrong native fixture/start binding')
                seeds.add(seed)
                if not result['capture_valid'] or not result['dataset']['command_proof']['accepted']:
                    raise ValueError('unverified native capture')
                steps_path=pathlib.Path(result['root'])/'dataset/steps.jsonl'
                rows=[json.loads(line) for line in steps_path.read_text().splitlines()]
                first=next(s for s in rows if s['owner']=='provider' and s.get('native_step') and s['observation']['health']>0)
                start=first['observation']['identity']['frame']
                initial=first['observation']
                initial_states[seed].append({k:initial.get(k) for k in
                    ['health','armor','weapon','position','velocity','view_angles','ducked','inventory']})
                trajectory=[];proposals=collections.Counter();observed=0;following=0;pending=[]
                for s in rows:
                    o=s['observation'];frame=o['identity']['frame']
                    if frame<start or not s.get('native_step') or o['health']<=0:continue
                    if not s['server_execution']['matched']:raise ValueError('unmatched dispatch')
                    weapon=canonical(o['weapon'])
                    if weapon and (not trajectory or trajectory[-1][1]!=weapon):
                        if trajectory:
                            observed+=1
                            if any(w==weapon and 0<frame-f<=30 for f,w in pending):following+=1
                        trajectory.append((frame-start,weapon))
                    if s['owner']=='provider':
                        request=s['action']['weapon']
                        if not available(o,request):raise ValueError('unavailable weapon choice')
                        if request:
                            proposals['nonkeep']+=1
                            if request!=weapon:
                                proposals['different_from_equipped']+=1;pending.append((frame,request))
                life=result['first_life'];damage=life['damage']
                factors[f'{init}-{cadence}'].append(dict(seed=seed,stream=stream,
                    win=bool(result['goal_stop']) and not bool(result['death_stop']),death=bool(result['death_stop']),
                    received_hp=damage['received_health_damage'],self_hp=damage['self_health_damage'],
                    observed_switches=observed,temporally_after_request=following,
                    nonkeep_proposals=proposals['nonkeep'],different_proposals=proposals['different_from_equipped'],
                    trajectory=trajectory,source=str(steps_path)))
        if seeds!=set(fixtures):raise ValueError('missing world')
    if any(any(o!=states[0] for o in states) for states in initial_states.values()):
        raise ValueError('initial observed equipment/physical state differs')
    output={}
    for factor,episodes in factors.items():
        if len(episodes)!=32:raise ValueError('missing factor episodes')
        diversity=[]
        for seed in fixtures:
            trajectories=[e['trajectory'] for e in episodes if e['seed']==seed]
            diversity.append(dict(seed=seed,
                distinct_ordered_weapon_paths=len({tuple(w for _,w in t) for t in trajectories}),
                distinct_timed_weapon_paths=len({tuple(map(tuple,t)) for t in trajectories})))
        keys=['win','death','received_hp','self_hp','observed_switches','temporally_after_request','nonkeep_proposals','different_proposals']
        totals={k:sum(e[k] for e in episodes) for k in keys}
        totals['episodes']=32
        totals['worlds_with_multiple_weapon_paths']=sum(v['distinct_ordered_weapon_paths']>1 for v in diversity)
        output[factor]=dict(totals=totals,diversity=diversity,episodes=episodes)
    save(a.out,dict(version='hjepa_weapon_exploration_probe_v1',promotion=False,initial_observed_core_equal=True,
        factors=output,scope='TRAIN-only paired initialization/cadence probe, no fitting. Observed weapon paths '
        'include engine autoswitch; temporal request matching is not exclusive causation. Actual state diversity '
        'is the exploration measure, not request count. Not final combat quality evidence.'))
    print(json.dumps({k:v['totals'] for k,v in output.items()},indent=2))


if __name__=='__main__':main()
