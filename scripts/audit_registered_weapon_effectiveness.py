"""Sealed observed weapon response metrics; no inference or physical-shot claims."""
import argparse,collections,json,pathlib
from audit_registered_weapon_recovery import summarize as recovery_summary
from combat_equipped_weapon_features import equipped_weapon_index
from combat_weapon_features import inventory_tail,MINIMUM
from process_combat_architecture_pool import read,save,sha


def summarize(frames):
    result=recovery_summary(frames);counts=collections.Counter();previous=None;first=attack=None
    for obs,applied,owner in frames:
        identity=obs['identity'];frame=identity['frame']
        world=tuple(identity[k] for k in ('map','spawncount','connection','actor','life'))
        if owner!='provider':previous=None;continue
        current=equipped_weapon_index(obs);request=applied.get('weapon') or ''
        requested=equipped_weapon_index(dict(weapon=request)) if request else 0
        if first is None:first=frame
        if applied.get('attack') and attack is None:attack=frame
        contiguous=previous is not None and previous[:2]==(world,frame-1)
        counts['observed_equipped_changes']+=int(bool(contiguous and previous[2] and current and previous[2]!=current))
        mask=inventory_tail(obs)[19:31];fresh=bool(obs.get('inventory') is not None and isinstance(obs.get('inventory_age_frames'),int) and 0<=obs['inventory_age_frames']<=20)
        ammo=obs.get('ammo')
        if current in (2,3,4) and isinstance(ammo,int) and ammo>=0 and ammo<MINIMUM[current]:
            counts['below_stock_min_hud_frames']+=1
            counts['below_stock_min_hud_attack_commands']+=int(bool(applied.get('attack')))
            if fresh:
                counts['fresh_inventory_below_min_hud_frames']+=1
                counts['hud_low_inventory_still_allows_current_frames']+=int(bool(mask[current]))
                alternatives=[i for i in range(1,5) if i!=current and mask[i]]
                if alternatives:
                    counts['hud_low_with_available_alternative_frames']+=1
                    counts['hud_low_keep_or_same_request_frames']+=int(not request or requested==current)
                    counts['hud_low_alternative_request_frames']+=int(requested in alternatives)
        previous=(world,frame,current)
    result['efficiency_counts']=dict(counts)
    result['first_attack_command_delay_frames']=None if attack is None else attack-first
    result['request_statuses']=dict(collections.Counter(e['status'] for e in result['choice_audit']['request_events']))
    return result


def aggregate(cases):
    delays=[c['first_supported_delay_frames'] for c in cases if c['first_supported_delay_frames'] is not None]
    attacks=[c['first_attack_command_delay_frames'] for c in cases if c['first_attack_command_delay_frames'] is not None]
    return dict(cases=len(cases),wins=sum(c['goal'] for c in cases),
        observed_supported_cases=sum(c['observed_supported'] for c in cases),
        without_weapon_request_cases=sum(c['first_request_delay_frames'] is None for c in cases),
        without_attack_command_cases=sum(c['first_attack_command_delay_frames'] is None for c in cases),
        first_supported_delay_frames_mean=sum(delays)/len(delays) if delays else None,
        first_attack_command_delay_frames_mean=sum(attacks)/len(attacks) if attacks else None,
        counts=dict(sum((collections.Counter(c['counts']) for c in cases),collections.Counter())),
        efficiency_counts=dict(sum((collections.Counter(c['efficiency_counts']) for c in cases),collections.Counter())),
        request_statuses=dict(sum((collections.Counter(c['request_statuses']) for c in cases),collections.Counter())))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation',type=pathlib.Path,required=True)
    ap.add_argument('--spec',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists();root=args.evaluation.resolve()
    seal=read(root/'complete.json');spec=read(args.spec)
    assert seal['state']=='complete' and not seal['training_performed'] and sha(args.spec)==seal['spec_sha256']
    pool_path=root/'pool/report.json';pool=read(pool_path)
    assert pool['state']=='complete' and pool['source_unchanged'] and sha(pool_path)==seal['pool_sha256']
    pins={root/'complete.json':sha(root/'complete.json'),args.spec:sha(args.spec),pool_path:seal['pool_sha256']}
    variants={}
    for index,variant in enumerate(spec['variants']):
        if variant['mode']!='learned':continue
        label=variant['label'];plan_path=root/(label+'-plan.json');plan=read(plan_path)
        assert sha(plan_path)==pool['plans'][index]['sha256'];pins[plan_path]=sha(plan_path)
        execution_path=root/('execution-'+label+'.json');execution=read(execution_path);pins[execution_path]=sha(execution_path)
        assert execution['state']=='complete' and execution['pool_sha256']==seal['pool_sha256'] and execution['selected_plan_index']==index
        bindings={v['job']:v for v in execution['cases']};cases=[]
        for job in pool['jobs']:
            if job['plan_index']!=index:continue
            assert not job['error'];binding=bindings[job['job']]
            capture_path=pathlib.Path(job['root'])/'report.json';assert sha(capture_path)==binding['capture_sha256']
            capture=read(capture_path);result=capture['results'][0];trace=pathlib.Path(result['root'])/'bot.jsonl'
            assert sha(trace)==binding['trace_sha256'];frames=[];seen=set()
            with trace.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    control=json.loads(line).get('combat_policy') or {};owner=(control.get('selection') or {}).get('owner')
                    if not owner:continue
                    obs=control['observation'];identity=obs['identity']
                    if identity['life']!=1 or obs['health']<=0 or identity['frame']<binding['network_release_frame']:continue
                    assert identity['spawncount']==binding['release_generation']
                    key=tuple(identity[k] for k in ('map','spawncount','connection','actor','life','frame'))
                    assert key not in seen;seen.add(key);frames.append((obs,control.get('applied'),owner))
            case=summarize(frames);assert sha(trace)==binding['trace_sha256']
            episode=plan['tasks'][job['task_index']]['episode']['id']
            case.update(job=job['job'],episode=episode,seed=job['seed'],goal=bool(result['goal_stop']),
                trace_sha256=binding['trace_sha256'],capture_sha256=binding['capture_sha256'])
            cases.append(case)
        assert len(cases)==len(execution['cases'])
        groups={'all':cases,'recovery':[c for c in cases if c['episode'].endswith('-recovery')],
                'ordinary':[c for c in cases if not c['episode'].endswith('-recovery')]}
        variants[label]=dict(cases=cases,groups={k:aggregate(v) for k,v in groups.items() if v})
        print(label,'cases',len(cases),flush=True)
    for p,d in pins.items():assert sha(p)==d,str(p)
    save(args.out,dict(state='complete',version='registered_weapon_effectiveness_observation_v1',
        pool_sha256=seal['pool_sha256'],evaluation_sha256=pins[root/'complete.json'],script_sha256=sha(__file__),
        helpers={name:sha(pathlib.Path(__file__).with_name(name)) for name in ['audit_registered_weapon_recovery.py','audit_registered_weapon_choices.py','combat_weapon_features.py','combat_equipped_weapon_features.py']},
        variants=variants,training_performed=False,promotion=False,
        scope='First-life provider choices after verified network release. HUD ammo below stock minima, observed equipment and inventory availability are separate observations; not proof of inability to fire or completed dispatch. Delays censor cases without events; request supersession/attack commands are not physical shots or causal waste. No inference or fitting.'))


if __name__=='__main__':main()
