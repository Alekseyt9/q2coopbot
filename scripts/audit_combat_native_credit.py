"""Independent raw-event/window/reward/terminal TRAIN audit, without NN calls."""
import argparse
import collections
import json
import math
import re
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from audit_registered_damage_timing import damage_events


def raw_windows(path,events):
    pattern=re.compile(r'^sv_test_step version=1 phase=(begin|end) spawncount=(-?\d+) frame=(\d+) seq=(\d+) actor=(\d+)$')
    windows={};pending=None;index=0
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            line=line.rstrip('\r\n')
            if line.startswith('sv_test_damage '):
                assert index<len(events)
                if pending is not None:
                    e=events[index];assert e['spawncount']==pending['spawncount'] and pending['begin_frame']<=e['frame']<=pending['begin_frame']+1
                    pending['damage_indexes'].append(index)
                index+=1
            elif line.startswith('sv_test_step '):
                match=pattern.fullmatch(line);assert match,'Malformed native boundary'
                phase,generation,frame,seq,actor=match.groups();generation,frame,seq,actor=map(int,(generation,frame,seq,actor))
                key=(generation,actor,seq)
                if phase=='begin':
                    assert pending is None and key not in windows
                    pending=dict(spawncount=generation,actor=actor,sequence=seq,begin_frame=frame,damage_indexes=[])
                else:
                    assert pending is not None and key==(pending['spawncount'],pending['actor'],pending['sequence']) and frame==pending['begin_frame']+1
                    pending['end_frame']=frame;windows[key]=pending;pending=None
    assert pending is None and index==len(events)
    return windows


def spacing(o,c):
    distances=[math.hypot(*e['relative'][:2]) for e in o.get('enemies') or [] if e['class']=='monster_parasite' and e.get('clear_shot')]
    return -c['spacing_potential']*max(0.,1-min(distances,default=math.inf)/c['parasite_range'])


def effects(events,actor):
    result=dict(monster_health_damage=0,received_health_damage=0,self_health_damage=0,teammate_health_damage=0,monster_kills=0,deaths=0)
    for e in events:
        damage=max(e['health_before'],0)-max(e['health_after'],0);dead=e['health_before']>0>=e['health_after']
        if e['target']==actor:result['received_health_damage']+=damage;result['deaths']+=int(dead)
        if e['attacker']==actor:
            if e['target']==actor:result['self_health_damage']+=damage
            elif e['target_class'].startswith('monster_'):result['monster_health_damage']+=damage;result['monster_kills']+=int(dead)
            elif e['target_class']=='player':result['teammate_health_damage']+=damage
    return result


def unique_rows(path,key):
    rows=list(map(json.loads,path.open(encoding='utf-8-sig')));d={r[key]:r for r in rows}
    assert len(d)==len(rows),'Duplicate index: '+str(path)
    return d


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    closed=read(pin(a.corpus/'complete.json'));assert closed['state']=='complete' and closed['native_cases']==closed['cuda_cases']>0
    protocol=read(pin(a.corpus/'protocol.json',closed['protocol_sha256']));assert protocol['train_only']
    pool=read(pin(a.corpus/'pool/report.json',closed['pool_sha256']))
    assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])>=closed['native_cases']
    assert all(not j['error'] for j in pool['jobs'])
    jobs={j['job']:j for j in pool['jobs']};processing=read(pin(a.corpus/'processing-train/report.json'))
    assert len(jobs)==len(pool['jobs']) and len({c['job'] for c in processing['cases']})==closed['native_cases']
    counts=collections.Counter();components=collections.Counter();cases=[];handoffs=[];unassigned=[];telescoping=[];post_goal_tail=[]
    for case_info in processing['cases']:
        job=jobs[case_info['job']];assert not job['error'] and case_info['split']=='train' and job['seed']==case_info['seed']
        capture=read(pin(Path(job['root'])/'report.json'));manifest=read(pin(Path(job['root'])/'manifest.json'))
        assert capture['capture_complete'] and capture['provenance_valid'];result=capture['results'][0]
        log=pin(Path(result['root'])/'server.log')
        with log.open(encoding='utf-8-sig') as stream:raw=list(damage_events(stream))
        windows=raw_windows(log,raw)
        directory=a.corpus/'processing-train'/f"case-{job['job']:04d}"
        meta=read(pin(directory/'cuda/report.json'));assert meta['numerical_verification']=='cuda_verified_v1'
        files=list((directory/'native').glob('replay-*/steps.jsonl'));assert len(files)==1;replay=files[0].parent
        for name in ('steps.jsonl','rewards.jsonl','server_outcomes.jsonl'):
            path=(replay/name).resolve();pin(path,meta['source_sha256'][str(path)])
        config=read(pin(replay/'reward-config.json'));assert config['version']=='combat_reward_v13'
        steps=unique_rows(replay/'steps.jsonl','index');rewards=unique_rows(replay/'rewards.jsonl','step');outcomes=unique_rows(replay/'server_outcomes.jsonl','step')
        exported=unique_rows(pin(directory/'cuda/rollout.jsonl',meta['rollout_sha256']),'index')
        c=collections.Counter();used=set();ordered=list(exported.values());segments=[]
        for i,row in enumerate(ordered):
            s=steps[row['index']];r=rewards[row['index']];o=outcomes[row['index']];native=s['native_step'];ident=s['observation']['identity'];nxt=s['next_observation']
            assert ident['life']==1 and s['owner']=='provider' and r['available'] and o['available']
            assert row['seed']==job['seed'] and row['frame']==ident['frame'] and row['next_frame']==nxt['identity']['frame']
            assert native['begin_frame']==row['frame'] and native['end_frame']==row['next_frame']==row['frame']+1
            assert native['sequence']==s['client_sequence'] and native['actor']==ident['actor'] and native['spawncount']==ident['spawncount']
            assert s['server_execution']['matched'] and s['server_execution']['window_exclusive'] and s['server_execution']['recovery_commands']==0
            window=windows[(native['spawncount'],native['actor'],native['sequence'])]
            assert native==window,'Stored window differs from raw begin/end boundary ordering'
            expected=[k for k in window['damage_indexes'] if raw[k]['attacker']==ident['actor'] or raw[k]['target']==ident['actor']]
            selected=[k for k in native.get('damage_indexes',[]) if raw[k]['attacker']==ident['actor'] or raw[k]['target']==ident['actor']]
            assert selected==expected,'Native index versus independently derived window differs'
            assert not used.intersection(expected);used.update(expected)
            ev=[raw[k] for k in expected];assert o['events']==ev
            counts['effects_at_begin_frame']+=sum(e['frame']==native['begin_frame'] for e in ev)
            counts['effects_at_end_frame']+=sum(e['frame']==native['end_frame'] for e in ev)
            measured=effects(ev,ident['actor'])
            for key,value in measured.items():assert o[key]==value;counts[key]+=value;c[key]+=value
            weights={'monster_health_damage':('monster_damage','monster_damage'),'received_health_damage':('received_damage','received_damage'),
                'self_health_damage':('self_damage_extra','self_damage_extra'),'teammate_health_damage':('friendly_damage','friendly_damage'),
                'monster_kills':('monster_kill','monster_kill'),'deaths':('death','death')}
            for key,(component,weight) in weights.items():assert abs(r['components'][component]-measured[key]*config[weight])<1e-8
            assert abs(row['reward']-r['score'])<1e-8 and abs(math.fsum(r['components'].values())-row['reward'])<1e-8
            handoff_mode=config.get('handoff_bootstrap','')
            assert handoff_mode in ('','nonterminal_v1')
            zero=bool(s['terminal'] or (s['truncated'] and s.get('reason')=='control_handoff' and
                      (handoff_mode=='' or s.get('goal_handoff_verified',False))))
            assert row['bootstrap_zero']==zero and (not zero or row['next_value']==0)
            if measured['deaths']:
                assert row['terminal'] and s['reason']=='observed_death' and nxt['health']<=0
            if s.get('reason')=='control_handoff':
                goal=result['goal_stop'];confirmed=bool(goal and goal['kill_frame']<=native['end_frame']<=goal['observed_frame'] and nxt['health']>0)
                if handoff_mode=='nonterminal_v1':
                    assert bool(s.get('goal_handoff_verified',False))==confirmed,'Goal marker differs from raw supervisor evidence'
                    assert row['bootstrap_zero']==confirmed,'Unfinished handoff must retain CUDA next-state value'
                else:
                    assert confirmed,'Control handoff ended an unfinished goal'
                handoffs.append(dict(job=job['job'],seed=job['seed'],index=row['index'],confirmed_goal=confirmed,
                    bootstrap_contract_verified=True,handoff_bootstrap=handoff_mode or 'legacy_goal_only',
                    bootstrap_zero=row['bootstrap_zero'],next_value=row['next_value'],
                    end_frame=native['end_frame'],goal_kill_frame=goal['kill_frame'] if goal else None,kill_reward=r['components']['monster_kill']))
                c['goal_handoff' if confirmed else 'unfinished_handoff']+=1
            if s.get('reason')=='combat_goal_complete':
                assert result['goal_stop'] and row['terminal'] and nxt['health']>0 and row['next_frame']>=result['goal_stop']['kill_frame']
                c['goal_terminal']+=1
            before=spacing(s['observation'],config);after=0. if zero else spacing(nxt,config)
            assert abs(r['components']['spacing_potential']-(config['aim_gamma']*after-before))<1e-8
            ref=r.get('aim_reference')
            if ref and ref['applied']:
                assert ref['reason']=='stable_selected_target' and abs(r['components']['aim_potential']-config['aim_gamma']*(ref['after']-ref['before']))<1e-8
            else:assert r['components']['aim_potential']==0
            counts['rows']+=1;components.update(r['components'])
            c['projectile_inflictor_effects']+=sum(e['attacker']==ident['actor'] and e['inflictor']!=ident['actor'] and e['target_class'].startswith('monster_') for e in ev)
            previous=ordered[i-1] if i else None
            consecutive=previous is not None and not previous['terminal'] and not previous['truncated'] and row['index']==previous['index']+1 and row['frame']==previous['next_frame']
            if not consecutive:segments.append([])
            segments[-1].append((row,s,r,before,after))
        life=result['first_life'];actor=life['bot_entity']
        goal=result['goal_stop']
        if goal:
            lethal=[e for e in raw if e['map']==life['map'] and e['spawncount']==goal['spawncount'] and e['attacker']==actor and e['health_before']>0>=e['health_after'] and e['target_class'].startswith('monster_')]
            for kill in goal['kills']:assert any(e['target']==kill['target'] and e['target_class']==kill['target_class'] and e['frame']==kill['frame'] for e in lethal)
            assert goal['kill_frame']==max(k['frame'] for k in goal['kills'])
            assert goal['kill_frame']<goal['observed_frame']<=life['end_frame'] and goal['health']>0
        first_life_events=[k for k,e in enumerate(raw) if e['map']==life['map'] and e['spawncount']==life['spawncount'] and life['start_frame']<=e['frame']<=life['end_frame'] and e['game_frame']>=manifest['release_game_frame'] and (e['attacker']==actor or e['target']==actor)]
        # The same independently verified supervisor boundary used by the
        # exporter closes the finite training task. Keep stop-latency effects
        # visible, but never attach them to an earlier on-policy command.
        tail=[k for k in first_life_events if goal and raw[k]['frame']>goal['observed_frame']]
        assert not used.intersection(tail),'Exported learning window extends beyond verified goal'
        for k in tail:post_goal_tail.append(dict(job=job['job'],seed=job['seed'],goal_observed_frame=goal['observed_frame'],event=raw[k]))
        eligible=[k for k in first_life_events if k not in set(tail)]
        outside=[k for k in eligible if k not in used]
        for k in outside:
            e=raw[k]
            if e['health_before']>0 and (e['target']==actor or e['target_class'].startswith('monster_')):unassigned.append(dict(job=job['job'],seed=job['seed'],event=e))
        for segment in segments:
            gamma=config['aim_gamma'];actual=math.fsum(gamma**i*v[2]['components']['spacing_potential'] for i,v in enumerate(segment))
            expected=-segment[0][3]+gamma**len(segment)*segment[-1][4]
            jumps=[abs(segment[i-1][4]-segment[i][3]) for i in range(1,len(segment))]
            error=abs(actual-expected)
            telescoping.append(dict(job=job['job'],seed=job['seed'],start_index=segment[0][0]['index'],rows=len(segment),
                discounted_spacing=actual,expected_boundary=expected,error=error,max_adjacent_potential_jump=max(jumps,default=0.)))
        cases.append(dict(job=job['job'],seed=job['seed'],rows=len(exported),counts=dict(c),segments=len(segments),goal=bool(goal),death=bool(result['death_stop']),
            native_events=len(eligible),joined_events=len(used),unassigned_post_release_events=len(outside),excluded_post_goal_tail_events=len(tail)))
        print(job['job'],'native effects checked',len(exported),'rows',flush=True)
    assert len(cases)==closed['native_cases'] and counts['rows']==processing['rows']
    for path,digest in pins.items():assert sha(Path(path))==digest
    a.out.mkdir(parents=True)
    report=dict(state='complete',version='combat_native_credit_audit_v1',neural_execution=False,training_performed=False,new_games=False,
        totals=dict(counts),component_sums=dict(components),cases=cases,handoffs=handoffs,
        unassigned_native_effects=unassigned,excluded_post_goal_tail_effects=post_goal_tail,spacing_segments=telescoping,
        spacing_telescoping_max_error=max(x['error'] for x in telescoping),spacing_adjacent_max_jump=max(x['max_adjacent_potential_jump'] for x in telescoping),
        inputs=pins,script_sha256=sha(__file__),
        scope='Independent raw-log actor/world/frame join and reward arithmetic. Effect-time reward is not proof the latest action launched a projectile. Selected aim is an explicitly masked progress preference, not invariant potential. No fitting or claim of optimal reward design.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print({k:report[k] for k in ('totals','spacing_telescoping_max_error','spacing_adjacent_max_jump')})


if __name__=='__main__':main()
