"""Boundary-checked native reward-to-go labels; never policy observations."""
import bisect
import collections
import math
import pathlib
import json
from train_combat_hjepa import sha


def episode_ledger(result, steps, outcomes):
    life=result['first_life'];actor=life['bot_entity']
    start,end=life['start_frame'],life['end_frame']
    indexed={s['index']:s for s in steps}
    if len(indexed)!=len(steps):raise ValueError('duplicate causal step index')
    reward=collections.defaultdict(float)
    sums=dict(received_health_damage=0,self_health_damage=0,teammate_health_damage=0,monster_health_damage=0)
    initial_hp=sum(x['health'] for x in result['generated_start']['entities'] if x['class'].startswith('monster_'))
    previous_progress=0.;outside=0;seen=set()
    for outcome in outcomes:
        index=outcome['step']
        if index in seen or index not in indexed:raise ValueError('duplicate/missing native outcome step')
        seen.add(index);s=indexed[index];native=s.get('native_step')
        if not outcome['available']:
            if outcome['events']:raise ValueError('events in unavailable native window')
            continue
        if not native or not s['server_execution']['matched'] or outcome['version']!='server_step_effects_v1':
            raise ValueError('causal native join required')
        if native['end_frame']!=native['begin_frame']+1 or native['actor']!=actor:
            raise ValueError('single native actor tick required')
        for e in outcome['events']:
            if (e['map']!=life['map'] or e['spawncount']!=life['spawncount'] or
                    not native['begin_frame']<=e['frame']<=native['end_frame'] or
                    (e['target']!=actor and e['attacker']!=actor)):
                raise ValueError('event outside causal native actor/window')
            if not start<=e['frame']<=end:
                outside+=1;continue
            if e['health_before']-e['take']!=e['health_after']:
                raise ValueError('inconsistent native health loss')
            loss=max(0,min(max(0,e['health_before']),e['take']))
            if e['target']==actor:
                sums['received_health_damage']+=loss;reward[index]-=.1*loss
                if e['attacker']==actor:
                    sums['self_health_damage']+=loss;reward[index]-=loss
            elif e['attacker']==actor:
                if e['target_class'].startswith('monster_'):
                    sums['monster_health_damage']+=loss
                    progress=min(1.,sums['monster_health_damage']/max(1,initial_hp))
                    reward[index]+=10*(progress-previous_progress);previous_progress=progress
                elif e['target_class']=='player':
                    sums['teammate_health_damage']+=loss;reward[index]-=.5*loss
    if seen!=set(indexed):raise ValueError('missing native outcome rows')
    damage=life['damage']
    expected={k:damage[k] for k in sums if k!='monster_health_damage'}
    expected['monster_health_damage']=sum(x['health_damage'] for x in life['outgoing_by_target_class'])
    if sums!=expected:raise ValueError(f'first-life reward balance mismatch: {sums} != {expected}')
    death=bool(result['death_stop']);win=bool(result['goal_stop']) and not death
    terminal=None
    if death or win:
        receipt=result['death_stop'] if death else result['goal_stop']
        observed=receipt['observed_frame']
        candidates=[s['index'] for s in steps if s.get('next_observation') and
                    s['next_observation']['identity']['frame']==observed and s.get('native_step')]
        if len(candidates)!=1:raise ValueError('terminal observation lacks unique causal transition')
        terminal=candidates[0];reward[terminal]+= -100 if death else 100
    scalar=(100*win-100*death+10*min(1.,expected['monster_health_damage']/max(1,initial_hp))
            -.1*expected['received_health_damage']-expected['self_health_damage']-.5*expected['teammate_health_damage'])
    if not math.isclose(sum(reward.values()),scalar,abs_tol=1e-8):raise ValueError('scalar return changed')
    # Step indexes, not event.frame guesses, preserve immediate ClientThink vs
    # delayed world effects. Bonus belongs to the transition reaching the
    # verified terminal observation. Inclusive same-frame costs retain the
    # existing first-life accounting; decisions after its endpoint have no credit.
    ordered=sorted(indexed)
    if any(indexed[b]['observation']['identity']['frame']<=indexed[a]['observation']['identity']['frame']
           for a,b in zip(ordered,ordered[1:])):raise ValueError('nonmonotonic causal steps')
    suffix={};total=0.
    for index in reversed(ordered):
        total+=reward[index];suffix[index]=total
    timeline=[(indexed[index]['observation']['identity']['frame'],suffix[index]) for index in ordered]
    return dict(reward=dict(reward),suffix=suffix,timeline=timeline,scalar_return=scalar,
                terminal_step=terminal,end_frame=end,events_outside_first_life=outside)


def at_frame(ledger,frame):
    timeline=ledger['timeline']
    index=bisect.bisect_left([v[0] for v in timeline],frame)
    return timeline[index][1] if index<len(timeline) else 0.


def build_credit(rows,episodes):
    ledgers={};inputs={};proof=[]
    for key,ep in episodes.items():
        source=next(r['source'] for r in rows if r['episode_key']==key)
        steps_path=pathlib.Path(source);outcomes_path=steps_path.parent/'server_outcomes.jsonl'
        steps=[json.loads(l) for l in steps_path.read_text().splitlines()]
        outcomes=[json.loads(l) for l in outcomes_path.read_text().splitlines()]
        ledger=episode_ledger(ep['result'],steps,outcomes)
        if not math.isclose(ledger['scalar_return'],ep['reward'],abs_tol=1e-8):raise ValueError('episode objective changed')
        ledgers[key]=ledger
        inputs[str(outcomes_path.resolve())]=sha(outcomes_path)
        proof.append(dict(stream=key[0],seed=key[1],scalar_return=ledger['scalar_return'],
                          terminal_step=ledger['terminal_step'],events_outside_first_life=ledger['events_outside_first_life']))
    masked=0
    for row in rows:
        key=row['episode_key'];ledger=ledgers[key];frame=row['identity']['frame']
        if frame>ledger['end_frame']:
            row['credit_advantage']=0.;masked+=1;continue
        other=[value for (stream,seed),value in ledgers.items() if seed==key[1] and stream!=key[0]]
        own=ledger['suffix'][row['step_index']]
        baseline=sum(at_frame(value,frame) for value in other)/len(other)
        row['credit_advantage']=(own-baseline)/100
    return inputs,dict(version='hjepa_first_life_reward_to_go_v1',episodes=proof,
                       post_endpoint_choices_masked=masked,scope='Label-only causal native reward suffix, '
                       'gamma1 finite horizon, independent other-stream time baseline. No critic or future policy inputs.')
