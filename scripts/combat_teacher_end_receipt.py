"""Canonical native death proof when an optional supervisor stop tag races EOF."""
import json
from pathlib import Path

from combat_teacher_movement_receipt import admitted_terminal_receipt_v2
from process_combat_architecture_pool import read,sha


def admitted_terminal_receipt_v3(result,first_admitted_frame,native_directory):
    receipt,proof=admitted_terminal_receipt_v2(result,first_admitted_frame)
    life=receipt['first_life']
    if life['minimum_health']>0:
        assert result.get('death_stop') is None
        proof['native_death']=None
        return receipt,proof
    assert life['end_reason']=='first_observed_death' and result.get('goal_stop') is None
    directory=Path(native_directory);meta=read(directory/'report.json')
    def bound_file(name):
        matches=[Path(p) for p in meta['source_sha256'] if Path(p).name==name]
        assert len(matches)==1
        path=matches[0];assert sha(path)==meta['source_sha256'][str(path)]
        return path
    steps,effects=bound_file('steps.jsonl'),bound_file('server_outcomes.jsonl')
    with steps.open(encoding='utf-8-sig') as stream:
        endings=[s for s in map(json.loads,stream) if (s.get('next_observation') or {}).get('identity',{}).get('frame')==life['end_frame']]
    assert len(endings)==1;step=endings[0]
    before,after=step['observation'],step['next_observation'];identity,nxt=before['identity'],after['identity']
    assert step['terminal'] and not step['truncated'] and step['reason']=='observed_death'
    assert before['health']>0 and after['health']<=0 and after['health']==life['minimum_health']
    assert identity['life']==nxt['life']==1
    for key in ('map','spawncount','actor'):assert identity[key]==nxt[key]
    assert identity['map']==life['map'] and identity['spawncount']==life['spawncount'] and identity['actor']==life['bot_entity']
    native,dispatch=step['native_step'],step['server_execution']
    assert native['begin_frame']==identity['frame'] and native['end_frame']==nxt['frame']==life['end_frame']
    assert native['actor']==identity['actor'] and native['spawncount']==identity['spawncount'] and native['damage_indexes']
    assert dispatch['matched'] and dispatch['window_exclusive'] and dispatch['recovery_commands']==0
    with effects.open(encoding='utf-8-sig') as stream:
        matches=[v for v in map(json.loads,stream) if v['step']==step['index']]
    assert len(matches)==1;effect=matches[0]
    assert effect['version']=='server_step_effects_v1' and effect['available'] and effect['deaths']==1
    lethal=[v for v in effect['events'] if v['target']==identity['actor'] and v['target_class']=='player' and v['health_before']>0 and v['health_after']<=0]
    assert len(lethal)==1;event=lethal[0]
    assert event['map']==identity['map'] and event['spawncount']==identity['spawncount']
    assert native['begin_frame']<=event['frame']<=native['end_frame'] and event['health_after']==after['health']
    canonical=dict(version='combat_first_life_death_stop_v1',reason='combat_first_life_death',
        map=identity['map'],spawncount=identity['spawncount'],actor=identity['actor'],
        death_frame=event['frame'],observed_frame=nxt['frame'],health=int(after['health']))
    if result.get('death_stop') is not None:assert result['death_stop']==canonical,'Supervisor death contradicts native evidence'
    receipt['death_stop']=canonical
    proof['native_death']=dict(canonical=canonical,original_stop_tag_present=result.get('death_stop') is not None,
        native_report_sha256=sha(directory/'report.json'),steps_sha256=sha(steps),effects_sha256=sha(effects))
    return receipt,proof
